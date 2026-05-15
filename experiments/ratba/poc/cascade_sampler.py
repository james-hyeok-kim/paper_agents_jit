"""
RATBA STEP 2: Cascade Sampler PoC
==================================
- 초반 T_switch 비율 step: jit-b-32 (coarse, 64 tokens)
- 나머지 step: jit-b-16 (fine, 256 tokens)
- Handoff: flow matching이므로 단순히 z를 넘기면 됨 (DDIM σ re-noise 불필요)

T_switch 스윕: {0.3, 0.5, 0.7}
각 setting: 1000샘플, latency, avg_tokens, FID-proxy
"""
import sys
import os
import time
import copy
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

JIT_SRC = "/home/jovyan/workspace/Workspace_JiT"
sys.path.insert(0, JIT_SRC)
sys.path.insert(0, os.path.join(JIT_SRC, "src/torch-fidelity"))

from model_jit import JiT_models
from denoiser import Denoiser
import torch_fidelity


def load_jit_model(model_name, ckpt_path, device, cfg=3.0):
    class FakeArgs:
        pass
    args = FakeArgs()
    args.model           = model_name
    args.img_size        = 256
    args.in_channels     = 3
    args.class_num       = 1000
    args.attn_dropout    = 0.0
    args.proj_dropout    = 0.0
    args.label_drop_prob = 0.1
    args.P_mean          = -1.1
    args.P_std           = 2.0
    args.t_eps           = 5e-2
    args.noise_scale     = 1.0
    args.ema_decay1      = 0.9999
    args.ema_decay2      = 0.9999
    args.sampling_method = "euler"
    args.num_sampling_steps = 10
    args.cfg             = cfg
    args.interval_min    = 0.1
    args.interval_max    = 1.0

    model = Denoiser(args)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # EMA1 파라미터로 swap
    ema_state_dict = copy.deepcopy(checkpoint['model'])
    ema1_sd = checkpoint['model_ema1']
    for name in ema_state_dict.keys():
        if name in ema1_sd:
            ema_state_dict[name] = ema1_sd[name]

    # torch.compile 비활성화
    import types
    for block in model.net.blocks:
        if hasattr(block.forward, '__wrapped__'):
            raw_fn = block.forward.__wrapped__
            block.forward = types.MethodType(raw_fn, block)
    if hasattr(model.net.final_layer.forward, '__wrapped__'):
        raw_fn = model.net.final_layer.forward.__wrapped__
        model.net.final_layer.forward = types.MethodType(raw_fn, model.net.final_layer)

    model.load_state_dict(ema_state_dict, strict=False)
    model.eval()
    model.to(device)
    model.ema_params1 = list(model.parameters())
    model.ema_params2 = list(model.parameters())
    return model, args


class CascadeDenoiser(nn.Module):
    """
    두 JiT 모델을 cascade 방식으로 사용.
    - 초반 coarse_steps: coarse_model (b-32, 64 tokens)
    - 나머지 fine_steps: fine_model (b-16, 256 tokens)
    flow matching이므로 handoff는 단순 z 전달.
    """
    def __init__(self, coarse_model, fine_model, total_steps=10, t_switch_ratio=0.5,
                 cfg=3.0, interval_min=0.1, interval_max=1.0, t_eps=5e-2, noise_scale=1.0):
        super().__init__()
        self.coarse = coarse_model
        self.fine   = fine_model
        self.total_steps     = total_steps
        self.t_switch_ratio  = t_switch_ratio
        self.cfg             = cfg
        self.interval_min    = interval_min
        self.interval_max    = interval_max
        self.t_eps           = t_eps
        self.noise_scale     = noise_scale

        # coarse steps 수 (최소 1, 최대 total_steps-1)
        self.coarse_steps = max(1, min(int(total_steps * t_switch_ratio), total_steps - 1))
        self.fine_steps   = total_steps - self.coarse_steps

    def _forward_sample(self, model, z, t, labels):
        # conditional
        x_cond = model.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        # unconditional
        x_uncond = model.net(z, t.flatten(), torch.full_like(labels, model.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)
        # cfg interval
        interval_mask = (t < self.interval_max) & ((self.interval_min == 0) | (t > self.interval_min))
        cfg_scale_interval = torch.where(interval_mask, self.cfg, 1.0)
        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    def _euler_step(self, model, z, t, t_next, labels):
        v_pred = self._forward_sample(model, z, t, labels)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, 256, 256, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.total_steps + 1, device=device)
        timesteps = timesteps.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        # 코스 phase
        for i in range(self.coarse_steps):
            t      = timesteps[i]
            t_next = timesteps[i + 1]
            z = self._euler_step(self.coarse, z, t, t_next, labels)

        # 파인 phase (handoff: z 그대로 전달)
        for i in range(self.coarse_steps, self.total_steps - 1):
            t      = timesteps[i]
            t_next = timesteps[i + 1]
            z = self._euler_step(self.fine, z, t, t_next, labels)

        # 마지막 step
        z = self._euler_step(self.fine, z, timesteps[-2], timesteps[-1], labels)
        return z


def generate_and_save(model_or_cascade, num_images, batch_size, device, save_dir, seed=42):
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)

    class_num = 1000
    if num_images >= class_num:
        labels_all = np.arange(0, class_num).repeat(num_images // class_num)
    else:
        labels_all = np.arange(0, num_images)

    total = len(labels_all)
    t_start = time.perf_counter()
    generated = 0

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, total, batch_size):
                batch_labels = labels_all[i:i+batch_size]
                labels_t = torch.tensor(batch_labels, dtype=torch.long, device=device)

                imgs = model_or_cascade.generate(labels_t)
                imgs = (imgs + 1) / 2
                imgs = imgs.float().cpu()

                for b_id in range(imgs.size(0)):
                    img_np = np.round(
                        np.clip(imgs[b_id].numpy().transpose(1, 2, 0) * 255, 0, 255)
                    ).astype(np.uint8)
                    img_bgr = img_np[:, :, ::-1]
                    cv2.imwrite(os.path.join(save_dir, f"{i+b_id:05d}.png"), img_bgr)
                    generated += 1

                print(f"  [{generated}/{total}]", flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    return elapsed


def compute_fid(sample_dir, fid_stats_file):
    metrics = torch_fidelity.calculate_metrics(
        input1=sample_dir,
        input2=None,
        fid_statistics_file=fid_stats_file,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )
    return metrics['frechet_inception_distance'], metrics['inception_score_mean']


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    CKPT_BASE  = "/data/jameskimh/james_jit_pretrained/jit-h-16"
    FID_STATS  = os.path.join(JIT_SRC, "fid_stats/jit_in256_stats.npz")
    OUT_BASE   = "/data/jameskimh/ratba"
    POC_DIR    = "/home/jovyan/workspace/paper_agents_jit/experiments/ratba/poc"
    FIG_DIR    = "/home/jovyan/workspace/paper_agents_jit/experiments/ratba/figures"
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(POC_DIR, exist_ok=True)

    NUM_IMAGES = 1000
    BATCH_SIZE = 16
    CFG        = 3.0
    TOTAL_STEPS = 10

    # sanity_probe 결과 로드 (baseline 시간 참조)
    sanity_path = "/home/jovyan/workspace/paper_agents_jit/experiments/ratba/sanity_probe_results.json"
    with open(sanity_path) as f:
        sanity = json.load(f)
    baseline_elapsed = sanity["jit-b-16"]["elapsed_sec"]
    baseline_fid     = sanity["jit-b-16"]["fid"]
    baseline_tokens  = 256  # jit-b-16

    print(f"Baseline (jit-b-16): {baseline_elapsed:.1f}s, FID={baseline_fid:.4f}")

    # 모델 로드
    print("\njit-b-32 로드...")
    coarse_model, _ = load_jit_model(
        "JiT-B/32", os.path.join(CKPT_BASE, "jit-b-32/checkpoint-last.pth"), device, cfg=CFG
    )
    print("jit-b-16 로드...")
    fine_model, _ = load_jit_model(
        "JiT-B/16", os.path.join(CKPT_BASE, "jit-b-16/checkpoint-last.pth"), device, cfg=CFG
    )

    T_SWITCH_RATIOS = [0.3, 0.5, 0.7]
    results = {}

    for t_switch in T_SWITCH_RATIOS:
        print(f"\n{'='*60}")
        print(f"T_switch={t_switch}  (coarse steps: {int(TOTAL_STEPS*t_switch)}/{TOTAL_STEPS})")

        cascade = CascadeDenoiser(
            coarse_model, fine_model,
            total_steps=TOTAL_STEPS,
            t_switch_ratio=t_switch,
            cfg=CFG
        )
        cascade.eval()

        # avg_tokens 계산
        coarse_steps = cascade.coarse_steps
        fine_steps   = cascade.fine_steps
        avg_tokens   = (coarse_steps * 64 + fine_steps * 256) / TOTAL_STEPS
        tokens_ratio = avg_tokens / baseline_tokens  # vs baseline

        save_dir = os.path.join(OUT_BASE, f"poc_tswitch{t_switch:.1f}")
        print(f"샘플 생성 ({NUM_IMAGES}장, avg_tokens={avg_tokens:.1f})...")
        elapsed = generate_and_save(cascade, NUM_IMAGES, BATCH_SIZE, device, save_dir)

        sec_per_img = elapsed / NUM_IMAGES
        imgs_per_sec = NUM_IMAGES / elapsed
        speedup = baseline_elapsed / elapsed

        print(f"생성 완료: {elapsed:.1f}s ({imgs_per_sec:.2f} img/s, speedup={speedup:.3f}x)")

        print("FID 계산 중...")
        fid, is_score = compute_fid(save_dir, FID_STATS)
        fid_gap = fid - baseline_fid
        print(f"FID={fid:.4f} (gap vs baseline: {fid_gap:+.4f}), IS={is_score:.4f}")

        results[f"t_switch_{t_switch}"] = {
            "t_switch_ratio":  t_switch,
            "coarse_steps":    coarse_steps,
            "fine_steps":      fine_steps,
            "avg_tokens":      round(avg_tokens, 2),
            "tokens_ratio":    round(tokens_ratio, 4),
            "elapsed_sec":     round(elapsed, 2),
            "imgs_per_sec":    round(imgs_per_sec, 3),
            "ms_per_img":      round(sec_per_img * 1000, 2),
            "speedup":         round(speedup, 4),
            "fid":             round(fid, 4),
            "fid_gap":         round(fid_gap, 4),
            "inception_score": round(is_score, 4),
        }

    # ─── 성공 기준 분석 ───
    print(f"\n{'='*60}")
    print("CASCADE PoC 결과")
    print(f"{'='*60}")
    print(f"Baseline (jit-b-16): elapsed={baseline_elapsed:.1f}s, FID={baseline_fid:.4f}")
    print()

    best_setting = None
    for key, v in results.items():
        tokens_ok = v["tokens_ratio"] <= 0.7
        fid_ok    = v["fid_gap"] <= 2.0
        go        = tokens_ok and fid_ok
        print(f"T_switch={v['t_switch_ratio']}: speedup={v['speedup']:.3f}x, "
              f"tokens_ratio={v['tokens_ratio']:.3f}, FID={v['fid']:.4f} (gap={v['fid_gap']:+.4f}) "
              f"[tokens_ok={tokens_ok}, fid_ok={fid_ok}] => {'GO' if go else 'NO-GO'}")
        if go and best_setting is None:
            best_setting = v

    verdict = "GO" if best_setting is not None else "NO-GO"
    print(f"\nVERDICT: {verdict}")
    if best_setting:
        print(f"최선 설정: T_switch={best_setting['t_switch_ratio']}, "
              f"speedup={best_setting['speedup']:.3f}x, FID gap={best_setting['fid_gap']:+.4f}")

    results["baseline"] = {
        "model": "jit-b-16",
        "fid": baseline_fid,
        "elapsed_sec": baseline_elapsed,
        "avg_tokens": 256,
        "tokens_ratio": 1.0,
    }
    results["analysis"] = {
        "verdict": verdict,
        "best_t_switch": best_setting["t_switch_ratio"] if best_setting else None,
    }

    out_json = os.path.join(POC_DIR, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n결과 저장: {out_json}")

    # ─── 그래프 ───
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    t_switches = [v["t_switch_ratio"] for v in results.values() if "t_switch_ratio" in v]
    fids       = [v["fid"]            for v in results.values() if "t_switch_ratio" in v]
    tokens     = [v["avg_tokens"]     for v in results.values() if "t_switch_ratio" in v]
    speedups   = [v["speedup"]        for v in results.values() if "t_switch_ratio" in v]

    # FID vs avg_tokens
    ax = axes[0]
    ax.scatter(tokens, fids, c='blue', s=100, zorder=5, label='Cascade')
    ax.axhline(baseline_fid, color='red', linestyle='--', label=f'Baseline FID={baseline_fid:.2f}')
    ax.axvline(baseline_tokens * 0.7, color='gray', linestyle=':', label='70% token budget')
    ax.axhline(baseline_fid + 2.0, color='orange', linestyle=':', label='FID+2 threshold')
    for i, (x, y, ts) in enumerate(zip(tokens, fids, t_switches)):
        ax.annotate(f"T={ts}", (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("avg tokens per step")
    ax.set_ylabel("FID")
    ax.set_title("FID vs avg_tokens (RATBA Cascade PoC)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Speedup vs T_switch
    ax = axes[1]
    ax.plot(t_switches, speedups, 'bo-', markersize=8, label='Cascade speedup')
    ax.axhline(1.0, color='red', linestyle='--', label='Baseline')
    ax.axhline(1.5, color='green', linestyle=':', label='1.5x target')
    ax.set_xlabel("T_switch ratio (coarse fraction)")
    ax.set_ylabel("Speedup vs jit-b-16")
    ax.set_title("Speedup vs T_switch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "poc_tradeoff.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"그래프 저장: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
