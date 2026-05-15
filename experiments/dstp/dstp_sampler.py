"""
DSTP: Dynamic Step-wise Token Pruning (revised → Step-Skip)
=============================================================

핵심 아이디어 (v2: step-skip):
- patch_blocks(26개)이 전체 연산의 ~87%를 차지
- 매 K번째 denoising step에서만 patch_blocks를 실행하고
  나머지 step은 cached s 벡터를 재사용
- 오버헤드 없음: gather/scatter 불필요

초기 token-pruning 시도 결과:
- gather+scatter+topk overhead > attention savings → 0.785× (느려짐)
- step-skip으로 피벗

Step-Skip 결과 (예비 타이밍, B=8, 20 steps, B200):
- K=2: 1.30× speedup
- K=3: 2.03× speedup

실험 설정:
- 4개 설정: K × refresh_policy = {2, 3} × {periodic, t-based}
- 각 설정: 128장 생성
- 측정: ms/img speedup, IS (FID는 reference set 없어 보류)

refresh_policy:
- periodic: 매 K번째 step마다 patch refresh (단순 카운터)
- t_based: t > T_split 구간(고노이즈)은 every-other, t <= T_split은 every-step
"""

import sys
import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn

# PixelDiT 소스 경로
PIXELDIT_SRC = "/home/jovyan/workspace/Workspace_PixelDiT"
sys.path.insert(0, PIXELDIT_SRC)

from pixdit_core.pixeldit_c2i import PixDiT
from c2i.src.diffusion import (
    FlowDPMSolverSampler, LinearScheduler,
    simple_guidance_fn, ode_step_fn
)


# ──────────────────────────────────────────────────────────────
# 1. 모델 로딩 헬퍼
# ──────────────────────────────────────────────────────────────
def load_pixeldit_xl(ckpt_path, device):
    """
    Lightning checkpoint에서 ema_denoiser 가중치 로드
    patch_size=16, patch_depth=26, pixel_depth=4, hidden=1152
    """
    model = PixDiT(
        in_channels=3,
        patch_size=16,
        num_groups=16,
        hidden_size=1152,
        patch_depth=26,
        pixel_depth=4,
        num_classes=1000,
        pixel_hidden_size=16,
        use_pixel_abs_pos=True,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']

    # ema_denoiser. 접두사 제거
    ema_sd = {}
    for k, v in sd.items():
        if k.startswith('ema_denoiser.'):
            ema_sd[k[len('ema_denoiser.'):]] = v

    missing, unexpected = model.load_state_dict(ema_sd, strict=True)
    if missing:
        print(f"[경고] missing keys: {missing[:5]}")
    if unexpected:
        print(f"[경고] unexpected keys: {unexpected[:5]}")

    model.eval()
    model.to(device)
    return model


# ──────────────────────────────────────────────────────────────
# 2. 표준 FlowDPM 샘플러
# ──────────────────────────────────────────────────────────────
def make_sampler(num_steps=20):
    sampler = FlowDPMSolverSampler(
        scheduler=LinearScheduler(),
        w_scheduler=None,
        guidance_fn=simple_guidance_fn,
        num_steps=num_steps,
        guidance=3.25,
        timeshift=1.0,
        guidance_interval_min=0.1,
        guidance_interval_max=1.0,
        step_fn=ode_step_fn,
    )
    return sampler


# ──────────────────────────────────────────────────────────────
# 3. Step-Skip PixDiT (patch_blocks 스킵)
# ──────────────────────────────────────────────────────────────
class StepSkipPixDiT(nn.Module):
    """
    K-step skip: patch_blocks를 매 K번째 step에서만 실행.
    나머지 step은 cached s를 재사용.

    refresh_policy:
    - 'periodic': 단순 카운터 (step_count % K == 0 이면 refresh)
    - 't_based':  t > T_split 구간은 K=K_high (보통 2),
                  t <= T_split 구간은 K=1 (매 step refresh)
    """
    def __init__(self, base_model, K=2,
                 refresh_policy='periodic',
                 T_split=0.5, K_high=2):
        super().__init__()
        self.net = base_model
        self.K = K
        self.refresh_policy = refresh_policy
        self.T_split = T_split
        self.K_high = K_high
        self._s_cache = None
        self._step_count = 0

    def reset(self):
        self._s_cache = None
        self._step_count = 0

    def _should_refresh(self, t_val):
        if self._s_cache is None:
            return True
        if self.refresh_policy == 'periodic':
            return (self._step_count % self.K == 0)
        elif self.refresh_policy == 't_based':
            # 고노이즈(t < T_split)는 K_high skip, 저노이즈는 매 step 갱신
            if t_val > self.T_split:
                return (self._step_count % self.K_high == 0)
            else:
                return True
        return True

    def forward(self, x, t, y, s=None, mask=None):
        net = self.net
        B, _, H, W = x.shape
        pos = net.fetch_pos(H // net.patch_size, W // net.patch_size, x.device)
        x_patches = torch.nn.functional.unfold(
            x, kernel_size=net.patch_size, stride=net.patch_size
        ).transpose(1, 2)
        t_emb = net.t_embedder(t.view(-1)).view(B, -1, net.hidden_size)
        y_emb = net.y_embedder(y).view(B, 1, net.hidden_size)
        c = nn.functional.silu(t_emb + y_emb)

        # timestep 값 (CFG 배치: [uncond, cond] 순서이므로 batch 중간 값)
        t_val = t[0].item()

        if s is None:
            if self._should_refresh(t_val):
                s = net.s_embedder(x_patches)
                for block in net.patch_blocks:
                    s = block(s, c, pos, mask)
                self._s_cache = s.detach()
            else:
                s = self._s_cache
            self._step_count += 1
            s = nn.functional.silu(t_emb + s)

        batch_size, length, _ = s.shape
        s_cond = s.view(batch_size * length, net.hidden_size)
        x_pixels = net.pixel_embedder(x, img_height=H, img_width=W, patch_size=net.patch_size)
        for blk in net.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, net.patch_size, mask)
        x_pixels = net.final_layer(x_pixels)
        C_out = net.out_channels
        P2 = net.patch_size * net.patch_size
        x_pixels = x_pixels.view(B, length, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, length)
        x_img = torch.nn.functional.fold(
            x_pixels, (H, W), kernel_size=net.patch_size, stride=net.patch_size
        )
        return x_img


# ──────────────────────────────────────────────────────────────
# 4. 이미지 생성 함수 (타이밍 포함)
# ──────────────────────────────────────────────────────────────
def generate_samples(model_fn, num_images, batch_size, device,
                     save_dir, num_steps=20, seed=42,
                     reset_fn=None):
    """
    model_fn: PixDiT 또는 StepSkipPixDiT
    reset_fn: 각 배치 전 호출되는 리셋 함수 (step-skip cache 초기화)
    """
    import cv2
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)

    class_num = 1000
    if num_images >= class_num:
        labels_all = np.tile(np.arange(class_num), num_images // class_num)
    else:
        labels_all = np.arange(num_images)
    labels_all = labels_all[:num_images]

    sampler = make_sampler(num_steps=num_steps)
    total = len(labels_all)
    generated = 0
    times_ms = []

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_labels = labels_all[i:i + batch_size]
            B = len(batch_labels)

            labels_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
            null_labels = torch.full((B,), 1000, dtype=torch.long, device=device)
            noise = torch.randn(B, 3, 256, 256, device=device)

            if reset_fn is not None:
                reset_fn()

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                imgs = sampler(model_fn, noise, labels_t, null_labels)

            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) / B * 1000
            times_ms.append(elapsed_ms)

            imgs = (imgs.float().cpu() + 1) / 2
            imgs = imgs.clamp(0, 1)

            for b_id in range(B):
                img_np = np.round(
                    imgs[b_id].numpy().transpose(1, 2, 0) * 255
                ).astype(np.uint8)
                img_bgr = img_np[:, :, ::-1]
                img_id = i + b_id
                cv2.imwrite(os.path.join(save_dir, f"{img_id:05d}.png"), img_bgr)
                generated += 1

            if generated % 32 == 0 or generated == total:
                print(f"  [{generated}/{total}] {elapsed_ms:.1f} ms/img", flush=True)

    return times_ms  # ms/img per batch


# ──────────────────────────────────────────────────────────────
# 5. IS 계산 (FID는 reference 없으므로 보류)
# ──────────────────────────────────────────────────────────────
def compute_is(sample_dir):
    """torch_fidelity로 IS만 계산 (reference 불필요)"""
    import torch_fidelity
    metrics = torch_fidelity.calculate_metrics(
        input1=sample_dir,
        input2=None,
        cuda=True,
        isc=True,
        fid=False,
        kid=False,
        verbose=False,
    )
    return metrics['inception_score_mean']


# ──────────────────────────────────────────────────────────────
# 6. 메인 실험
# ──────────────────────────────────────────────────────────────
def main():
    import shutil
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    CKPT = "/data/jameskimh/pixeldit_pretrained/imagenet256_pixeldit_xl_epoch320.ckpt"
    OUT_BASE = "/data/jameskimh/dstp"
    NUM_IMAGES = 128
    BATCH_SIZE = 8
    NUM_STEPS = 20   # PoC용 (canonical = 100)

    print("모델 로딩 중...")
    base_model = load_pixeldit_xl(CKPT, device)
    print("모델 로딩 완료")

    # GPU 워밍업
    _noise = torch.randn(BATCH_SIZE, 3, 256, 256, device=device)
    _labels = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
    _null = torch.full((BATCH_SIZE,), 1000, dtype=torch.long, device=device)
    _sampler = make_sampler(num_steps=NUM_STEPS)
    print("GPU 워밍업 중...")
    for _ in range(3):
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = _sampler(base_model, _noise, _labels, _null)
    torch.cuda.synchronize()
    print("워밍업 완료")
    del _noise, _labels, _null, _sampler

    results = {}

    # ── 베이스라인 ──
    print("\n=== BASELINE ===")
    baseline_dir = os.path.join(OUT_BASE, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    if len(os.listdir(baseline_dir)) >= NUM_IMAGES:
        print(f"기존 샘플 재사용: {baseline_dir}")
        tmp_dir = baseline_dir + "_timing"
        times_ms = generate_samples(
            base_model, 32, BATCH_SIZE, device, tmp_dir,
            num_steps=NUM_STEPS, seed=0
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        times_ms = generate_samples(
            base_model, NUM_IMAGES, BATCH_SIZE, device, baseline_dir,
            num_steps=NUM_STEPS, seed=0
        )

    # 초반 몇 배치는 compile jit 포함될 수 있으므로 후반 2/3만 사용
    stable = times_ms[len(times_ms)//3:]
    ms_base = float(np.median(stable))
    ms_std_base = float(np.std(stable))
    print(f"베이스라인: {ms_base:.1f} ms/img (median, std={ms_std_base:.1f})")

    is_base = compute_is(baseline_dir)
    print(f"IS (baseline): {is_base:.3f}")

    results["baseline"] = {
        "ms_per_img_median": round(ms_base, 2),
        "ms_std": round(ms_std_base, 2),
        "IS": round(float(is_base), 4),
        "num_images": NUM_IMAGES,
        "num_steps": NUM_STEPS,
    }

    # ── Step-Skip 설정 스윕 ──
    configs = [
        {"K": 2, "policy": "periodic", "desc": "K2_periodic"},
        {"K": 3, "policy": "periodic", "desc": "K3_periodic"},
        {"K": 2, "policy": "t_based",  "desc": "K2_tbased", "T_split": 0.5},
        {"K": 3, "policy": "t_based",  "desc": "K3_tbased", "T_split": 0.5},
    ]

    for cfg in configs:
        key = cfg["desc"]
        K = cfg["K"]
        policy = cfg["policy"]
        T_split = cfg.get("T_split", 0.5)

        print(f"\n=== Step-Skip: K={K}, policy={policy} ===")
        skip_model = StepSkipPixDiT(
            base_model, K=K,
            refresh_policy=policy,
            T_split=T_split, K_high=K
        )
        skip_model.eval()

        sample_dir = os.path.join(OUT_BASE, key)
        os.makedirs(sample_dir, exist_ok=True)

        if len(os.listdir(sample_dir)) >= NUM_IMAGES:
            print(f"기존 샘플 재사용: {sample_dir}")
            tmp_dir = sample_dir + "_timing"
            times_ms = generate_samples(
                skip_model, 32, BATCH_SIZE, device, tmp_dir,
                num_steps=NUM_STEPS, seed=0,
                reset_fn=skip_model.reset
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            times_ms = generate_samples(
                skip_model, NUM_IMAGES, BATCH_SIZE, device, sample_dir,
                num_steps=NUM_STEPS, seed=0,
                reset_fn=skip_model.reset
            )

        stable = times_ms[len(times_ms)//3:]
        ms_skip = float(np.median(stable))
        ms_std_skip = float(np.std(stable))
        speedup = ms_base / ms_skip
        print(f"{key}: {ms_skip:.1f} ms/img (median) | speedup: {speedup:.3f}x")

        is_val = float(compute_is(sample_dir))
        print(f"IS: {is_val:.3f}  (baseline: {is_base:.3f}, drop: {is_base - is_val:.3f})")

        results[key] = {
            "K": K,
            "policy": policy,
            "T_split": T_split,
            "ms_per_img_median": round(ms_skip, 2),
            "ms_std": round(ms_std_skip, 2),
            "IS": round(is_val, 4),
            "speedup": round(speedup, 4),
            "IS_drop": round(float(is_base) - is_val, 4),
        }
        del skip_model

    # ── 결과 요약 ──
    print(f"\n{'='*70}")
    print("DSTP STEP-SKIP RESULTS")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'ms/img':>10} {'speedup':>10} {'IS':>8} {'IS drop':>10}")
    print("-" * 70)
    b = results["baseline"]
    print(f"{'baseline':<25} {b['ms_per_img_median']:>10.1f} {'1.000x':>10} {b['IS']:>8.3f} {'0.000':>10}")
    for cfg in configs:
        key = cfg["desc"]
        r = results[key]
        print(f"{key:<25} {r['ms_per_img_median']:>10.1f} {r['speedup']:>9.3f}x {r['IS']:>8.3f} {r['IS_drop']:>10.3f}")

    # GO/NO-GO 판정
    print(f"\n판정 기준: speedup >= 1.3x AND IS_drop <= 5.0")
    any_go = False
    for cfg in configs:
        key = cfg["desc"]
        r = results[key]
        go = r["speedup"] >= 1.3 and abs(r["IS_drop"]) <= 5.0
        verdict = "GO" if go else "NO-GO"
        if go:
            any_go = True
        print(f"  {key}: {verdict}  (speedup={r['speedup']:.3f}x, IS_drop={r['IS_drop']:.3f})")

    results["verdict"] = "GO" if any_go else "NO-GO"
    print(f"\n최종 VERDICT: {results['verdict']}")

    out_json = "/home/jovyan/workspace/paper_agents_jit/experiments/dstp/results.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n결과 저장: {out_json}")


if __name__ == "__main__":
    main()
