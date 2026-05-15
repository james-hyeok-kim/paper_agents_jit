"""
JiT Step-Skip 캐싱 적용 (일반화 검증용).

JiT-B/16 (12 blocks) 또는 JiT-H/16 (32 blocks) 대상.
PixelDiT는 patch_blocks/pixel_blocks 분할이지만, JiT는 monolithic trunk.
두 가지 캐싱 모드:
  - 'full': 매 K번째 step에서만 net forward, 나머지는 cached output 재사용
  - 'partial': 첫 N개 block의 output을 캐싱, 마지막 block들만 매 step 실행

PoC: K=3, t_based (T_split=0.5)
"""
import sys, os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn

JIT_SRC = "/home/jovyan/workspace/Workspace_JiT"
sys.path.insert(0, JIT_SRC)

from model_jit import JiT_models


# ────────────────────────────────────────
# 모델 로딩 (denoiser 없이 raw JiT 직접 사용)
# ────────────────────────────────────────
def load_jit(model_name, ckpt_path, device, img_size=256, num_classes=1000):
    print(f"loading {model_name}...")
    net = JiT_models[model_name](
        input_size=img_size,
        in_channels=3,
        num_classes=num_classes,
        attn_drop=0.0,
        proj_drop=0.0,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
    # ema 우선 (있으면)
    if any(k.startswith('ema_params2.') for k in sd.keys()):
        # ema_params2가 list면 dict로 매핑 어려움; checkpoint 구조에 따라 분기 필요
        pass
    # denoiser.net.* 형태인지 확인
    cleaned = {}
    for k, v in sd.items():
        if k.startswith('net.'):
            cleaned[k[4:]] = v
        elif k.startswith('module.net.'):
            cleaned[k[len('module.net.'):]] = v
        elif k.startswith('module.'):
            cleaned[k[len('module.'):]] = v
        else:
            cleaned[k] = v
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[경고] missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[경고] unexpected ({len(unexpected)}): {unexpected[:5]}")
    net.eval().to(device)
    return net


# ────────────────────────────────────────
# Step-Skip wrapper for JiT (full output caching)
# ────────────────────────────────────────
class JiTStepSkip(nn.Module):
    """JiT 전체 forward 출력을 K step마다 캐싱."""
    def __init__(self, net, K=3, refresh_policy='t_based',
                 T_split=0.5, K_high=3):
        super().__init__()
        self.net = net
        self.K = K
        self.refresh_policy = refresh_policy
        self.T_split = T_split
        self.K_high = K_high
        self._cache = None
        self._step_count = 0

    def reset(self):
        self._cache = None
        self._step_count = 0

    def _should_refresh(self, t_val):
        if self._cache is None:
            return True
        if self.refresh_policy == 'periodic':
            return (self._step_count % self.K == 0)
        elif self.refresh_policy == 't_based':
            # JiT: t=0 노이즈, t=1 클린 (sampling: t 0->1 increasing)
            # 고노이즈(t < T_split)에서 skip, 저노이즈(t >= T_split)에서 refresh always
            if t_val < self.T_split:
                return (self._step_count % self.K_high == 0)  # 고노이즈 → skip
            else:
                return True  # 저노이즈 → refresh always
        return True

    def forward(self, x, t, y):
        t_val = float(t.flatten()[0].item())
        if self._should_refresh(t_val):
            out = self.net(x, t, y)
            self._cache = out.detach()
        else:
            out = self._cache
        self._step_count += 1
        return out


# ────────────────────────────────────────
# 간단 ODE sampler (Euler + CFG)
# ────────────────────────────────────────
@torch.no_grad()
def euler_sample(model_fn, noise, labels, num_classes,
                 num_steps=50, cfg_scale=3.0,
                 cfg_interval=(0.1, 1.0),
                 t_eps=1e-3, noise_scale=1.0,
                 reset_fn=None):
    """JiT 호환 Euler ODE + CFG."""
    device = noise.device
    bsz = labels.size(0)
    z = noise * noise_scale

    if reset_fn is not None:
        reset_fn()

    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    null_labels = torch.full_like(labels, num_classes)

    for i in range(num_steps):
        t = torch.full((bsz,), float(timesteps[i]), device=device)
        t_next = float(timesteps[i + 1])

        # batched: [cond | uncond] 한 번에 forward
        z_cat = torch.cat([z, z], dim=0)
        t_cat = torch.cat([t, t], dim=0)
        y_cat = torch.cat([labels, null_labels], dim=0)
        x_cat = model_fn(z_cat, t_cat, y_cat)
        x_cond, x_uncond = x_cat.chunk(2, dim=0)

        denom = (1.0 - t.view(-1, 1, 1, 1)).clamp_min(t_eps)
        v_cond = (x_cond - z) / denom
        v_uncond = (x_uncond - z) / denom

        # CFG interval
        low, high = cfg_interval
        t_val = float(t[0].item())
        if low <= t_val <= high:
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v_pred = v_cond
        z = z + (t_next - float(timesteps[i])) * v_pred

    return z


# ────────────────────────────────────────
# 메인
# ────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="JiT-B/16",
                        choices=["JiT-B/16","JiT-B/32","JiT-L/16","JiT-L/32","JiT-H/16","JiT-H/32"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # 체크포인트 자동 매핑
    if args.ckpt is None:
        size_tag = args.model.replace("JiT-", "jit-").replace("/", "-").lower()
        args.ckpt = f"/data/jameskimh/james_jit_pretrained/jit-h-16/{size_tag}/checkpoint-last.pth"

    out_dir = args.out or f"/data/jameskimh/jit_dstp/{args.model.replace('/', '_')}"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda")
    print(f"Model={args.model}  Ckpt={args.ckpt}\nGPU={torch.cuda.get_device_name(0)}\n")

    net = load_jit(args.model, args.ckpt, device, img_size=256, num_classes=1000)

    # 워밍업
    print("워밍업...")
    _n = torch.randn(args.batch, 3, 256, 256, device=device)
    _l = torch.randint(0, 1000, (args.batch,), device=device)
    for _ in range(2):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = euler_sample(net, _n, _l, 1000, num_steps=args.steps, cfg_scale=args.cfg)
    torch.cuda.synchronize()
    del _n, _l
    print("워밍업 완료\n")

    configs = [
        ("baseline", None),
        ("K3_tbased", JiTStepSkip(net, K=3, refresh_policy='t_based', T_split=0.5, K_high=3)),
        ("K3_periodic", JiTStepSkip(net, K=3, refresh_policy='periodic', T_split=0.5, K_high=3)),
    ]

    results = {"meta": {"model": args.model, "steps": args.steps, "batch": args.batch,
                        "n_per_config": args.n, "cfg": args.cfg}}
    base_med = None

    n_batches = args.n // args.batch
    torch.manual_seed(42)

    for tag, skip_module in configs:
        print(f"=== {tag} ===")
        if skip_module is not None:
            skip_module.eval()
            model_fn = skip_module
            reset_fn = skip_module.reset
        else:
            model_fn = net
            reset_fn = None

        save_dir = os.path.join(out_dir, tag)
        os.makedirs(save_dir, exist_ok=True)

        times = []
        with torch.no_grad():
            for i in range(n_batches):
                labels = torch.randint(0, 1000, (args.batch,), device=device)
                noise = torch.randn(args.batch, 3, 256, 256, device=device)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    imgs = euler_sample(model_fn, noise, labels, 1000,
                                        num_steps=args.steps, cfg_scale=args.cfg,
                                        reset_fn=reset_fn)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) / args.batch * 1000
                times.append(elapsed)
                print(f"  [{(i+1)*args.batch}/{args.n}] {elapsed:.1f}ms/img", flush=True)

                # 첫 배치만 저장 (시각 검증용)
                if i == 0:
                    import cv2
                    imgs = (imgs.float().cpu() + 1) / 2
                    imgs = imgs.clamp(0, 1)
                    for b_id in range(args.batch):
                        arr = np.round(imgs[b_id].numpy().transpose(1,2,0) * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(save_dir, f"{b_id:03d}_cls{int(labels[b_id]):04d}.png"),
                                    arr[:, :, ::-1])

        stable = times[len(times)//3:]
        med = float(np.median(stable))
        if tag == "baseline":
            base_med = med
            speedup = 1.0
        else:
            speedup = base_med / med

        results[tag] = {
            "median_ms": round(med, 2),
            "mean_ms": round(float(np.mean(stable)), 2),
            "std_ms": round(float(np.std(stable)), 2),
            "speedup": round(speedup, 4),
        }
        print(f"  {tag}: median={med:.2f}ms speedup={speedup:.3f}x\n")

    out_json = "/home/jovyan/workspace/paper_agents_jit/experiments/jit_dstp/results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n저장: {out_json}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
