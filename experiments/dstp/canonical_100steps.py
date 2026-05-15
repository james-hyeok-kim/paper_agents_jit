"""
100-step canonical setting: baseline vs K3_tbased.
- 256장, batch=16, 디스크 I/O 없음 (timing 격리)
- baseline과 K3_tbased를 alternating batch로 인터리빙해 GPU contention을 균등화
"""
import sys, os, time, json
import numpy as np
import torch

PIXELDIT_SRC = "/home/jovyan/workspace/Workspace_PixelDiT"
sys.path.insert(0, PIXELDIT_SRC)
sys.path.insert(0, "/home/jovyan/workspace/paper_agents_jit/experiments/dstp")

from dstp_sampler import (
    load_pixeldit_xl, make_sampler, StepSkipPixDiT,
)


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    CKPT = "/data/jameskimh/pixeldit_pretrained/imagenet256_pixeldit_xl_epoch320.ckpt"

    print("모델 로딩...")
    base = load_pixeldit_xl(CKPT, device)
    skip = StepSkipPixDiT(base, K=3, refresh_policy='t_based',
                          T_split=0.5, K_high=3)
    skip.eval()
    print("로딩 완료\n")

    BATCH = 16
    NSTEPS = 100
    N = 256
    n_batches = N // BATCH

    sampler = make_sampler(num_steps=NSTEPS)

    # 워밍업
    print(f"워밍업 (NSTEPS={NSTEPS})...")
    _n = torch.randn(BATCH, 3, 256, 256, device=device)
    _l = torch.zeros(BATCH, dtype=torch.long, device=device)
    _u = torch.full((BATCH,), 1000, dtype=torch.long, device=device)
    for _ in range(2):
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = sampler(base, _n, _l, _u)
    torch.cuda.synchronize()
    del _n, _l, _u
    print("워밍업 완료\n")

    # Alternating: baseline / K3 / baseline / K3 ...
    times_base = []
    times_skip = []
    torch.manual_seed(42)
    print(f"=== 인터리빙 측정 ({n_batches} 배치 × 2 = {2*n_batches} forward) ===")
    with torch.no_grad():
        for i in range(n_batches):
            labels_t = torch.randint(0, 1000, (BATCH,), device=device)
            null_labels = torch.full((BATCH,), 1000, dtype=torch.long, device=device)
            noise = torch.randn(BATCH, 3, 256, 256, device=device)

            # baseline
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = sampler(base, noise, labels_t, null_labels)
            torch.cuda.synchronize()
            tb = (time.perf_counter() - t0) / BATCH * 1000
            times_base.append(tb)

            # K3_tbased (같은 noise/label로 측정)
            skip.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = sampler(skip, noise, labels_t, null_labels)
            torch.cuda.synchronize()
            ts = (time.perf_counter() - t0) / BATCH * 1000
            times_skip.append(ts)

            print(f"  [{(i+1)*BATCH}/{N}] base={tb:.1f}ms  K3={ts:.1f}ms  speedup={tb/ts:.3f}x", flush=True)

    # 초반 1/3 warmup으로 제외
    cut = len(times_base) // 3
    base_stable = times_base[cut:]
    skip_stable = times_skip[cut:]

    base_med = float(np.median(base_stable))
    skip_med = float(np.median(skip_stable))
    base_mean = float(np.mean(base_stable))
    skip_mean = float(np.mean(skip_stable))

    # paired speedup (각 배치별 baseline/K3 비율의 median - 외부 contention에 더 robust)
    paired = [b/s for b, s in zip(base_stable, skip_stable)]
    paired_med = float(np.median(paired))

    results = {
        "meta": {
            "batch": BATCH,
            "num_steps": NSTEPS,
            "n_per_config": N,
            "method": "interleaved baseline/K3 forward, paired speedup",
        },
        "baseline": {
            "median_ms": round(base_med, 2),
            "mean_ms": round(base_mean, 2),
            "std_ms": round(float(np.std(base_stable)), 2),
        },
        "K3_tbased": {
            "median_ms": round(skip_med, 2),
            "mean_ms": round(skip_mean, 2),
            "std_ms": round(float(np.std(skip_stable)), 2),
        },
        "speedup_median_of_pairs": round(paired_med, 4),
        "speedup_median_of_medians": round(base_med / skip_med, 4),
    }

    out = "/home/jovyan/workspace/paper_agents_jit/experiments/dstp/canonical_100steps.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n저장: {out}")
    print("="*60)
    print(f"100-step canonical (paired):")
    print(f"  baseline median: {base_med:.2f} ms/img (mean {base_mean:.2f}, std {results['baseline']['std_ms']})")
    print(f"  K3_tbased median: {skip_med:.2f} ms/img (mean {skip_mean:.2f}, std {results['K3_tbased']['std_ms']})")
    print(f"  speedup (median of pairs): {paired_med:.3f}x")
    print(f"  speedup (median/median):   {base_med/skip_med:.3f}x")


if __name__ == "__main__":
    main()
