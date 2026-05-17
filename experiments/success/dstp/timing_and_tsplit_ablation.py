"""
Timing 격리 재측정 + T_split ablation.
- 디스크 I/O 제거: 이미지 저장 안 함
- 각 설정 256장, batch=16 (16 배치)
- 워밍업 후 timing만 측정
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


def time_config(model_fn, n_total, batch_size, device, num_steps, reset_fn=None, tag=""):
    n_batches = n_total // batch_size
    sampler = make_sampler(num_steps=num_steps)
    times_ms = []
    torch.manual_seed(42)
    with torch.no_grad():
        for i in range(n_batches):
            labels_t = torch.randint(0, 1000, (batch_size,), device=device)
            null_labels = torch.full((batch_size,), 1000, dtype=torch.long, device=device)
            noise = torch.randn(batch_size, 3, 256, 256, device=device)
            if reset_fn is not None:
                reset_fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = sampler(model_fn, noise, labels_t, null_labels)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) / batch_size * 1000
            times_ms.append(elapsed_ms)
            if (i + 1) % 4 == 0:
                print(f"  {tag} [{(i+1)*batch_size}/{n_total}] last_batch={elapsed_ms:.1f}ms/img", flush=True)
    # 초반 1/3 warmup으로 제외
    stable = times_ms[len(times_ms)//3:]
    return {
        "median_ms": float(np.median(stable)),
        "mean_ms":   float(np.mean(stable)),
        "std_ms":    float(np.std(stable)),
        "min_ms":    float(np.min(stable)),
    }


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    CKPT = "/data/jameskimh/pixeldit_pretrained/imagenet256_pixeldit_xl_epoch320.ckpt"

    print("모델 로딩...")
    base = load_pixeldit_xl(CKPT, device)
    print("로딩 완료")

    BATCH = 16
    NSTEPS = 20
    N = 256

    # 워밍업 (compile JIT)
    print("\n워밍업...")
    _n = torch.randn(BATCH, 3, 256, 256, device=device)
    _l = torch.zeros(BATCH, dtype=torch.long, device=device)
    _u = torch.full((BATCH,), 1000, dtype=torch.long, device=device)
    _s = make_sampler(num_steps=NSTEPS)
    for _ in range(3):
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = _s(base, _n, _l, _u)
    torch.cuda.synchronize()
    del _n, _l, _u, _s
    print("워밍업 완료\n")

    results = {"meta": {"batch": BATCH, "num_steps": NSTEPS, "n_per_config": N}}

    # 1) Baseline
    print("=== Baseline ===")
    results["baseline"] = time_config(base, N, BATCH, device, NSTEPS, tag="[BASE]")
    base_med = results["baseline"]["median_ms"]
    print(f"  baseline median: {base_med:.2f} ms/img\n")

    # 2) T_split sweep (K=3, t_based)
    configs = [
        {"K": 3, "policy": "t_based", "T_split": 0.3, "tag": "K3_t0.3"},
        {"K": 3, "policy": "t_based", "T_split": 0.4, "tag": "K3_t0.4"},
        {"K": 3, "policy": "t_based", "T_split": 0.5, "tag": "K3_t0.5"},
        {"K": 3, "policy": "t_based", "T_split": 0.6, "tag": "K3_t0.6"},
        {"K": 3, "policy": "t_based", "T_split": 0.7, "tag": "K3_t0.7"},
        # 비교용 periodic (단순)
        {"K": 3, "policy": "periodic", "T_split": 0.5, "tag": "K3_periodic"},
    ]

    for cfg in configs:
        tag = cfg["tag"]
        print(f"=== {tag} (K={cfg['K']}, policy={cfg['policy']}, T_split={cfg['T_split']}) ===")
        skip = StepSkipPixDiT(
            base, K=cfg["K"], refresh_policy=cfg["policy"],
            T_split=cfg["T_split"], K_high=cfg["K"],
        )
        skip.eval()
        timing = time_config(skip, N, BATCH, device, NSTEPS,
                             reset_fn=skip.reset, tag=f"[{tag}]")
        speedup = base_med / timing["median_ms"]
        timing["speedup"] = speedup
        timing["K"] = cfg["K"]
        timing["policy"] = cfg["policy"]
        timing["T_split"] = cfg["T_split"]
        results[tag] = timing
        print(f"  {tag} median: {timing['median_ms']:.2f} ms/img | speedup: {speedup:.3f}x\n")
        del skip
        torch.cuda.empty_cache()

    # 결과 저장
    out = "/home/jovyan/workspace/paper_agents_jit/experiments/dstp/tsplit_ablation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"저장: {out}")

    # 요약 출력
    print("\n" + "="*70)
    print("T_split ABLATION (clean timing, no disk I/O)")
    print("="*70)
    print(f"{'Config':<15} {'median (ms)':>12} {'mean (ms)':>12} {'std':>8} {'speedup':>10}")
    print("-"*70)
    b = results["baseline"]
    print(f"{'baseline':<15} {b['median_ms']:>12.2f} {b['mean_ms']:>12.2f} {b['std_ms']:>8.2f} {'1.000x':>10}")
    for cfg in configs:
        r = results[cfg["tag"]]
        print(f"{cfg['tag']:<15} {r['median_ms']:>12.2f} {r['mean_ms']:>12.2f} {r['std_ms']:>8.2f} {r['speedup']:>9.3f}x")


if __name__ == "__main__":
    main()
