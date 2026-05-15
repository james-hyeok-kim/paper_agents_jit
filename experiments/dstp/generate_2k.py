"""
DSTP FID-2K: baseline과 K3_tbased 각 2000장 생성.
ImageNet 클래스 1000개를 균등하게 (class당 2장) 생성하여 ref set 분포와 정합.

출력:
  /data/jameskimh/dstp/fid_2k/baseline/   (2000 PNG)
  /data/jameskimh/dstp/fid_2k/K3_tbased/  (2000 PNG)
"""
import sys
import os
import time
import json
import argparse
import numpy as np
import torch

PIXELDIT_SRC = "/home/jovyan/workspace/Workspace_PixelDiT"
sys.path.insert(0, PIXELDIT_SRC)
sys.path.insert(0, "/home/jovyan/workspace/paper_agents_jit/experiments/dstp")

from dstp_sampler import (
    load_pixeldit_xl, make_sampler, StepSkipPixDiT,
)


def generate_balanced(model_fn, n_total, batch_size, device,
                      save_dir, num_steps, reset_fn=None, log_prefix=""):
    """클래스당 균등 (n_per_class = n_total // 1000) 분포로 생성"""
    import cv2
    os.makedirs(save_dir, exist_ok=True)

    n_classes = 1000
    n_per_class = n_total // n_classes
    labels_all = np.tile(np.arange(n_classes), n_per_class)
    # 부족분 채우기
    rem = n_total - len(labels_all)
    if rem > 0:
        labels_all = np.concatenate([labels_all, np.arange(rem)])
    np.random.seed(0)
    np.random.shuffle(labels_all)

    sampler = make_sampler(num_steps=num_steps)
    times_ms = []
    n_done = 0
    t_start = time.time()

    torch.manual_seed(42)
    with torch.no_grad():
        for i in range(0, n_total, batch_size):
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
            n_done += B

            if (n_done % 200 == 0) or (n_done >= n_total):
                rate = n_done / (time.time() - t_start + 1e-6)
                eta = (n_total - n_done) / rate
                print(f"  {log_prefix}[{n_done}/{n_total}] {elapsed_ms:.1f}ms/img | rate={rate:.1f}/s | ETA={eta:.0f}s",
                      flush=True)

    return times_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"GPU: {torch.cuda.get_device_name(args.gpu)} (cuda:{args.gpu})")

    CKPT = "/data/jameskimh/pixeldit_pretrained/imagenet256_pixeldit_xl_epoch320.ckpt"
    OUT_BASE = "/data/jameskimh/dstp/fid_2k"

    print("모델 로딩 중...")
    base_model = load_pixeldit_xl(CKPT, device)
    print("로딩 완료")

    # 워밍업
    print("워밍업...")
    _n = torch.randn(args.batch, 3, 256, 256, device=device)
    _l = torch.zeros(args.batch, dtype=torch.long, device=device)
    _u = torch.full((args.batch,), 1000, dtype=torch.long, device=device)
    _s = make_sampler(num_steps=args.steps)
    for _ in range(2):
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = _s(base_model, _n, _l, _u)
    torch.cuda.synchronize()
    del _n, _l, _u, _s
    print("워밍업 완료\n")

    timings = {}

    # ── BASELINE 2000장 ──
    if not args.skip_baseline:
        baseline_dir = os.path.join(OUT_BASE, "baseline")
        existing = len([f for f in os.listdir(baseline_dir)]) if os.path.exists(baseline_dir) else 0
        if existing >= args.n:
            print(f"baseline 이미 {existing}장 존재 — skip\n")
        else:
            print(f"=== BASELINE {args.n}장 ===")
            t0 = time.time()
            times = generate_balanced(
                base_model, args.n, args.batch, device,
                baseline_dir, args.steps, log_prefix="[BASE] "
            )
            dt = time.time() - t0
            stable = times[len(times)//3:]
            timings["baseline"] = {
                "ms_per_img_median": round(float(np.median(stable)), 2),
                "ms_per_img_std": round(float(np.std(stable)), 2),
                "total_seconds": round(dt, 1),
                "num_images": args.n,
            }
            print(f"BASELINE 완료: {dt:.1f}s, median={timings['baseline']['ms_per_img_median']}ms/img\n")

    # ── K3_tbased 2000장 ──
    k3_dir = os.path.join(OUT_BASE, "K3_tbased")
    existing = len([f for f in os.listdir(k3_dir)]) if os.path.exists(k3_dir) else 0
    if existing >= args.n:
        print(f"K3_tbased 이미 {existing}장 존재 — skip\n")
    else:
        print(f"=== K3_tbased {args.n}장 ===")
        skip_model = StepSkipPixDiT(
            base_model, K=3, refresh_policy='t_based',
            T_split=0.5, K_high=3,
        )
        skip_model.eval()
        t0 = time.time()
        times = generate_balanced(
            skip_model, args.n, args.batch, device,
            k3_dir, args.steps,
            reset_fn=skip_model.reset, log_prefix="[K3] "
        )
        dt = time.time() - t0
        stable = times[len(times)//3:]
        timings["K3_tbased"] = {
            "ms_per_img_median": round(float(np.median(stable)), 2),
            "ms_per_img_std": round(float(np.std(stable)), 2),
            "total_seconds": round(dt, 1),
            "num_images": args.n,
            "K": 3, "policy": "t_based", "T_split": 0.5,
        }
        print(f"K3_tbased 완료: {dt:.1f}s, median={timings['K3_tbased']['ms_per_img_median']}ms/img\n")

    if "baseline" in timings and "K3_tbased" in timings:
        speedup = timings["baseline"]["ms_per_img_median"] / timings["K3_tbased"]["ms_per_img_median"]
        timings["speedup_2k"] = round(speedup, 4)
        print(f"Speedup (2K): {speedup:.3f}x")

    out_json = "/home/jovyan/workspace/paper_agents_jit/experiments/dstp/timings_2k.json"
    with open(out_json, "w") as f:
        json.dump(timings, f, indent=2)
    print(f"timing 저장: {out_json}")


if __name__ == "__main__":
    main()
