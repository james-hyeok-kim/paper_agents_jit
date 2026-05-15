"""
FID-10K 생성 (paper-grade): baseline OR K3_tbased 한 번에 하나씩.
두 인스턴스를 다른 GPU에서 병렬 실행하면 wall time 절반.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 generate_10k_split.py --config baseline
  CUDA_VISIBLE_DEVICES=2 python3 generate_10k_split.py --config K3_tbased
"""
import sys, os, time, json, argparse
import numpy as np
import torch

PIXELDIT_SRC = "/home/jovyan/workspace/Workspace_PixelDiT"
sys.path.insert(0, PIXELDIT_SRC)
sys.path.insert(0, "/home/jovyan/workspace/paper_agents_jit/experiments/dstp")

from dstp_sampler import (
    load_pixeldit_xl, make_sampler, StepSkipPixDiT,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, choices=["baseline", "K3_tbased"], required=True)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    import cv2
    device = torch.device("cuda")
    print(f"Config={args.config}, GPU={torch.cuda.get_device_name(0)}")

    CKPT = "/data/jameskimh/pixeldit_pretrained/imagenet256_pixeldit_xl_epoch320.ckpt"
    OUT_DIR = f"/data/jameskimh/dstp/fid_10k/{args.config}"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("모델 로딩...")
    base = load_pixeldit_xl(CKPT, device)
    if args.config == "K3_tbased":
        skip = StepSkipPixDiT(base, K=3, refresh_policy='t_based',
                              T_split=0.5, K_high=3)
        skip.eval()
        model_fn = skip
        reset_fn = skip.reset
    else:
        model_fn = base
        reset_fn = None
    print("로딩 완료\n")

    # 클래스 균등 (10/class)
    n_per_class = args.n // 1000
    labels_all = np.tile(np.arange(1000), n_per_class)
    rem = args.n - len(labels_all)
    if rem > 0:
        labels_all = np.concatenate([labels_all, np.arange(rem)])
    np.random.seed(0)
    np.random.shuffle(labels_all)

    sampler = make_sampler(num_steps=args.steps)

    # 워밍업
    print("워밍업...")
    _n = torch.randn(args.batch, 3, 256, 256, device=device)
    _l = torch.zeros(args.batch, dtype=torch.long, device=device)
    _u = torch.full((args.batch,), 1000, dtype=torch.long, device=device)
    for _ in range(2):
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = sampler(base, _n, _l, _u)
    torch.cuda.synchronize()
    del _n, _l, _u
    print("워밍업 완료\n")

    times_ms = []
    t_start = time.time()
    n_done = 0

    torch.manual_seed(42)
    with torch.no_grad():
        for i in range(0, args.n, args.batch):
            batch_labels = labels_all[i:i + args.batch]
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
                cv2.imwrite(os.path.join(OUT_DIR, f"{img_id:05d}.png"), img_bgr)
            n_done += B

            if (n_done % 200 == 0) or (n_done >= args.n):
                rate = n_done / (time.time() - t_start + 1e-6)
                eta = (args.n - n_done) / rate
                print(f"[{args.config}] {n_done}/{args.n}  {elapsed_ms:.1f}ms/img  rate={rate:.2f}/s  ETA={eta/60:.1f}min",
                      flush=True)

    stable = times_ms[len(times_ms)//3:]
    summary = {
        "config": args.config,
        "n_total": args.n,
        "num_steps": args.steps,
        "batch": args.batch,
        "median_ms": round(float(np.median(stable)), 2),
        "mean_ms": round(float(np.mean(stable)), 2),
        "std_ms": round(float(np.std(stable)), 2),
        "total_seconds": round(time.time() - t_start, 1),
    }
    out_json = f"/home/jovyan/workspace/paper_agents_jit/experiments/dstp/fid_10k_timing_{args.config}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n저장: {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
