"""
DeepCache analog for PixelDiT-XL: skip middle blocks of patch_blocks every K steps.

DSTP는 patch_blocks 26개 전체를 K step마다 한번만 실행.
DeepCache 스타일은 patch_blocks의 중간 N개 블록만 skip하고,
앞쪽 + 뒤쪽 블록은 매 step 실행 → quality 보존하면서 부분 가속.

이 스크립트는:
  - DeepCache analog: skip middle blocks[skip_start:skip_end] every K steps
  - DSTP (full skip): 비교용
  - baseline
"""
import sys, os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn

PIXELDIT_SRC = "/home/jovyan/workspace/Workspace_PixelDiT"
sys.path.insert(0, PIXELDIT_SRC)
sys.path.insert(0, "/home/jovyan/workspace/paper_agents_jit/experiments/dstp")

from dstp_sampler import (
    load_pixeldit_xl, make_sampler, StepSkipPixDiT,
)


class DeepCachePixDiT(nn.Module):
    """
    PixelDiT의 patch_blocks 중 [skip_start:skip_end] 만 K step마다 실행.
    앞부분 patch_blocks[0:skip_start]와 뒷부분 patch_blocks[skip_end:]는 매 step 실행.
    """
    def __init__(self, base_model, K=3, skip_start=4, skip_end=22,
                 refresh_policy='periodic', T_split=0.5, K_high=3):
        super().__init__()
        self.net = base_model
        self.K = K
        self.skip_start = skip_start
        self.skip_end = skip_end
        self.refresh_policy = refresh_policy
        self.T_split = T_split
        self.K_high = K_high
        self._mid_cache = None
        self._step_count = 0

    def reset(self):
        self._mid_cache = None
        self._step_count = 0

    def _should_refresh_mid(self, t_val):
        if self._mid_cache is None:
            return True
        if self.refresh_policy == 'periodic':
            return (self._step_count % self.K == 0)
        elif self.refresh_policy == 't_based':
            if t_val > self.T_split:
                return (self._step_count % self.K_high == 0)
            return True
        return True

    def forward(self, x, t, y, s=None, mask=None):
        net = self.net
        B, _, H, W = x.shape
        pos = net.fetch_pos(H // net.patch_size, W // net.patch_size, x.device)
        x_patches = nn.functional.unfold(
            x, kernel_size=net.patch_size, stride=net.patch_size
        ).transpose(1, 2)
        t_emb = net.t_embedder(t.view(-1)).view(B, -1, net.hidden_size)
        y_emb = net.y_embedder(y).view(B, 1, net.hidden_size)
        c = nn.functional.silu(t_emb + y_emb)
        t_val = float(t[0].item())

        if s is None:
            # 항상 실행: s_embedder + 앞부분 blocks[0:skip_start]
            s = net.s_embedder(x_patches)
            for block in net.patch_blocks[:self.skip_start]:
                s = block(s, c, pos, mask)

            # 중간 부분: K step마다만 실행
            if self._should_refresh_mid(t_val):
                for block in net.patch_blocks[self.skip_start:self.skip_end]:
                    s = block(s, c, pos, mask)
                self._mid_cache = s.detach()
            else:
                s = self._mid_cache

            # 항상 실행: 뒷부분 blocks[skip_end:]
            for block in net.patch_blocks[self.skip_end:]:
                s = block(s, c, pos, mask)

            self._step_count += 1
            s = nn.functional.silu(t_emb + s)

        # pixel_blocks (항상 실행)
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
        x_img = nn.functional.fold(
            x_pixels, (H, W), kernel_size=net.patch_size, stride=net.patch_size
        )
        return x_img


def time_config(model_fn, n_total, batch_size, device, num_steps,
                reset_fn=None, save_dir=None, tag=""):
    sampler = make_sampler(num_steps=num_steps)
    times = []
    n_batches = n_total // batch_size
    torch.manual_seed(42)

    save_first_batch = (save_dir is not None)
    if save_first_batch:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(n_batches):
            labels = torch.randint(0, 1000, (batch_size,), device=device)
            null_l = torch.full((batch_size,), 1000, dtype=torch.long, device=device)
            noise = torch.randn(batch_size, 3, 256, 256, device=device)
            if reset_fn is not None:
                reset_fn()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                imgs = sampler(model_fn, noise, labels, null_l)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / batch_size * 1000
            times.append(elapsed)

            if i == 0 and save_first_batch:
                import cv2
                imgs = (imgs.float().cpu() + 1) / 2
                imgs = imgs.clamp(0, 1)
                for b_id in range(batch_size):
                    arr = np.round(imgs[b_id].numpy().transpose(1,2,0) * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir, f"{b_id:03d}_cls{int(labels[b_id]):04d}.png"),
                                arr[:, :, ::-1])
            print(f"  [{tag}] {(i+1)*batch_size}/{n_total} {elapsed:.1f}ms/img", flush=True)

    stable = times[len(times)//3:]
    return {
        "median_ms": float(np.median(stable)),
        "mean_ms":   float(np.mean(stable)),
        "std_ms":    float(np.std(stable)),
    }


def main():
    device = torch.device("cuda")
    CKPT = "/data/jameskimh/pixeldit_pretrained/imagenet256_pixeldit_xl_epoch320.ckpt"
    OUT = "/data/jameskimh/dstp/deepcache_compare"
    BATCH = 16
    NSTEPS = 50
    N = 128

    print(f"GPU={torch.cuda.get_device_name(0)}\n")
    print("모델 로딩...")
    base = load_pixeldit_xl(CKPT, device)
    print("로딩 완료\n")

    # 워밍업
    sampler = make_sampler(num_steps=NSTEPS)
    print("워밍업...")
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

    results = {"meta": {"steps": NSTEPS, "batch": BATCH, "n": N}}

    # 1) Baseline
    print("=== baseline ===")
    r = time_config(base, N, BATCH, device, NSTEPS, save_dir=os.path.join(OUT, "baseline"), tag="BASE")
    results["baseline"] = r
    base_med = r["median_ms"]
    print(f"  baseline median: {base_med:.2f}ms\n")

    # 2) DSTP (full skip)
    print("=== DSTP K=3 t_based (전체 patch_blocks skip) ===")
    dstp = StepSkipPixDiT(base, K=3, refresh_policy='t_based', T_split=0.5, K_high=3)
    dstp.eval()
    r = time_config(dstp, N, BATCH, device, NSTEPS, reset_fn=dstp.reset,
                    save_dir=os.path.join(OUT, "DSTP_K3_tbased"), tag="DSTP")
    r["speedup"] = base_med / r["median_ms"]
    results["DSTP_K3_tbased"] = r
    print(f"  DSTP median: {r['median_ms']:.2f}ms speedup={r['speedup']:.3f}x\n")
    del dstp; torch.cuda.empty_cache()

    # 3) DeepCache analog: 중간 블록만 skip [4:22] = 18 blocks
    for skip_range, k_val in [((4, 22), 3), ((6, 20), 3), ((4, 22), 5)]:
        s_, e_ = skip_range
        tag = f"DeepCache_b{s_}_{e_}_K{k_val}"
        print(f"=== {tag} ===")
        dc = DeepCachePixDiT(base, K=k_val, skip_start=s_, skip_end=e_,
                             refresh_policy='periodic')
        dc.eval()
        r = time_config(dc, N, BATCH, device, NSTEPS, reset_fn=dc.reset,
                        save_dir=os.path.join(OUT, tag), tag=tag)
        r["speedup"] = base_med / r["median_ms"]
        r["skip_range"] = list(skip_range)
        r["K"] = k_val
        results[tag] = r
        print(f"  {tag}: median={r['median_ms']:.2f}ms speedup={r['speedup']:.3f}x\n")
        del dc; torch.cuda.empty_cache()

    out_json = "/home/jovyan/workspace/paper_agents_jit/experiments/dstp/deepcache_compare.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n저장: {out_json}")

    print("\n" + "="*70)
    print(f"{'Config':<30} {'median (ms)':>12} {'speedup':>10}")
    print("-"*70)
    for k, v in results.items():
        if k == "meta": continue
        sp = v.get("speedup", 1.0)
        print(f"{k:<30} {v['median_ms']:>12.2f} {sp:>9.3f}x")


if __name__ == "__main__":
    main()
