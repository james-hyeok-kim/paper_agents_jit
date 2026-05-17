"""
JiT-H/16 (32 blocks) partial block caching.
- DeepCache style: blocks[start:end] 만 K step마다 실행
- 앞 N개 + 뒤 N개 blocks는 매 step 실행
- 전체 forward 캐싱(jit_stepskip.py)과 비교

목표: monolithic JiT에서도 architecture-aware caching이 작동하는지 검증.
"""
import sys, os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn

JIT_SRC = "/home/jovyan/workspace/Workspace_JiT"
sys.path.insert(0, JIT_SRC)

from model_jit import JiT_models


def load_jit(model_name, ckpt_path, device, img_size=256, num_classes=1000):
    print(f"loading {model_name}...")
    net = JiT_models[model_name](
        input_size=img_size, in_channels=3, num_classes=num_classes,
        attn_drop=0.0, proj_drop=0.0,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
    cleaned = {}
    for k, v in sd.items():
        if k.startswith('net.'): cleaned[k[4:]] = v
        elif k.startswith('module.net.'): cleaned[k[len('module.net.'):]] = v
        elif k.startswith('module.'): cleaned[k[len('module.'):]] = v
        else: cleaned[k] = v
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    if missing: print(f"[경고] missing ({len(missing)}): {missing[:5]}")
    if unexpected: print(f"[경고] unexpected ({len(unexpected)}): {unexpected[:5]}")
    net.eval().to(device)
    return net


class JiTPartialCache(nn.Module):
    """
    JiT의 blocks[skip_start:skip_end] 만 K step마다 실행.
    blocks[0:skip_start] 와 blocks[skip_end:depth] 는 매 step 실행.

    캐싱 위치: skip_start 직후 (즉, blocks[0:skip_start] 이후의 hidden state)
    """
    def __init__(self, net, K=3, skip_start=4, skip_end=28,
                 refresh_policy='periodic', T_split=0.5, K_high=3):
        super().__init__()
        self.net = net
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
        if self._mid_cache is None: return True
        if self.refresh_policy == 'periodic':
            return (self._step_count % self.K == 0)
        elif self.refresh_policy == 't_based':
            # JiT: t < T_split = 고노이즈 → skip, t >= T_split = 저노이즈 → refresh
            if t_val < self.T_split:
                return (self._step_count % self.K_high == 0)
            return True
        return True

    def forward(self, x, t, y):
        net = self.net
        t_val = float(t.flatten()[0].item())

        # Embed
        t_emb = net.t_embedder(t)
        y_emb = net.y_embedder(y)
        c = t_emb + y_emb
        h = net.x_embedder(x)
        h = h + net.pos_embed

        # 앞부분: 항상 실행
        for i in range(self.skip_start):
            block = net.blocks[i]
            if net.in_context_len > 0 and i == net.in_context_start:
                in_ctx = y_emb.unsqueeze(1).repeat(1, net.in_context_len, 1) + net.in_context_posemb
                h = torch.cat([in_ctx, h], dim=1)
            h = block(h, c, net.feat_rope if i < net.in_context_start else net.feat_rope_incontext)

        # 중간부분: K step마다만 실행
        if self._should_refresh_mid(t_val):
            h_mid = h
            for i in range(self.skip_start, self.skip_end):
                block = net.blocks[i]
                if net.in_context_len > 0 and i == net.in_context_start:
                    in_ctx = y_emb.unsqueeze(1).repeat(1, net.in_context_len, 1) + net.in_context_posemb
                    h_mid = torch.cat([in_ctx, h_mid], dim=1)
                h_mid = block(h_mid, c, net.feat_rope if i < net.in_context_start else net.feat_rope_incontext)
            self._mid_cache = h_mid.detach()
        else:
            h_mid = self._mid_cache

        # 뒷부분: 항상 실행
        h = h_mid
        for i in range(self.skip_end, len(net.blocks)):
            block = net.blocks[i]
            if net.in_context_len > 0 and i == net.in_context_start:
                in_ctx = y_emb.unsqueeze(1).repeat(1, net.in_context_len, 1) + net.in_context_posemb
                h = torch.cat([in_ctx, h], dim=1)
            h = block(h, c, net.feat_rope if i < net.in_context_start else net.feat_rope_incontext)

        h = h[:, net.in_context_len:]
        h = net.final_layer(h, c)
        out = net.unpatchify(h, net.patch_size)

        self._step_count += 1
        return out


@torch.no_grad()
def euler_sample(model_fn, noise, labels, num_classes, num_steps=50, cfg_scale=3.0,
                 cfg_interval=(0.1, 1.0), t_eps=1e-3, noise_scale=1.0, reset_fn=None):
    device = noise.device
    bsz = labels.size(0)
    z = noise * noise_scale
    if reset_fn is not None: reset_fn()
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    null_labels = torch.full_like(labels, num_classes)
    for i in range(num_steps):
        t = torch.full((bsz,), float(timesteps[i]), device=device)
        t_next = float(timesteps[i + 1])
        z_cat = torch.cat([z, z], dim=0)
        t_cat = torch.cat([t, t], dim=0)
        y_cat = torch.cat([labels, null_labels], dim=0)
        x_cat = model_fn(z_cat, t_cat, y_cat)
        x_cond, x_uncond = x_cat.chunk(2, dim=0)
        denom = (1.0 - t.view(-1, 1, 1, 1)).clamp_min(t_eps)
        v_cond = (x_cond - z) / denom
        v_uncond = (x_uncond - z) / denom
        low, high = cfg_interval
        t_val = float(t[0].item())
        if low <= t_val <= high:
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v_pred = v_cond
        z = z + (t_next - float(timesteps[i])) * v_pred
    return z


def time_config(model_fn, n, batch, device, num_steps, reset_fn=None,
                save_dir=None, tag=""):
    times = []
    n_batches = n // batch
    save_first = (save_dir is not None)
    if save_first: os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(42)
    with torch.no_grad():
        for i in range(n_batches):
            labels = torch.randint(0, 1000, (batch,), device=device)
            noise = torch.randn(batch, 3, 256, 256, device=device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                imgs = euler_sample(model_fn, noise, labels, 1000,
                                    num_steps=num_steps, cfg_scale=3.0, reset_fn=reset_fn)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / batch * 1000
            times.append(elapsed)
            print(f"  [{tag}] {(i+1)*batch}/{n} {elapsed:.1f}ms/img", flush=True)
            if i == 0 and save_first:
                import cv2
                imgs = (imgs.float().cpu() + 1) / 2
                imgs = imgs.clamp(0, 1)
                for b_id in range(batch):
                    arr = np.round(imgs[b_id].numpy().transpose(1,2,0) * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir, f"{b_id:03d}_cls{int(labels[b_id]):04d}.png"),
                                arr[:, :, ::-1])
    stable = times[len(times)//3:]
    return {
        "median_ms": float(np.median(stable)),
        "mean_ms": float(np.mean(stable)),
        "std_ms": float(np.std(stable)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="JiT-H/16")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)  # H/16은 큰 모델
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    size_tag = args.model.replace("JiT-", "jit-").replace("/", "-").lower()
    ckpt = f"/data/jameskimh/james_jit_pretrained/jit-h-16/{size_tag}/checkpoint-last.pth"
    out_base = f"/data/jameskimh/jit_dstp/{args.model.replace('/', '_')}_partial"
    os.makedirs(out_base, exist_ok=True)

    device = torch.device("cuda")
    print(f"Model={args.model} GPU={torch.cuda.get_device_name(0)}\n")

    net = load_jit(args.model, ckpt, device)
    depth = len(net.blocks)
    print(f"Depth={depth} blocks\n")

    # 워밍업
    print("워밍업...")
    _n = torch.randn(args.batch, 3, 256, 256, device=device)
    _l = torch.randint(0, 1000, (args.batch,), device=device)
    for _ in range(2):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = euler_sample(net, _n, _l, 1000, num_steps=args.steps, cfg_scale=3.0)
    torch.cuda.synchronize()
    del _n, _l
    print("워밍업 완료\n")

    # 설정: depth에 따라 partial range 조정
    if depth >= 32:  # H/16
        cache_configs = [
            ("baseline", None),
            (f"partial_b4_28_K3_periodic", lambda: JiTPartialCache(net, K=3, skip_start=4, skip_end=28, refresh_policy='periodic')),
            (f"partial_b4_28_K3_tbased",  lambda: JiTPartialCache(net, K=3, skip_start=4, skip_end=28, refresh_policy='t_based', T_split=0.5, K_high=3)),
            (f"partial_b6_26_K3_periodic", lambda: JiTPartialCache(net, K=3, skip_start=6, skip_end=26, refresh_policy='periodic')),
            (f"partial_b8_24_K3_tbased",  lambda: JiTPartialCache(net, K=3, skip_start=8, skip_end=24, refresh_policy='t_based', T_split=0.5, K_high=3)),
        ]
    else:  # B/16: 12 blocks
        cache_configs = [
            ("baseline", None),
            (f"partial_b2_10_K3_periodic", lambda: JiTPartialCache(net, K=3, skip_start=2, skip_end=10, refresh_policy='periodic')),
            (f"partial_b2_10_K3_tbased",  lambda: JiTPartialCache(net, K=3, skip_start=2, skip_end=10, refresh_policy='t_based', T_split=0.5, K_high=3)),
        ]

    results = {"meta": {"model": args.model, "depth": depth, "steps": args.steps,
                        "batch": args.batch, "n": args.n}}
    base_med = None
    for tag, ctor in cache_configs:
        print(f"=== {tag} ===")
        if ctor is None:
            model_fn = net
            reset_fn = None
        else:
            wrapper = ctor()
            wrapper.eval()
            model_fn = wrapper
            reset_fn = wrapper.reset
        save_dir = os.path.join(out_base, tag)
        r = time_config(model_fn, args.n, args.batch, device, args.steps,
                        reset_fn=reset_fn, save_dir=save_dir, tag=tag)
        if tag == "baseline":
            base_med = r["median_ms"]
            r["speedup"] = 1.0
        else:
            r["speedup"] = base_med / r["median_ms"]
        results[tag] = r
        print(f"  {tag}: median={r['median_ms']:.2f}ms speedup={r['speedup']:.3f}x\n")
        if ctor is not None: del wrapper
        torch.cuda.empty_cache()

    out = "/home/jovyan/workspace/paper_agents_jit/experiments/jit_dstp/partial_caching_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n저장: {out}")
    print("\n" + "="*70)
    print(f"{'Config':<30} {'median ms':>12} {'speedup':>10}")
    print("-"*70)
    for k, v in results.items():
        if k == "meta": continue
        sp = v.get("speedup", 1.0)
        print(f"{k:<30} {v['median_ms']:>12.2f} {sp:>9.3f}x")


if __name__ == "__main__":
    main()
