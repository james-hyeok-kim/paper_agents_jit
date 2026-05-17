"""
T_split ablation 결과를 그래프로 저장.
- speedup vs T_split (선 그래프)
- 노이즈 범위(min/max) 포함
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/home/jovyan/workspace/paper_agents_jit/experiments/dstp/tsplit_ablation.json") as f:
    data = json.load(f)

# T_split 곡선용 (K=3 t_based)
t_splits = [0.3, 0.4, 0.5, 0.6, 0.7]
medians = []
means = []
speedups = []
for t in t_splits:
    key = f"K3_t{t}"
    r = data[key]
    medians.append(r["median_ms"])
    means.append(r["mean_ms"])
    speedups.append(r["speedup"])

base_med = data["baseline"]["median_ms"]
periodic_speedup = data["K3_periodic"]["speedup"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# (1) speedup vs T_split
ax = axes[0]
ax.plot(t_splits, speedups, marker="o", lw=2, markersize=8, color="#2E86AB", label="K=3 t_based")
ax.axhline(periodic_speedup, color="#E63946", ls="--", lw=1.5, label=f"K=3 periodic ({periodic_speedup:.2f}x)")
ax.axhline(1.0, color="gray", ls=":", lw=1, label="baseline")
for t, s in zip(t_splits, speedups):
    ax.annotate(f"{s:.2f}x", (t, s), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax.set_xlabel("T_split (skip threshold)")
ax.set_ylabel("Speedup vs baseline")
ax.set_title("DSTP: T_split sweep (K=3, t_based)\nB200, 256 imgs, 20 steps, batch=16")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")

# (2) median latency
ax = axes[1]
ax.plot(t_splits, medians, marker="s", lw=2, markersize=8, color="#2E86AB", label="K=3 t_based")
ax.axhline(base_med, color="gray", ls=":", lw=1.5, label=f"baseline ({base_med:.1f}ms)")
ax.axhline(data["K3_periodic"]["median_ms"], color="#E63946", ls="--", lw=1.5,
           label=f"K=3 periodic ({data['K3_periodic']['median_ms']:.1f}ms)")
for t, m in zip(t_splits, medians):
    ax.annotate(f"{m:.0f}", (t, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax.set_xlabel("T_split (skip threshold)")
ax.set_ylabel("Median latency (ms/img)")
ax.set_title("Median latency by T_split\n(absolute timing noisy due to GPU contention)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")

plt.tight_layout()
out = "/home/jovyan/workspace/paper_agents_jit/experiments/dstp/figures/t_split_curve.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"저장: {out}")
