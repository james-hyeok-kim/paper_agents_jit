"""
DeepCache vs DSTP 비교 그래프.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/home/jovyan/workspace/paper_agents_jit/experiments/dstp/deepcache_compare.json") as f:
    data = json.load(f)

methods = []
speedups = []
medians = []
colors = []
markers = []
for k, v in data.items():
    if k == "meta": continue
    if k == "baseline":
        continue
    methods.append(k.replace("DeepCache_", "DC ").replace("DSTP_", "DSTP "))
    speedups.append(v["speedup"])
    medians.append(v["median_ms"])
    if "DSTP" in k:
        colors.append("#2E86AB")
        markers.append("o")
    else:
        colors.append("#E63946")
        markers.append("s")

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
xs = list(range(len(methods)))
bars = ax.bar(xs, speedups, color=colors, alpha=0.85, edgecolor='black')

for i, (sp, med) in enumerate(zip(speedups, medians)):
    ax.text(i, sp + 0.02, f"{sp:.2f}x\n({med:.0f}ms)",
            ha="center", va="bottom", fontsize=9)

ax.axhline(1.0, color="gray", ls="--", lw=1, label="baseline")
ax.set_xticks(xs)
ax.set_xticklabels(methods, rotation=30, ha="right")
ax.set_ylabel("Speedup vs baseline")
ax.set_title("DSTP vs DeepCache analog (PixelDiT-XL, 50 step, 128 imgs, B200)\n"
             "DSTP = full patch_blocks skip, DeepCache = middle blocks skip")

# 범례
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2E86AB", label="DSTP (ours)", alpha=0.85),
    Patch(facecolor="#E63946", label="DeepCache analog", alpha=0.85),
]
ax.legend(handles=legend_elements, loc="upper left")
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, max(speedups) * 1.25])

plt.tight_layout()
out = "/home/jovyan/workspace/paper_agents_jit/experiments/dstp/figures/deepcache_compare.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"저장: {out}")
