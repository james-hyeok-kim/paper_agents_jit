"""
모든 실험 결과를 paper-quality 테이블로 통합.
"""
import json
import os
from pathlib import Path

BASE = Path("/home/jovyan/workspace/paper_agents_jit/experiments")
DSTP = BASE / "dstp"

def load(p):
    if not p.exists(): return None
    with open(p) as f:
        return json.load(f)

results = {
    "PoC_20step": load(DSTP / "results.json"),
    "FID_2K": load(DSTP / "fid_2k_results.json"),
    "Tsplit": load(DSTP / "tsplit_ablation.json"),
    "Canonical_100step": load(DSTP / "canonical_100steps.json"),
    "DeepCache_compare": load(DSTP / "deepcache_compare.json"),
    "JiT_full": load(BASE / "jit_dstp/results.json"),
    "JiT_partial": load(BASE / "jit_dstp/partial_caching_results.json"),
}

print("=" * 100)
print(" " * 30 + "DSTP PAPER RESULTS — UNIFIED TABLE")
print("=" * 100)

# Table 1: Main result on PixelDiT
print("\n## Table 1: PixelDiT-XL Step-Skip Configurations (PoC, 20 step, 128 imgs)")
poc = results["PoC_20step"]
if poc:
    print(f"{'Config':<20} {'ms/img':>10} {'speedup':>10} {'IS':>8} {'IS drop':>10} {'Visual':>15}")
    print("-" * 80)
    visual_map = {
        "baseline": "기준",
        "K2_periodic": "weak(배경손실)",
        "K3_periodic": "FAIL(구성손실)",
        "K2_tbased": "PASS",
        "K3_tbased": "PASS(추천)",
    }
    for k in ["baseline", "K2_periodic", "K3_periodic", "K2_tbased", "K3_tbased"]:
        if k not in poc: continue
        r = poc[k]
        ms = r.get("ms_per_img_median", 0)
        sp = r.get("speedup", 1.0)
        is_v = r.get("IS", 0)
        is_d = r.get("IS_drop", 0.0)
        print(f"{k:<20} {ms:>10.1f} {sp:>9.3f}x {is_v:>8.3f} {is_d:>10.3f} {visual_map.get(k, ''):>15}")

# Table 2: T_split ablation
print("\n## Table 2: T_split Ablation (K=3 t_based, 256 imgs, 20 step)")
ts = results["Tsplit"]
if ts:
    print(f"{'T_split':<10} {'median ms':>12} {'speedup':>10}")
    print("-" * 40)
    print(f"{'baseline':<10} {ts['baseline']['median_ms']:>12.2f} {'1.000x':>10}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        key = f"K3_t{t}"
        if key in ts:
            r = ts[key]
            print(f"{t:<10} {r['median_ms']:>12.2f} {r['speedup']:>9.3f}x")
    if "K3_periodic" in ts:
        r = ts["K3_periodic"]
        print(f"{'periodic':<10} {r['median_ms']:>12.2f} {r['speedup']:>9.3f}x")

# Table 3: Canonical 100-step
print("\n## Table 3: 100-step Canonical (Paired Interleaved, 256 imgs)")
c = results["Canonical_100step"]
if c:
    print(f"{'Config':<15} {'median ms':>12} {'mean ms':>12} {'std':>8}")
    print("-" * 50)
    print(f"{'baseline':<15} {c['baseline']['median_ms']:>12.2f} {c['baseline']['mean_ms']:>12.2f} {c['baseline']['std_ms']:>8.2f}")
    print(f"{'K3_tbased':<15} {c['K3_tbased']['median_ms']:>12.2f} {c['K3_tbased']['mean_ms']:>12.2f} {c['K3_tbased']['std_ms']:>8.2f}")
    print(f"\nPaired speedup (median of per-batch ratios): {c['speedup_median_of_pairs']:.3f}x")

# Table 4: DeepCache comparison
print("\n## Table 4: DSTP vs DeepCache analog (PixelDiT-XL, 50 step, 128 imgs)")
dc = results["DeepCache_compare"]
if dc:
    print(f"{'Method':<35} {'median ms':>12} {'speedup':>10}")
    print("-" * 60)
    base_med = dc['baseline']['median_ms']
    print(f"{'baseline':<35} {base_med:>12.2f} {'1.000x':>10}")
    for k, v in dc.items():
        if k in ("meta", "baseline"): continue
        print(f"{k:<35} {v['median_ms']:>12.2f} {v['speedup']:>9.3f}x")

# Table 5: JiT generalization
print("\n## Table 5: JiT Generalization")
jf = results["JiT_full"]
jp = results["JiT_partial"]
if jf:
    print(f"\n### JiT-B/16 Full Output Caching (50 step, 64 imgs)")
    for k in ["baseline", "K3_tbased", "K3_periodic"]:
        if k in jf:
            r = jf[k]
            print(f"  {k:<20} median={r['median_ms']:>8.2f}ms  speedup={r['speedup']:>6.3f}x  std={r['std_ms']:>6.2f}")

if jp:
    print(f"\n### {jp['meta']['model']} Partial Block Caching ({jp['meta']['steps']} step)")
    base_med = jp['baseline']['median_ms']
    for k, v in jp.items():
        if k in ("meta", "baseline"): continue
        print(f"  {k:<35} median={v['median_ms']:>8.2f}ms  speedup={v['speedup']:>6.3f}x")
    print(f"  baseline                              median={base_med:>8.2f}ms")

# FID summary
print("\n## Table 6: FID Measurements")
fid2k = results["FID_2K"]
if fid2k:
    print("FID-2K (20 step, 2K samples vs 2K ImageNet val ref):")
    print(f"  baseline FID:  {fid2k['results']['baseline_FID']:.3f}")
    print(f"  K3_tbased FID: {fid2k['results']['K3_tbased_FID']:.3f}")
    print(f"  gap:           +{fid2k['results']['FID_gap']:.3f}")

# Try to load FID-10K if available
fid10k_p = DSTP / "fid_10k_results.json"
if fid10k_p.exists():
    fid10k = load(fid10k_p)
    print("\nFID-10K (100 step, 10K samples - paper grade):")
    if fid10k:
        for k, v in fid10k.items():
            print(f"  {k}: {v}")
else:
    print("\nFID-10K: 진행 중 (overnight)")

print("\n" + "=" * 100)
