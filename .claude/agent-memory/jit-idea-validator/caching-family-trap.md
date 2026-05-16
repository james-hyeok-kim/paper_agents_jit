---
name: caching-family-trap
description: DSTP validation 사고 회고 — caching/skip 계열은 이미 포화. 새 idea가 t-aware/block-level/threshold-based caching이면 자동 NO-GO
metadata:
  type: feedback
---

# Caching Family Trap (DSTP 사고 회고, 2026-05-15)

## 사고
"DSTP: Step-Skip Caching for Pixel-Space DiT" 아이디어가 🟡 CONDITIONAL GO 판정 받음.
실제로는 TeaCache/SmoothCache/AdaCache/ProCache/Block Caching/DeepCache 등 caching family와
**core mechanism 동일** — paper로 publish 불가능한 incremental work.

## 누락된 prior art (16+ papers)
- DeepCache (NeurIPS 2024), Block Caching (CVPR 2024)
- TeaCache (CVPR 2025) — timestep embedding aware
- SmoothCache (CVPRW 2025) — layer-wise error
- AdaCache (ICCV 2025) — adaptive cache for video
- HiCache, ProCache (AAAI 2026), FastCache, FirstBlockCache
- Learning-to-Cache (NeurIPS 2024), BWCache, SpectralCache, ERTACache, TaoCache, Foresight

## 일반화된 인사이트 (이미 published, 우리만 몰랐던 것)
> "Across diffusion timesteps, feature variations of DiT blocks exhibit a U-shaped pattern:
> early/late steps are sensitive (composition/detail), middle steps are tolerant (cacheable).
> Aggressive caching in middle + protect endpoints — TADS, TeaCache, SmoothCache가 모두 발표."

DSTP의 "t-aware refresh policy"가 본인 contribution이라 주장했으나 위와 동일.

## SOTA 비교 (DSTP가 진 부분)
| Method | Speedup |
|--------|---------|
| TeaCache (FLUX) | 2× |
| TeaCache (Open-Sora) | 4.41× |
| AdaCache | 4.49× |
| ProCache | 2.9× |
| SmoothCache | 1.7× |
| **DSTP (ours)** | **1.26×** ← 최하위 |

## 강제 적용 규칙 (future validator)

다음 키워드 조합이 idea에 있으면 **자동 🔴 NO-GO**:
1. "timestep aware" + "caching"
2. "block caching" + "skip"  
3. "threshold" + "refresh"
4. "U-shaped" + "middle tolerant"
5. "step-skip" + "diffusion"
6. "feature reuse across timesteps"

새 idea가 caching family와 mechanism 동일하고 "X model에 적용" only면 → 자동 NO-GO.

## 살아남으려면
1. 위 16개 paper 어디에도 없는 mechanism이어야 함
2. SOTA speedup 2× 이상 (training-free 기준)
3. Mechanism level 분석으로 latent와 pixel의 본질적 차이를 보여줘야 함 (단순 적용 X)

[[jit-no-codebook]] 와 함께 critical-finding 메모리.
