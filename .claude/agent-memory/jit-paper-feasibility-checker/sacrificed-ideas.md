---
name: sacrificed-ideas
description: validator는 통과했으나 web search로 prior art 발견되어 NO-GO된 idea들 (paper-feasibility-checker 사고 회고)
metadata:
  type: feedback
---

# Validator 통과 → web search로 NO-GO된 idea들

## DSTP (2026-05-15)
- validator: 🟡 CONDITIONAL
- web search: TeaCache (2411.19108), SmoothCache (2411.10510), AdaCache (2411.02397),
  ProCache (2512.17298), DeepCache (2312.00858), Block Caching (2312.03209),
  Learning-to-Cache, FastCache, HiCache, First Block Cache, BWCache, SpectralCache,
  ERTACache, TaoCache, Foresight, T-GATE 등 16+
- **결정**: "DSTP는 step-skip caching family의 PixelDiT 이식". TeaCache(4.41×), AdaCache(4.49×) 대비 1.26× 열등
- 교훈: caching family 전체를 prior art로 인식 못 함

## BP-Hybrid (2026-05-16)
- validator: 🟡 CONDITIONAL
- web search:
  - **BDPM (2501.13915, Jan 2025)** — "MSB stronger pixel correlation, LSB more independent" 그대로
  - **HART (2410.10812, Oct 2024)** — "discrete for structure, continuous diffusion for residual" 그대로
- **결정**: BP-Hybrid = BDPM + HART 결합. 두 개 published idea 합성
- 교훈: "X + Y의 결합"인데 X와 Y가 별도 paper면 자동 NO-GO

## SNR-vs-Patch-Size Scaling (2026-05-16)
- validator: 🟢 GO (유일하게 GO 받음)
- web search:
  - **Chen 2023 (2301.10972)** — "optimal noise scheduling shifts towards noisier with image size due to pixel redundancy" — 핵심 결과 이미
  - **Patchification Scaling Laws (2502.03738, Feb 2026)** — patch size scaling 직접 다룸
  - **Hierarchical Patch Diffusion (2406.07792)** — multi-scale patches in diffusion
- **결정**: Empirical result는 published, closed-form만 추가하면 workshop 수준
- 교훈: theory paper에서도 핵심 insight가 published면 closed-form 추가는 incremental

## Pattern
모두 validator의 framework check는 통과. **공통점: 정적 knowledge가 6-12개월 published를 못 잡음.**

→ 모든 idea는 web-grounded paper-feasibility-checker 통과 필수.

[[caching-family-trap]] 과 함께 critical-finding 메모리.
