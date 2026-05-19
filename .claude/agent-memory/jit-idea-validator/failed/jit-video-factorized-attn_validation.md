---
name: feasibility-jit-video-factorized-attn-2026-05-19
description: JiT-Video Factorized Spatial-Temporal Attention validation 2026-05-19 — 🔴 ABANDON. Latte(2401.03048) covers mechanism exactly + FrameDiT(2603.09721) already treats it as baseline-to-beat. BP-Hybrid lesson fires.
metadata:
  type: project
---

# Verdict: 🔴 ABANDON

**Idea**: JiT-Video Factorized Spatial-Temporal Attention
**Date**: 2026-05-19 KST
**Validator**: jit-idea-validator

---

## Web-Grounded Prior Art (확인된 prior art)

### 1. Latte (arXiv:2401.03048, TMLR 2025) — DECISIVE 🔴

Abstract에서 exact quote:
> "We propose Latte, a novel Latent Diffusion Transformer for video generation. Latte first extracts spatio-temporal tokens from input videos and then adopts a series of Transformer blocks to model video distribution in the latent space. In order to model a substantial number of tokens extracted from videos, four efficient variants are introduced from the perspective of decomposing the spatial and temporal dimensions of input videos."

Interleaved Fusion (Variant 1) 설명:
> "the Transformer backbone of this variant comprises two distinct types of Transformer blocks: spatial Transformer blocks and temporal Transformer blocks operating in an 'interleaved fusion' pattern where spatial blocks process tokens at matching temporal indices before temporal blocks capture cross-frame relationships."

- 평가 데이터셋: UCF-101 (우리 idea와 동일)
- 메커니즘: Spatial DiT blocks + interleaved temporal blocks = 제안 아이디어와 **완전 동일**
- 유일한 차이: latent space (Latte) vs pixel space (우리) — mechanism level이 아니라 application level

**자동 NO-GO 트리거**: "Pixel-only claim broken by latent-space same-mechanism paper → ABANDON"

---

### 2. HPDM (arXiv:2406.07792, CVPR 2024) — PARTIAL 🟡 (domain cover)

Abstract exact quote:
> "We improve PDMs in two principled ways. First, to enforce consistency between patches, we develop deep context fusion — an architectural technique that propagates the context information from low-scale to high-scale patches in a hierarchical manner. Second, to accelerate training and inference, we propose adaptive computation, which allocates more network capacity and computation towards coarse image details. The resulting model sets a new state-of-the-art FVD score of 66.32 and Inception Score of 87.68 in class-conditional video generation on UCF-101 256²..."
> "our model is the first diffusion-based architecture which is trained on such high resolutions entirely end-to-end."

- **Pixel-space video generation** 확인 (no VAE, end-to-end)
- 아키텍처: RIN (Recurrent Interface Networks), NOT DiT
- Cascade 구조 (hierarchical patch scales)
- UCF-101 FVD 66.32 SOTA — "first pixel-space video DiT" 클레임의 "first" 부분 약화

---

### 3. FrameDiT (arXiv:2603.09721, March 2026) — DECISIVE 🔴 (baseline-을-beat 위치)

Abstract exact quote:
> "High-fidelity video generation remains challenging for diffusion models due to the difficulty of modeling complex spatio-temporal dynamics efficiently. Recent video diffusion methods typically represent a video as a sequence of spatio-temporal tokens which can be modeled using Diffusion Transformers (DiTs). However, this approach faces a trade-off between the strong but expensive Full 3D Attention and the efficient but temporally limited Local Factorized Attention."

실험 결과:
> "spatial attention is first applied within each frame, followed by temporal attention across frames for each spatial location"

- **Local Factorized Attention** = 우리 제안 메커니즘을 **이미 baseline(개선 대상)**으로 취급
- UCF-101 FVD: FrameDiT-H 170.1 (latent space — 2026년 3월 현재)
- 2026년 3월 기준으로 이미 이 메커니즘은 "극복해야 할 것"으로 분류됨

---

### 4. EPG (arXiv:2510.12586) — Image-only (video extension 없음)

Abstract exact quote:
> "Pixel-space generative models are often more difficult to train and generally underperform compared to their latent-space counterparts, leaving a persistent performance and efficiency gap. In this paper, we introduce a novel two-stage training framework that closes this gap for pixel-space diffusion and consistency models..."

- Image-only, 비디오 없음
- "First pixel-space video DiT" framing에 무관

---

## 자동 NO-GO 트리거 발동 체크

| 트리거 | 발동 여부 | 근거 |
|--------|-----------|------|
| "X+Y combination where X,Y separately published" | ✅ FIRED | Latte(mechanism) + HPDM(pixel domain) + JiT(image init) = 3개 published ingredients |
| "Pixel-only claim broken by latent same-mechanism paper" | ✅ FIRED | Latte = 동일 메커니즘, latent만 다름 |
| "2+ papers cover core contribution" | ✅ FIRED | Latte(mechanism) + FrameDiT(already-baseline) |

---

## Mechanism Comparison

| Aspect | Our Idea | Latte (Variant 1) | Delta |
|--------|----------|-------------------|-------|
| Core mechanism | Spatial DiT blocks + interleaved temporal attention layers | "spatial Transformer blocks and temporal Transformer blocks operating in an interleaved fusion pattern" | **identical** |
| Space | pixel (no VAE) | latent space | framing only |
| Backbone | JiT (DiT variant) | DiT (standard) | trivial |
| Dataset | UCF-101 | UCF-101 | identical |
| Init | JiT image weights | random | training detail |

결론: mechanism level 차이 없음. latent→pixel은 application-level 변환이며 reviewer 즉시 reject.

---

## 왜 "Systems Contribution Paper" framing도 성립 안 되는가

1. **HPDM이 이미 pixel-space video transformer**: RIN 기반이지만 "first end-to-end pixel-space video diffusion" 클레임 선점
2. **FrameDiT가 factorized attention을 "개선 대상 baseline"으로 취급**: 2026년 3월 기준, 이 메커니즘을 proposal로 제출하면 reviewer가 "this is just Local Factorized Attention from FrameDiT, not novel"
3. **Architecture novelty = zero**: DiT in pixel space for video = Latte without VAE. No mechanism introduced.

---

## Part A — Rigor Scores

| Dimension | Score | Key concern |
|-----------|-------|-------------|
| Novelty residual | 1/5 | Latte Variant 1 = 완전 동일 메커니즘 |
| Technical feasibility | 3/5 | 64×64×16f @patch=16 token수 합리적이지만 무관 |
| Sham control design | 2/5 | Frame-stack baseline 제안 있으나 실험 전에 NO-GO |
| Scope | 2/5 | Single dataset (UCF-101), already-beaten mechanism |

## Part B — Publication Scores

| Dimension | Score | Key concern |
|-----------|-------|-------------|
| Venue fit | 1/5 | No venue accepts Latte variant without mechanism novelty |
| Competition timing | 1/5 | FrameDiT 2026-03에 이미 baseline으로 분류 |
| Reviewer-objection survival | 1/5 | "This is Latte in pixel space" — instant reject |
| Minimum bar reachability | 1/5 | UCF-101 FVD < 66.32(HPDM) 달성 불가능 (mechanism 동일) |

**Overall: 1.4/5**

---

## Web Queries 실행 이력

1. `pixel space video diffusion transformer DiT 2024 2025 2026 arxiv` — image-only results dominant
2. `HPDM "hierarchical patch diffusion" video pixel 2024 2025` — HPDM architecture confirmed
3. `pixel video DiT factorized temporal attention 2025 2026 arxiv video generation` — FrameDiT 발견
4. `Latte video diffusion transformer pixel space extension follow-up 2025 2026` — no pixel extension found
5. `"JiT" OR "PixelDiT" video temporal extension 2025 2026 arxiv generation` — no prior video extension
6. `FrameDiT factorized spatial temporal attention video generation pixel 2026` — FrameDiT mechanism confirmed
7. `pixel space video generation DiT "no VAE" "no autoencoder" temporal attention 2025 2026` — EPG image-only
8. `"pixel space" video diffusion transformer UCF-101 FVD 2024 2025 2026 arxiv` — comprehensive landscape

Total: 8 queries (≥5 mandatory 충족)
WebFetch: Latte abstract, HPDM abstract, FrameDiT full analysis, EPG scope check (4개)

---

## 피벗 옵션 (video-domain-knowledge Tier 1에서)

### Pivot A: Pixel-Space Video Distillation from Latent Teacher
- **Mechanism**: Latent video model (CogVideoX/Wan)에서 pixel-space video model로 cross-space knowledge distillation
- **Novel because**: Distillation mechanism itself (cross-space, not cross-model), prior art 없음
- **Check needed**: HART (2410.10812) — latent→pixel distillation 여부 확인

### Pivot B: Motion-Aware Temporal Token Compression (Pixel Space)
- **Mechanism**: Optical flow 기반 motion magnitude로 token sparsity 결정 — motion 없는 region은 temporal attention skip
- **Novel because**: Pixel-space specific (VAE에서 motion signal 손실), factorized attention과 orthogonal mechanism
- **Check needed**: VideoPoet motion masking, MAGVIT-v2 tokenizer

### Pivot C: Frame-Adaptive Spatial Resolution in Pixel Space
- **Mechanism**: High-motion frames → high spatial resolution, low-motion frames → low resolution; dynamic patch size
- **Novel because**: Adapts resolution at frame-level within single forward pass, pixel-space only (latent compression makes this impossible)
- **Check needed**: PyramidFlow, hierarchical video methods

---

## Pattern 기록 (향후 blacklist용)

- **Factorized spatial-temporal attention for video DiT**: Latte (2401.03048)가 완전 커버. latent vs pixel 차이로는 부족.
- **Interleaved spatial/temporal blocks**: FrameDiT (2603.09721)가 이를 "극복할 baseline"으로 분류 — 2026년 제출 시 구식 취급.
- [[caching-family-trap]] 패턴과 유사: well-known mechanism + domain shift = framing only

Related: [[caching-family-trap]] [[bp-hybrid-lesson]] [[snr-scaling-lesson]] [[video-domain-knowledge]]
