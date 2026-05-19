---
name: feasibility-jit-video-fap-2026-05-19
description: JiT-Video Frame-Adaptive Patch (FAP) validation — 🔴 ABANDON. VGDFR (2504.12259) covers per-frame motion-driven token reduction; APT+DDiT cover patch-size mechanism; pixel-only claim is framing not mechanism.
metadata:
  type: project
---

## Verdict: 🔴 ABANDON

## Summary

FAP = per-frame motion-driven patch-size variation in pixel-space video DiT.
Core signal: motion magnitude → token budget per frame.
This signal-to-token-budget mapping is already published (VGDFR, latent-space).
Patch-size as the reduction operator is a implementation-level choice, not mechanism-level.
Pixel-only claim is falsifiable (latent-FAP implementable in ~50 lines on CogVideoX).

---

## Web Queries Executed (8 total)

1. `"frame adaptive patch" video diffusion 2025 2026`
2. `"variable patch size" video generation per-frame motion 2025`
3. `"adaptive tokenization" video DiT diffusion 2024 2025 2026`
4. `"motion aware patch" video transformer diffusion 2025 2026`
5. `pixel space video "patch size" adaptive diffusion 2026`
6. `"dynamic patch" video generation temporal diffusion 2025 2026`
7. `VGDFR "latent token" video "motion" "dynamic" diffusion arxiv 2025`
8. `"per-frame" "patch" "motion" video diffusion transformer adaptive token 2025 arxiv`

---

## Found Prior Art (Web-Confirmed)

### arXiv:2504.12259 — VGDFR (DECISIVE)
> "VGDFR adaptively adjusts the number of elements in latent space based on the motion frequency of the latent space content, using fewer tokens for low-frequency segments while preserving detail in high-frequency segments."

- Signal: motion frequency per frame/segment → token count. Identical to FAP's signal.
- Operator: frame merging (temporal axis). FAP uses patch resize (spatial axis).
- Domain: latent space. FAP targets pixel space.
- Verdict: Same signal, different operator, different space. Operator difference is implementation-level, not mechanism-level. Pixel-only claim not defended.

### arXiv:2510.18091 — APT (PARTIAL)
> "Vision Transformers (ViTs) partition input images into uniformly sized patches regardless of their content, resulting in long input sequence lengths for high-resolution images."

- Image-only, homogeneity-driven (not motion-driven), not video.
- Covers the adaptive patch SIZE mechanism for content-driven allocation.
- Combined with VGDFR's motion signal = FAP mechanism fully decomposed into published components.

### arXiv:2602.16968 — DDiT (PARTIAL)
> "Our key insight is that early timesteps only require coarser patches to model global structure, while later iterations demand finer (smaller-sized) patches to refine local details."

- Timestep-axis only (confirmed by full paper: "for a given timestep, we use a fixed patch-size").
- DDiT authors explicitly list per-frame/region adaptive patch as future work: "A natural future research would involve investigating varied patch sizes within a given timestep, for further efficiency."
- This means: FAP is the "obvious next step" explicitly flagged by DDiT authors → reviewer will see this immediately.

### arXiv:2406.07792 — HPDM (COMPLEMENTARY)
> "we develop deep context fusion -- an architectural technique that propagates the context information from low-scale to high-scale patches in a hierarchical manner."

- No per-frame adaptive patch. FAP would be additive. But HPDM as base does not protect FAP novelty claim.

---

## Three Discriminators (Advisor-Required)

### Discriminator 1: VGDFR mechanism vs FAP mechanism
VGDFR signal: motion frequency → per-frame token count.
FAP signal: motion magnitude → per-frame token count.
Both use the same causal chain: detect inter-frame motion → allocate more tokens to high-motion frames.
The only difference: VGDFR merges frames (temporal), FAP resizes patches (spatial).
Judgment: **framing-level, not mechanism-level**. A reviewer will say "FAP is VGDFR with a spatial compression operator instead of temporal merging, ported to pixel space."

### Discriminator 2: Pixel-only claim
Claim: "VAE forces 8× spatial grid, so latent-FAP conflicts with compression."
Rebuttal: Patch sizes 2/4 in latent correspond to effective 16/32 in pixel. Latent patchification and VAE compression are orthogonal dimensions. CogVideoX uses (2,4,4) patchification by default — varying that to (1,2,2) per high-motion frame is a 50-line change with no VAE conflict.
Judgment: **Pixel-only claim is framing, not a technical barrier**. Reviewer can demonstrate this in 5 minutes.

### Discriminator 3: DDiT future work overlap
DDiT explicitly lists "varied patch sizes within a given timestep" as future work.
FAP = "varied patch sizes per frame" = a specific instance of this future work.
Judgment: **Obvious next step from published paper**. Reviewer will cite DDiT and reject as "incremental extension."

---

## Mechanism Comparison

| Aspect | FAP (Our Idea) | VGDFR (2504.12259) | APT (2510.18091) | DDiT (2602.16968) |
|--------|---------------|-------------------|------------------|-------------------|
| Core signal | motion magnitude | motion frequency | content homogeneity | denoising timestep |
| Per-frame adaptation | YES | YES | N/A (image) | NO (global) |
| Reduction operator | patch resize | frame merge | patch resize | patch resize |
| Domain | pixel (claimed) | latent | pixel image | latent |
| Video | YES | YES | NO | YES |

Conclusion: FAP = (VGDFR's signal) + (APT/DDiT's operator) + (pixel domain). Three-way combination where all components are published.

---

## Auto NO-GO Triggers Hit

- [x] "X + Y combination" where X (VGDFR's motion signal), Y (APT/DDiT's patch resize) separately published → ABANDON (BP-Hybrid lesson)
- [x] DDiT authors explicitly list FAP's mechanism as "natural future research" → "obvious next step" flag
- [x] Pixel-only claim broken by latent-FAP being technically feasible (Discriminator 2)

---

## Why We Trust This Verdict

- 8 web queries executed
- 6 papers fetched (VGDFR, APT, DDiT full paper, HPDM, DiP, VFRTok)
- Exact quotes, not paraphrases
- Three discriminator tests all point to ABANDON
- Advisor review confirms BP-Hybrid pattern recognition

---

## Pivot Options (If Salvageable)

### Pivot A: Motion-Driven Patch Size + Frequency Domain (not motion)
Use frequency-domain content complexity (DCT magnitude) instead of motion magnitude.
Risk: APT uses "homogeneity" which is similar. Not a clean differentiation.

### Pivot B: Asymmetric Pixel Patch (keyframe large, interpolated small)
Fixed ratio schedule based on position, not motion signal. Removes VGDFR overlap.
Risk: too simple, trivially implemented by prior work.

### Pivot C: Focus entirely on Pixel-Space Distillation (CSD)
The only video-pivot idea that survives all 3 discriminators as of 2026-05-19.
Per BLACKLIST Safe Zone item 1: "cross-space score-projection distillation, no direct prior art confirmed across 7+ queries."

Recommendation: Pivot to CSD (Cross-Space Distillation). FAP is abandoned.

---

*Validation date: 2026-05-19 KST*
*Agent: jit-idea-validator (Video Pivot branch)*
