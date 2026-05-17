---
name: validation-2026-05-16-batch12
description: Batch validation of 12 JiT/PixelDiT inference efficiency ideas on 2026-05-16; strict protocol with Family→SOTA→Auto-NoGo→ModelSwap→Self-check
metadata:
  type: project
---

# 2026-05-16 Batch12 Validation Result

## Top-level verdicts

| # | Idea | Family | Verdict | Headline reason |
|---|------|--------|---------|------------------|
| 1 | QNDS | quantization | 🔴 NO-GO | Q-Diffusion (2023) already analyzes quantization-as-noise-on-noise-schedule; bit×timestep schedule = EfficientDM/BitsFusion |
| 2 | LoCS-FP8 | hw-codesign | 🟡 CONDITIONAL | Hilbert+FP8 combo plausibly new but no FID/quality gain; hw-codesign + perfect-numerics + 2× only = MLSys not generation venue |
| 3 | SNR×Patch scaling law | theory | 🟢 GO | Theory paper, RATBA data already available to test prediction; minimum-bar is verifiable closed-form not SOTA FID |
| 4 | Attention Rank Collapse | theory | 🟡 CONDITIONAL | "Attention is not all you need" + AT-EDM cover the mechanism; pixel-specific singular-spectrum trajectory might still be novel as theory |
| 5 | SLCE Lipschitz Head Prune | pixel-stats | 🔴 NO-GO | ToMeSD + AT-EDM already prune in smooth regions; Lipschitz framing = relabeling, mechanism identical |
| 6 | PNAC anisotropic noise | noise-schedule | 🔴 NO-GO | Blurring/Cold/Soft-truncation diffusion + DeCo/FREPix in own prior-art table; anisotropic spectrum schedule conflicts with FREPix |
| 7 | PLDI weight transfer | training-eff | 🔴 NO-GO | OUT OF SCOPE — validator charter is inference efficiency, not training cost. Even in-scope, FLUX→pixel surgery is single trick, weak contribution |
| 8 | TC-PVDiT video diff | video | 🔴 NO-GO | No established pixel video DiT baseline → "two papers in a trenchcoat"; also residual-diff is in MAGVIT/Video Diffusion residual literature |
| 9 | CC-KV CLIP cluster KV | multimodal | 🔴 NO-GO | Only works for batched diverse prompts (narrow use case); LLM-side prompt caching (vLLM/SGLang) trivially adapts; T2I version published as PromptCache 2024 |
| 10 | ARF-512 depth-RF schedule | high-res | 🔴 NO-GO | Direct collision with HDiT (Hourglass DiT) — hourglass IS layer-varying receptive field, single-forward, listed in own prior-art table |
| 11 | BP-Hybrid bit-plane AR+diff | hybrid | 🟡 CONDITIONAL | Bit-Diffusion (Chen 2022) + MAR/VAR cover edges; bit-plane partition specifically across AR/diff might be new but high risk |
| 12 | CE-QAT codebook entropy | quantization | 🔴 NO-GO | PREMISE INVALID — VQ codebook does not exist in JiT/PixelDiT/DiP per [[jit-no-codebook]]. Author concedes "only valid when codebook exists" |

## Verdict count
- 🟢 GO: 1 (#3 SNR×Patch theory)
- 🟡 CONDITIONAL: 3 (#2 LoCS-FP8, #4 Rank Collapse, #11 BP-Hybrid)
- 🔴 NO-GO: 8 (#1, #5, #6, #7, #8, #9, #10, #12)

## Top 3 (risk-adjusted yield)
1. **#3 SNR×Patch scaling law** — only true GO; theory paper bar is verifiable prediction not SOTA-FID; RATBA empirical data exists to test fit
2. **#4 Attention Rank Collapse** (conditional) — theory venue, can stand on plotting alone if anisotropic spectrum hasn't been published; verify against AT-EDM
3. **#11 BP-Hybrid** (conditional, high risk) — high novelty if Bit-Diffusion hasn't been pushed this way; difficulty 5 = expensive PoC

## Recurring failure patterns this batch
1. **Repeat of codebook-premise error** (#12) — author claimed pixel-specific even while requiring VQ
2. **Hourglass collision** (#10 vs HDiT in same prior-art table) — same error as batch1 #6 (PPM-PixelDiT vs HDiT)
3. **Scope drift** (#7 = training not inference) — validator must enforce charter
4. **Two-papers-in-a-trenchcoat** (#8 needs pixel video DiT baseline first)
5. **Relabeling existing mechanism** (#5 Lipschitz = ToMe; #1 quant-schedule = Q-Diffusion)
6. **Latent-space analogue exists** (#9 PromptCache; #2 FP8 = FlashAttention-3)

## Family-level observations
- **Quantization family**: Q-Diffusion / PTQ4DM / EfficientDM / BitsFusion / BitsFusion-Plus saturate the timestep×bit schedule space; new entries need either FP4 hardware or codebook-aligned (none in JiT)
- **Theory family**: Lowest novelty bar — closed-form prediction or trajectory plot can be sufficient; both #3 and #4 land here
- **Pruning/sparse-attn**: ToMe variants saturate; "Lipschitz/locality/sliding-window" rebrandings all collide
- **HW-codesign**: MLSys-style 1.4×–2.6× wallclock without quality gain = wrong venue, but legitimate

## Honest take
1 of 12 GO is consistent with stricter protocol and harder family mix this batch. Pre-committed expectation in advisor call was "most NO-GO" and that held. **The 1 GO (#3) is the most defensible because theory papers don't have to beat EPG FID 1.58 — they have to make a verifiable prediction.**

Related: [[caching-family-trap]] [[jit-no-codebook]] [[validation-2026-05-15-batch12]]
