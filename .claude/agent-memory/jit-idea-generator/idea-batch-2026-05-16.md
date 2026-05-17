---
name: idea-batch-2026-05-16
description: Second batch of 12 JiT/PixelDiT ideas (2026-05-16) avoiding prior 12 + caching family + 16 paper saturation
metadata:
  type: project
---

# JiT/PixelDiT Idea Batch (2026-05-16)

12 new ideas generated under stricter constraints:
- Avoid prior 12 (CoGAS, Spatial-Causal Speculative, DSTP, TGJC, Rectified Pixel-Flow, PPM-PixelDiT, PiMaH, CCR, SVW Loss, RATBA, Cross-Batch KV, Hier Codebook ANN)
- Avoid caching family (16+ papers saturated; keywords: timestep-aware cache, block/layer/feature caching, step-skip, U-shape sensitivity, etc.)
- Avoid published pixel-DiT methods (JiT clean pred, PixelDiT dual-level, DiP global+local, DeCo/FREPix freq split, EPG SSL, PixelGen perceptual, PixelFlow cascade, HDiT hourglass)

## Mechanism distinctness principle (advisor guidance)

Each idea named a signal X that none of prior 12 used. Adopted spread strategy: ≤2 ideas per family, all 10 listed families covered, 4 most under-explored (quantization, hw-codesign, theory, pixel-statistics) prioritized.

## Idea index

| # | Family | Title | Signal X (unique) | Score (N/I/F) |
|---|--------|-------|-------------------|---------------|
| 1 | quantization | QNDS: Quantization-Noise-as-Denoising-Step | weight bit-width × noise schedule | 5/4/5 |
| 2 | hw-codesign | LoCS-FP8: Block-FP8 Locality Kernel | FP8 format + Hilbert 2D coords | 5/5/2 |
| 3 | theory | SNR-vs-Patch-Size Scaling Law | analytical p*(SNR) closed form | 5/3/4 |
| 4 | theory | Attention Rank Collapse Analysis | attention singular value spectrum | 4/3/4 |
| 5 | pixel-statistics | SLCE: Spatial Lipschitz Head Pruning | local Lipschitz constant per token | 4/4/4 |
| 6 | noise-schedule | PNAC: Frequency Anisotropy Calibration | direction-dependent freq power spectrum | 5/4/3 |
| 7 | training-eff | PLDI: Latent-DiT Weight Transfer | cross-domain pretrained weight init | 4/5/5 |
| 8 | video | TC-PVDiT: Frame-Difference Tokens | temporal frame-difference sparsity | 5/5/2 |
| 9 | multimodal | CC-KV: Prompt-Cluster KV Reuse | CLIP prompt semantic clustering | 4/4/5 |
| 10 | high-res | ARF-512: Layer-Schedule Receptive Field | depth-indexed receptive field schedule | 4/5/2 |
| 11 | hybrid | BP-Hybrid: Bit-Plane AR+Diffusion | RGB bit-plane decomposition | 5/4/2 |
| 12 | quantization | CE-QAT: Codebook-Entropy-Weighted QAT | VQ codebook usage entropy | 4/4/4 |

## Highest-priority for literature check
- High novelty + impact: 1 (QNDS), 2 (LoCS-FP8), 6 (PNAC), 11 (BP-Hybrid)
- Quick wins (low effort): 3 (Scaling Law), 4 (Rank Collapse), 7 (PLDI), 9 (CC-KV)
- Higher risk, larger payoff: 8 (Video), 10 (ARF-512)

## Gaps identified during generation
- Quantization completely unexplored in pixel DiT (justifies ideas 1, 12)
- No theoretical/analytical work on pixel DiT (justifies 3, 4)
- HW-level kernel co-design absent (justifies 2)
- Bit-plane structure is pixel-unique signal not used anywhere (justifies 11)
- Cross-domain weight transfer from FLUX/SD3 to pixel DiT not tried (justifies 7)
- Frame-difference token sparsity for video DiT (justifies 8)
