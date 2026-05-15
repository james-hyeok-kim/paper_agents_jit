---
name: idea-batch-2026-05-15
description: First batch of 12 JiT/PixelDiT inference efficiency ideas across 6 categories generated on 2026-05-15
metadata:
  type: project
---

# JiT/PixelDiT Idea Batch (2026-05-15)

12 ideas generated avoiding excluded methods (clean prediction, frequency decoupling, global+local split, perceptual loss, cascade flow, SSL pretraining).

## SOTA bars at time of generation
- EPG FID 1.58 @256
- PixelDiT FID 1.61 @256
- DiP 10x speedup
- PixelGen GenEval 0.79

## Idea index

| # | Category | Title | Novelty | Impact | Feasibility | Status |
|---|----------|-------|---------|--------|-------------|--------|
| 1 | Inference accel | Codebook-Guided Attention Sparsification (CoGAS) | 4 | 4 | 4 | proposed |
| 2 | Inference accel | Spatial-Causal Speculative Parallel Decoding for JiT-AR | 4 | 5 | 3 | proposed |
| 3 | Inference accel | Denoising-Step-Aware Token Pruning (DSTP) | 3 | 4 | 5 | proposed |
| 4 | Training eff | Tokenizer-Generator Joint Curriculum (TGJC) | 4 | 3 | 4 | proposed |
| 5 | Training eff | Rectified Pixel-Flow with Distillation-Free Step Reduction | 4 | 5 | 3 | proposed |
| 6 | Architecture | Progressive Patch Merging Transformer (PPM-PixelDiT) | 3 | 4 | 4 | proposed |
| 7 | Architecture | Pixel-Mamba Hybrid (PiMaH) | 4 | 4 | 3 | proposed |
| 8 | Loss/training | Codebook-Consistency Regularized Denoising Loss (CCR) | 4 | 3 | 5 | proposed |
| 9 | Loss/training | Spatial-Variance-Weighted Reconstruction Loss | 3 | 4 | 5 | proposed |
| 10 | High-res | Resolution-Adaptive Token Budget Allocation (RATBA) | 4 | 5 | 3 | proposed |
| 11 | Other | Cross-Batch Visual Redundancy KV Reuse | 5 | 4 | 4 | proposed |
| 12 | Other | Hierarchical Codebook Lookup with ANN | 3 | 2 | 5 | proposed |

## Key pixel-space gaps identified
- Tokenizer codebook structure is unique inductive bias absent in latent-space (drives ideas 1, 4, 8, 12)
- O(H*W) token explosion creates qualitatively different bottlenecks vs. latent (drives ideas 6, 7, 10)
- Pixel-space spatial statistics (variance, locality) are signals that are smoothed away in latent (drives ideas 2, 3, 5, 9)
- Cross-sample visual redundancy is preserved in pixels, removed by VAE compression (drives idea 11)

## Recommended priority for literature check
High priority (most novel + impactful): 2, 5, 10, 11
Medium: 1, 4, 7
Quick wins (low effort, easy paper): 3, 8, 9, 12
