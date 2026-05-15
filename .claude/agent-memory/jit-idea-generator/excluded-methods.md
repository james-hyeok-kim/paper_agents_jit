---
name: excluded-methods
description: Methods explicitly excluded from JiT/PixelDiT idea generation - already published, must not be re-proposed under different names
metadata:
  type: reference
---

# Excluded Methods (do not re-propose)

These methods are already published in the pixel-space generation literature and must NOT be proposed (even under different names) in future idea-generation sessions.

## Confirmed exclusions
1. **Clean prediction** — x0-prediction reformulations for pixel diffusion
2. **Frequency decoupling** — separating low/high frequency components for separate processing (e.g., EPG style)
3. **Global+local split** — dual-branch parallel architectures with global tokens + local patches
4. **Perceptual loss** — LPIPS or VGG-feature-based auxiliary losses
5. **Cascade flow** — multi-stage coarse-to-fine flow matching
6. **SSL pretraining** — DINO/MAE-style self-supervised pretraining for tokenizer or backbone

## How to apply
When generating new ideas, run each candidate against this list. Watch for renamings:
- "Multi-scale denoising" can be cascade flow in disguise — check for sequential stages
- "Spectral guidance" can be frequency decoupling
- "Dual-stream transformer" can be global+local split
- "Feature-matching loss" can be perceptual loss

If an idea looks similar, reframe with a clearly different mechanism or drop it.
