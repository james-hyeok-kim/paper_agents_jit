---
name: plan-ratba-2026-05-15
description: Experiment plan for RATBA (Resolution-Adaptive Token Budget Allocation) drafted 2026-05-15 — reframed without codebook premise, key risk is patch-size mismatch at inference
metadata:
  type: project
---

# RATBA Experiment Plan Summary

**Status:** Plan drafted 2026-05-15. PoC not yet started.
**Owner:** james-hyeok-kim
**Idea source:** [[idea-batch-2026-05-15]] item #10, validated [[validation-2026-05-15-batch12]] as GO.

## Critical reframing applied
Original brief proposed "multi-scale tokenizer + codebook hierarchy" — both are invalid for JiT/PixelDiT ([[jit-no-codebook]]). Plan was rewritten around **patch-size scheduling** (varying patch size = varying token grid resolution) since both JiT and PixelDiT are standard DiTs with fixed `patch_size` (JiT: 16 or 32; PixelDiT-XL: 2 over a 256/8 latent — actually PixelDiT operates on pixel patches directly, patch_size=8 typical).

## Key structural decision
A single pretrained checkpoint has size-specific embed/unembed weights — patch_size cannot be varied at inference time on one checkpoint. PoC must use either:
- **(a) Two-checkpoint cascade** using `jit-h-32` (early, 64 tokens@256²) → `jit-h-16` (late, 256 tokens@256²). Cheap, runs today, but collides with PixelFlow framing → only OK as PoC + caveat. **This is the chosen PoC path.**
- **(b) Single-model input-resolution scheduling** (128²→256²). Requires re-noising / re-embedding handoff and is OOD for pretrained models. Risky.

Single-model differentiator must come back at the **trained-method stage** (a shared trunk with multiple patch embedders trained jointly).

## Core experiments
1. Sanity probe (1 day): does jit-h-16 give usable FID at 128² inference? Decides whether option (b) is viable as fallback.
2. PoC (Week 1, 1×A100): cascade jit-h-32 → jit-h-16 with x_0 handoff at T_switch. Sweep T_switch ∈ {0.2, 0.4, 0.6, 0.8}, sweep CFG. Target: avg token reduction ≥40%, FID gap vs. jit-h-16 < 1.0.
3. Main (Week 3–4, 4×A100): train a single dual-head JiT-H with patch_size {16, 32} adapters + learned T_switch + handoff (re-noise vs. denoise-then-renoise). ImageNet 256².
4. Ablation: schedule shape (step / linear / cosine / learned), handoff strategy, # resolution levels (2 vs. 3).

## Compute estimate
- PoC: ~20 A100-hours (FID-10K × 16 sweep points + FID-50K at best).
- Main paper: ~600 A100-hours (training dual-head model + ablations).

## Risks
- T_switch sweep result is the entire claim — if FID degrades monotonically with earlier switch, no token savings story.
- Two-checkpoint PoC is "PixelFlow-lite" — must frame as PoC only, not the method.

## Paths
- Code/logs/plots: `/home/jovyan/workspace/paper_agents_jit/experiments/ratba/`
- Checkpoints/data: `/data/jameskimh/ratba/`
- Pretrained available: `/data/jameskimh/james_jit_pretrained/jit-h-16/{jit-h-16,jit-h-32,jit-b-16,jit-b-32,jit-l-16,jit-l-32}/checkpoint-last.pth`, `/data/jameskimh/pixeldit_pretrained/imagenet256_pixeldit_xl_epoch320.ckpt`
