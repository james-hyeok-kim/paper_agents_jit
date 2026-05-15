---
name: validation-2026-05-15-batch12
description: Batch validation of 12 JiT/PixelDiT inference efficiency ideas on 2026-05-15
metadata:
  type: project
---

12 ideas validated. Result summary:

- 🔴 NO-GO: #1 (CoGAS), #8 (CCR), #9 (variance-weighted), #12 (ANN codebook) — all premised on JiT having a discrete codebook, which is false (JiT/PixelDiT/DiP are all tokenizer-free)
- 🔴 NO-GO: #6 (PPM-PixelDiT) — direct collision with HDiT (Hourglass DiT), which is in the user's own prior-art list
- 🔴 NO-GO: #7 (PiMaH) — Mamba+diffusion space is saturated (DiM, ZigMa, DiffuSSM, Diffusion-RWKV); "pixel space" framing is not a novelty hook
- 🟡 CONDITIONAL: #2 (spatial speculative), #4 (TGJC — but generator is tokenizer-free, so "joint" loses meaning; needs reframe), #5 (rectified pixel-flow — distillation-free claim is suspect), #11 (cross-batch KV reuse)
- 🟢 GO: #3 (DSTP step-aware pruning), #10 (RATBA resolution-adaptive budget)

**Top 3:** #3 DSTP, #10 RATBA, #2 Spatial Speculative (in that order of risk-adjusted yield)

**Common failure patterns observed:**
1. Assuming JiT has a codebook (4 ideas)
2. "Why pixel space" being only "more tokens → more savings" rather than a mechanistic reason (most ideas)
3. Direct collision with prior art listed in the same prompt (HDiT vs #6)
4. Optimistic latency claims without distillation (#5 4-step claim)

**Why:** Recorded for future batches so the same premise errors don't recur.

**How to apply:** When validating future JiT efficiency ideas, run the JiT-codebook premise check first ([[jit-no-codebook]]). Then check against user's prior-art list literally — HDiT-style collisions are common.
