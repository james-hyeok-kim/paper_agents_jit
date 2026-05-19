# JiT Idea Validator — Memory Index

## Validation Results
- [Validation 2026-05-15 Batch12](validation-2026-05-15-batch12.md) — First batch validation (12 ideas)
- [Validation 2026-05-16 Batch12](validation-2026-05-16-batch12.md) — Second batch (1 GO, 3 CONDITIONAL, 8 NO-GO)
- [FAILED: JiT-Video Factorized Attn (2026-05-19)](failed/jit-video-factorized-attn_validation.md) — 🔴 ABANDON. Latte(2401.03048) = identical mechanism. FrameDiT(2603.09721) = already treating it as baseline-to-beat.
- [CONDITIONAL: JiT-Video CSD (2026-05-19)](conditional/jit-video-csd_validation.md) — 🟡 PIVOT REQUIRED. L2P (2605.12013, May 2026) preempts "latent-to-pixel" framing. Jacobian direction math wrong as written. 4 hard gates before GPU spend.

## Reusable Heuristics (Patterns)
- [Caching Family Trap](caching-family-trap.md) — Video caching poised to fire as well
- [JiT No Codebook](jit-no-codebook.md) — Premise check for VQ ideas

## Key Lessons Applied
- **BP-Hybrid lesson**: X+Y combination (Latte mechanism + pixel domain) → ABANDON
- **DSTP lesson**: Web search caught what static knowledge missed (FrameDiT 2026-03)
- **Pixel-only trap**: Latent-space same-mechanism paper breaks pixel-only claim (Latte → fires auto-ABANDON)
