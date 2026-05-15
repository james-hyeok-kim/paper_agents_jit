# Agent Memory Index — JiT & PixelDiT Research

<!-- Add memory entries below as research progresses. Format: -->
<!-- - [Title](file.md) — one-line hook (under 150 chars) -->

- [Idea Batch 2026-05-15](jit-idea-generator/idea-batch-2026-05-15.md) — 12 inference-efficiency ideas across 6 categories, novelty/impact/feasibility scored
- [Excluded Methods](jit-idea-generator/excluded-methods.md) — published methods that must not be re-proposed (clean prediction, frequency decoupling, global+local split, perceptual loss, cascade flow, SSL pretraining)
- [JiT/PixelDiT have no codebook](jit-idea-validator/jit-no-codebook.md) — tokenizer-free; any "JiT codebook" idea is premise-invalid
- [2026-05-15 batch12 validation](jit-idea-validator/validation-2026-05-15-batch12.md) — 12 ideas scored; 4 NO-GO on codebook premise, Top 3 = #3, #10, #2
- [RATBA experiment plan](jit-experiment-planner/plan-ratba-2026-05-15.md) — Plan for idea #10 RATBA; reframed to patch-size scheduling, PoC = two-checkpoint cascade jit-h-32 → jit-h-16
- [Experiments Log](experiments-log.md) — RATBA FAILED (all checkpoints = 256 tokens), DSTP GO (추천: K3_tbased 1.53×; K3_periodic 2.12×이나 시각 품질 손상)
