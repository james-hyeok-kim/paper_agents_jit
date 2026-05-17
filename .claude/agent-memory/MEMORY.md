# Agent Memory Index — JiT & PixelDiT Research

<!-- Add memory entries below as research progresses. Format: -->
<!-- - [Title](file.md) — one-line hook (under 150 chars) -->

- [Idea Batch 2026-05-15](jit-idea-generator/idea-batch-2026-05-15.md) — 12 inference-efficiency ideas across 6 categories, novelty/impact/feasibility scored
- [Idea Batch 2026-05-16](jit-idea-generator/idea-batch-2026-05-16.md) — 12 new ideas avoiding caching family + prior 12; spread across 10 families (quant/hw/theory/pixel-stats priority)
- [Excluded Methods](jit-idea-generator/excluded-methods.md) — published methods that must not be re-proposed (clean prediction, frequency decoupling, global+local split, perceptual loss, cascade flow, SSL pretraining)
- [JiT/PixelDiT have no codebook](jit-idea-validator/jit-no-codebook.md) — tokenizer-free; any "JiT codebook" idea is premise-invalid
- [2026-05-15 batch12 validation](jit-idea-validator/validation-2026-05-15-batch12.md) — 12 ideas scored; 4 NO-GO on codebook premise, Top 3 = #3, #10, #2
- [RATBA experiment plan](jit-experiment-planner/plan-ratba-2026-05-15.md) — Plan for idea #10 RATBA; reframed to patch-size scheduling, PoC = two-checkpoint cascade jit-h-32 → jit-h-16
- [Experiments Log](experiments-log.md) — RATBA FAILED (all checkpoints = 256 tokens), DSTP GO (추천: K3_tbased 1.53×; K3_periodic 2.12×이나 시각 품질 손상)
- [Caching family trap (DSTP 사고)](jit-idea-validator/caching-family-trap.md) — caching/skip 계열 자동 NO-GO 규칙
- [2026-05-16 batch12 validation](jit-idea-validator/validation-2026-05-16-batch12.md) — 8 NO-GO / 3 conditional / 1 GO (#3 SNR×Patch theory). HDiT-collision and codebook-premise errors recurred
- [Sacrificed ideas (web NO-GO)](jit-paper-feasibility-checker/sacrificed-ideas.md) — DSTP/BP-Hybrid/SNR-Scaling이 validator 통과했으나 web으로 prior art 발견
- [Video domain knowledge](shared/video-domain-knowledge.md) — pixel-space video DiT 도메인 지식, 모든 agent 공유
- [Domain pivot 2026-05-16](shared/domain-pivot-2026-05-16.md) — image → video pivot 결정
