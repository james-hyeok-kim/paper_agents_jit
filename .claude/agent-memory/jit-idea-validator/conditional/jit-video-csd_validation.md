---
name: feasibility-jit-video-csd-2026-05-19
description: JiT-Video Cross-Space Distillation (CSD) validation. 🟡 PIVOT REQUIRED. L2P (2605.12013) preempts the framing; Jacobian direction math unresolved; score-vs-architecture mechanism diff is real but narrow.
metadata:
  type: project
---

## Verdict: 🟡 PIVOT REQUIRED

### Validation date: 2026-05-19 KST
### Web queries executed: 13
### Abstracts fetched: 7

---

## Found Prior Art (Web-Confirmed)

### DECISIVE (mechanism-adjacent, framing kill)
- arXiv:2605.12013 (L2P, May 2026) — "we propose the Latent-to-Pixel (L2P) transfer paradigm, an efficient framework that directly harnesses the rich knowledge of pre-trained LDMs to build powerful pixel-space models. Specifically, L2P discards the VAE in favor of large-patch tokenization and freezes the source LDM's intermediate layers, exclusively training shallow layers to learn the latent-to-pixel transformation." — **FRAMING KILL**: CSD can no longer claim "first latent-to-pixel transfer paradigm." Same month (May 2026). VAE-free inference advantage also preempted.

### PARTIAL (mechanism-partial, does not kill)
- arXiv:2305.18455 (Diff-Instruct) — "we propose a general framework called Diff-Instruct to instruct the training of arbitrary generative models as long as the generated samples are differentiable with respect to the model parameters" — Same-space distillation. Does not directly cover latent-teacher → pixel-student cross-space setup. But if CSD's math reduces to "encoder in the forward pass," this becomes Pattern 2.
- arXiv:2510.12586 ("There is No VAE") — "closes this gap for pixel-space diffusion and consistency models… achieves FID of 1.58 on ImageNet-256 without relying on pre-trained VAEs" — No teacher at all. Weakens CSD's VAE-free advantage claim (already achievable without latent teacher).
- arXiv:2309.15818 (Show-1) — Hybrid pixel + latent pipeline at inference, NOT distillation. Different paradigm.
- arXiv:2409.17565 (Pixel-Space Post-Training) — Adds pixel loss to latent LDM training, same-space, not cross-space distillation.

### COMPLEMENTARY
- arXiv:2602.11401 (Latent Forcing) — Latent + pixel joint processing via trajectory reordering. Different mechanism (joint inference, not distillation).
- arXiv:2602.02493 (PixelGen) — Perceptual loss for pixel diffusion, no teacher. Provides pixel-space baseline comparison.

---

## Mechanism Comparison

| Aspect | CSD (Proposed) | L2P (2605.12013) | Diff-Instruct (2305.18455) |
|---|---|---|---|
| Transfer direction | latent teacher → pixel student score | latent LDM → pixel student feature | same-space teacher → same-space student |
| Training signal | latent score projected via Jacobian | frozen intermediate layers + flow matching | Integral KL divergence on scores |
| Architecture coupling | none (teacher not retained at inference) | strong (frozen LDM layers in student) | same-architecture variants typical |
| VAE at inference | none (CSD claimed advantage) | none (L2P also VAE-free) | N/A |
| Video scope | target | images only | images only |
| Novelty residual | score-level cross-space for VIDEO | — | — |

---

## Critical Math Problem (Unresolved — HARD GATE)

The proposed loss `MSE(ε_φ(x_t,t), J·ε_θ(E(x_t),t))` where `J = ∂D/∂z` has two unresolved issues:

### Issue 1: Jacobian Direction
Chain rule for score mapping from latent to pixel space requires **encoder Jacobian** `∂E/∂x` applied as VJP, NOT decoder Jacobian `∂D/∂z`. The decoder Jacobian maps in the wrong direction (latent → pixel, when we need pixel → latent adjoint for gradient flow). If the actual implementation uses "encoder in the forward pass + gradient backprop," this reduces to Diff-Instruct with VAE encoder in the loop = **Pattern 2 (same mechanism, domain shift only) → AUTO NO-GO**.

### Issue 2: Out-of-Distribution Teacher Query
`E(x_t)` where `x_t = α_t x_0 + σ_t ε_pixel` (pixel noise schedule) is NOT the distribution the latent teacher was trained to denoise. The teacher was trained on `α_t E(x_0) + σ_t ε_latent` (latent noise schedule). Because E is nonlinear, `E(x_t) ≠ α_t E(x_0) + σ_t ε'`. Teacher's score at `E(x_t)` may produce meaningless supervision.

**Resolution required BEFORE any GPU spend.**

---

## Pivot Options

### Pivot A (Recommended): Score-Level Cross-Space Transfer for Video (CSD)
- **What changes**: Explicitly differentiate CSD from L2P at mechanism level: "CSD learns via score distillation, L2P learns via feature alignment. CSD allows arbitrary student architecture (no LDM backbone coupling). CSD is architecture-agnostic for the student."
- **Video specificity**: L2P is images-only. CSD-Video is the video extension — this IS mechanism-level difference if temporal consistency behavior differs.
- **Hard requirement**: Math must be correct (encoder Jacobian, or justified alternative). OOD probe must show teacher score is useful.

### Pivot B (Safe): Abandon Score Projection, Keep VAE-Free + Latent Teacher
- Use L2P-style (freeze intermediate LDM layers) but extend to VIDEO — would be "L2P for Video"
- Easier, lower novelty risk
- Risk: L2P authors likely working on video extension right now

### Pivot C: Score Distillation ONLY for Temporal Consistency Signal
- Use latent teacher's temporal score as consistency regularizer, not primary supervisor
- Orthogonal to L2P's feature-freezing approach
- Narrower but cleaner contribution

---

## Hard Pre-Experiment Gates (MUST PASS before GPU spend)

### Gate 1: Math Derivation (BLOCKER)
Write the exact loss derivation in LaTeX. Identify which of:
(a) Decoder Jacobian `∂D/∂z` multiplication — justify why this direction is correct
(b) Encoder Jacobian VJP `(∂E/∂x)^T` — if so, is this equivalent to Diff-Instruct with encoder in loop? If yes → Pattern 2 → ABANDON
(c) Something else (e.g., approximate inversion, E-D cycle, Tweedie formula)
**If (b) and equivalent to Diff-Instruct → immediately re-route to jit-ideation**

### Gate 2: OOD Probe (BLOCKER)
Run 5-minute image toy: does `ε_θ(E(x_t), t)` produce meaningful pseudo-ground-truth for pixel denoising? Compare:
- Teacher score at `E(x_t)` [OOD query]
- Teacher score at `α_t E(x_0) + σ_t ε` [in-distribution query]
If OOD score ≈ in-distribution score → proceed. If not → must use in-distribution query (changes training setup significantly).

### Gate 3: Sham Control Design
Sham = random-init "teacher" with same architecture, frozen, replaces CogVideoX teacher. If sham student trains to similar FVD → mechanism is NOT the latent score, it's the architecture prior. Must falsify this.

### Gate 4: L2P Positioning
Explicitly compute: what does CSD-Video achieve that L2P-Video (hypothetical extension) cannot?
- Architecture-agnostic student (no LDM backbone coupling) — testable
- Score-level temporal consistency (vs feature-level) — must measure
If no clear advantage → merge into L2P paradigm as video extension (credit L2P)

### HARD ABORT Conditions
- Gate 1 resolves to case (b) equivalent to Diff-Instruct → ABANDON
- Gate 2: OOD score is uninformative → training signal invalid → ABANDON
- Gate 3: Sham matches CSD within FVD ±50 → mechanism confounded → ABANDON
- Gate 4: No architecture-agnostic or temporal advantage vs L2P-Video → merge or ABANDON

---

## Scoring Matrix

### Rigor (Part A)
| Dimension | Score (1-5) | Key concern |
|---|---|---|
| Novelty residual | 3/5 | L2P (2605.12013) preempts framing; score-vs-architecture diff is mechanism-level but narrow |
| Technical feasibility | 2/5 | Jacobian direction wrong as written; OOD teacher query unvalidated; full Jacobian ~O(C_latent × H × W) infeasible at 1080p |
| Sham control design | 3/5 | Sham design clear; must falsify mechanism vs architecture prior |
| Scope | 3/5 | Video = right scope; image component covered by L2P + "There is No VAE" |

### Publication (Part B — Web-Grounded)
| Dimension | Score (1-5) | Key concern |
|---|---|---|
| Venue fit | 3/5 | NeurIPS 2026 (submit ~May 2026) tight; ICLR 2027 (~Sep 2026) more realistic |
| Competition timing (13 web queries) | 2/5 | L2P (May 2026 same month) = strong contemporaneous threat; must cite and differentiate |
| Reviewer-objection survival | 2/5 | "Isn't this L2P for video?" + "Jacobian direction is wrong" = two immediate kills if unaddressed |
| Minimum bar reachability | 3/5 | FVD parity with or below Imagen Video (~2880) on UCF-101 + demonstrate architecture-agnostic advantage vs L2P |

**Overall: 2.6/5**

---

## Top 3 Risks

1. **Math collapses to Pattern 2**: If Jacobian formulation resolves to "encoder in forward pass" = standard Diff-Instruct with cross-space wrapping → AUTO NO-GO per BLACKLIST Pattern 2.
2. **L2P video extension preempts**: L2P (May 2026) is images-only now, but the extension to video is straightforward. If L2P authors or another group publishes before submission, CSD loses the only remaining domain advantage.
3. **OOD teacher signal**: The latent teacher's score at `E(x_t)` (pixel-noised input encoded) may be garbage because noise schedules differ. If so, the entire supervision signal is invalid.

---

## Venue Recommendation
- **Primary**: NeurIPS 2026 (deadline ~May 29 2026 — TIGHT, likely missed already). Route to ICLR 2027 (deadline ~Sep 2026).
- **Backup**: ECCV 2026 (check deadline) or CVPR 2027.

## Minimum Bar for Publication
1. FVD ≤ 2000 on UCF-101 256×256 (beats Imagen Video ~2880)
2. Architecture-agnostic student: must demonstrate CSD training works with a student that does NOT share backbone with the LDM teacher (if it requires backbone sharing, it's L2P variant)
3. Sham control: random-init teacher student FVD must be ≥ 2× worse than CSD student
4. Math derivation published as standalone appendix, with Jacobian approximation justified

## Strongest Version of This Idea
CSD-Video with analytically correct encoder-Jacobian score projection, applied to video where L2P's feature-freezing approach cannot trivially be extended (because temporal consistency requires more than feature re-use). The core paper becomes: "Score-level latent-to-pixel transfer enables architecture-agnostic pixel video generation with no LDM backbone coupling at inference."

---

## Web Queries Executed
1. "cross space distillation video diffusion latent pixel 2024 2025 2026"
2. "score distillation video latent pixel space 2024 2025"
3. "VAE decoder Jacobian distillation diffusion 2024 2025 2026"
4. "knowledge distillation latent diffusion pixel video 2025 2026"
5. "latent to pixel video generation distillation 2025 2026"
6. "SDS VSD score distillation video generation pixel space 2024 2025"
7. "pixel space video diffusion model latent teacher distillation VAE-free training 2025 2026 arxiv"
8. "cross-space OR cross space score distillation latent pixel diffusion model 2025 arxiv"
9. "diffusion distillation latent teacher pixel student video generation arxiv 2025 2026"
10. "CogVideoX pixel space student distillation Wan video generation score matching 2025 2026 arxiv"
11. "Imagen Video pixel diffusion model training VAE-free end-to-end 2024 2025 arxiv improvement"
12. "Diff-Instruct pixel student latent teacher diffusion generator training arxiv 2024 2025"
13. "Distribution Matching Distillation generator pixel space latent teacher video 2024 2025"

## Abstracts Fetched
1. arXiv:2309.15818 (Show-1) — NOT cross-space distillation, hybrid inference pipeline
2. arXiv:2409.17565 (Pixel-Space Post-Training) — same-space pixel loss addition
3. arXiv:2602.11401 (Latent Forcing) — trajectory reordering, not distillation
4. arXiv:2510.15301 (SVG / LDM without VAE) — DINO feature space, not cross-space distillation
5. arXiv:2510.12586 ("There is No VAE") — SSL pre-training, no latent teacher
6. arXiv:2503.16397 (Scale-wise Distillation) — patch MMD, no cross-space
7. arXiv:2305.18455 (Diff-Instruct) — same-space IKL distillation
8. arXiv:2405.14867 (DMD2) — same-space distribution matching
9. arXiv:2605.12013 (L2P) — **DECISIVE framing kill**: latent-to-pixel transfer paradigm, images only, layer-freezing mechanism
10. arXiv:2602.02493 (PixelGen) — pixel diffusion + perceptual loss, no teacher

## Why We Trust This Verdict
- 13 web queries executed (all 6 mandatory + 7 additional)
- 10 abstracts directly fetched
- L2P found via independent search, not seeded by user
- Pattern 2 risk explicitly tested (Diff-Instruct searches)
- Scarcity bias acknowledged (only surviving idea) — explicitly resisted per BLACKLIST lesson
- Math direction issue independently flagged by advisor and retained in verdict

*Last updated: 2026-05-19 KST*
