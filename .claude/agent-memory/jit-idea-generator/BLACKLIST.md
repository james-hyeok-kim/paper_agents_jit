# JiT Idea Generator — BLACKLIST

이 파일은 validation에서 폐기된 아이디어의 핵심 fail mechanism을 기록한다.
새 아이디어 생성 시 이 패턴들은 **즉시 제외**한다.

---

## Blacklist Table

| Round | 날짜 | 폐기 아이디어 | 핵심 fail mechanism | Preempting arXiv ID |
|-------|------|---------------|---------------------|---------------------|
| Video-Pivot-R1 | 2026-05-19 | JiT-Video Factorized Spatial-Temporal Attention | Latte (2401.03048)가 동일 메커니즘(interleaved spatial-temporal DiT blocks)을 latent space에서 완전 구현. FrameDiT (2603.09721, 2026-03)가 이를 "개선할 baseline"으로 취급. Pixel vs latent 차이는 mechanism level이 아니라 framing. | arXiv:2401.03048 (Latte), arXiv:2603.09721 (FrameDiT) |
| Video-Pivot-R2 | 2026-05-19 | JiT-Video Motion-Aware Token Sparsity (MATS) | Motion-Adaptive Temporal Attention (2603.17398, Mar 2026)가 latent SD 위에서 motion-content 기반 temporal RF 조정 메커니즘 완전 구현. ADAPTOR (CVPRW 2025)가 motion-based token reduction을 latent video DiT에 적용. "Pixel-only" 차별점은 categorical하지 않고 empirical에 불과 — 2603.17398이 latent에서 작동함이 증명됨. Pattern 2 (Mechanism+Domain shift) trap. | arXiv:2603.17398 (Motion-Adaptive TA), CVPRW 2025 (ADAPTOR), arXiv:2502.01776 (Sparse VideoGen) |
| Video-Pivot-R2 | 2026-05-19 | JiT-Video Frame-Adaptive Patch (FAP) | VGDFR (2504.12259)가 per-frame motion-driven token reduction을 latent space에서 완전 구현 (동일 인과 체인: motion → per-frame token budget). APT (2510.18091)가 patch-size 변경 메커니즘 cover. DDiT (2602.16968) 저자가 per-frame/region patch variation을 "natural future research"로 명시 → obvious next step. Pixel-only claim은 framing (latent-FAP는 CogVideoX 위에서 50줄 구현 가능). BP-Hybrid 패턴: X(VGDFR signal) + Y(APT/DDiT operator) 조합. | arXiv:2504.12259 (VGDFR), arXiv:2510.18091 (APT), arXiv:2602.16968 (DDiT) |

---

## Blacklisted Patterns (자동 NO-GO)

### Pattern 1: Factorized Spatial-Temporal Attention for Video DiT
- **금지 이유**: Latte(2401.03048)가 4개 variant로 완전 탐색. FrameDiT(2603.09721, 2026-03)가 이를 "Local Factorized Attention = baseline to beat"으로 정의.
- **pixel space도 포함**: pixel vs latent는 mechanism level이 아님
- **키워드**: interleaved spatial/temporal blocks, factorized attention video, separated spatial temporal DiT

### Pattern 2: Published Mechanism + Domain Shift Only
- **금지 이유**: BP-Hybrid 사례 반복. 메커니즘은 A 논문에, 도메인은 B 논문에 이미 있을 때 조합만으로는 reviewer reject.
- **video에 특히 위험**: Latte = 모든 video DiT factorized attention 패턴의 원조

### Pattern 3: Caching Family (Video)
- **금지 이유**: AdaCache/PAB/TaoCache/BWCache/MixCache/ProfilingDiT 등 6+개 포화
- **[[caching-family-trap]] 참조**

### Pattern 4: Latent Video DiT Acceleration
- **금지 이유**: 이미 dominant. Sora/CogVideoX/Wan/HunyuanVideo 급 baseline 대비 의미있는 gain 불가능

### Pattern 6: Per-Frame Motion-Driven Patch Size Variation (FAP)
- **금지 이유**: VGDFR (2504.12259)가 동일 인과 체인(motion → per-frame token budget)을 latent space에서 구현. DDiT (2602.16968) 저자가 per-frame/region patch variation을 future work로 명시 → obvious next step으로 reviewer 즉시 reject. Pixel-only claim은 latent-FAP가 CogVideoX에서 50줄 구현 가능하므로 framing에 불과.
- **BP-Hybrid 패턴**: X(VGDFR motion signal) + Y(APT/DDiT patch resize operator) 조합
- **pixel space도 포함**: operator가 다를 뿐 signal과 mechanism은 동일
- **키워드**: per-frame adaptive patch size video, motion-driven patch budget, frame-adaptive tokenization video diffusion, variable patch video motion

### Pattern 5: Motion-Driven Temporal Sparsity in Video Diffusion
- **금지 이유**: Motion-Adaptive Temporal Attention (2603.17398, Mar 2026)이 latent SD 위에서 motion-driven temporal RF 변경을 완전 구현. ADAPTOR (CVPRW 2025)가 motion-based token reduction을 latent video에서 2.85× compute 달성. "Pixel space에서만 motion signal 가능" claim은 empirically 틀림 (2603.17398은 latent에서 motion 추정).
- **pixel space 포함**: pixel vs latent는 mechanism axis가 아닌 framing
- **키워드**: motion-aware temporal attention, motion token pruning video, static region skip attention video, motion-adaptive RF video diffusion
- **Salvage 가능 영역**: motion signal을 다른 mechanism(예: patch-size selection)의 input으로 사용 → 그러면 Pattern 5 회피

---

## Safe Zone (video idea에서 mechanism-level 차별화 가능한 방향)

아래는 blacklist가 아닌 **탐색 가능한 영역** (2026-05-19 현재):

1. **Pixel-space video distillation from latent teacher (CSD)** — 🟡 PIVOT REQUIRED (downgraded from NOVEL-LEANING). L2P (2605.12013, May 2026)가 "Latent-to-Pixel transfer paradigm" 이름을 선점. CSD 생존 조건: 수식 수정(Jacobian 방향 오류 해결) + video domain으로 명확한 구분 + 4개 Hard Gate 통과. 세부: `/jit-idea-validator/conditional/jit-video-csd_validation.md`
2. ~~Motion-aware temporal token sparsity~~ → **Pattern 5 BLACKLISTED 2026-05-19 (R2)**
3. ~~Frame-adaptive spatial resolution (Pivot C, FAP)~~ → **Pattern 6 BLACKLISTED 2026-05-19 (R2)** — VGDFR (2504.12259) + APT + DDiT "future work" 명시로 폐기.
4. **Pixel-space video consistency without VAE** — temporal consistency mechanism new to pixel domain (아직 탐색 안 됨)

---

## CSD-Specific Caveats (Pattern 7: Latent-to-Pixel Framing Preemption)

### Pattern 7: "Latent-to-Pixel Transfer Paradigm" Framing
- **금지 이유**: L2P (arXiv:2605.12013, May 2026)가 이 framing을 완전히 선점. "L2P transfer paradigm, an efficient framework that directly harnesses the rich knowledge of pre-trained LDMs to build powerful pixel-space models" — 이미 published.
- **Images only**: L2P는 이미지에만 적용. Video extension은 오픈 상태이나 L2P 저자들이 작업 중일 가능성 높음.
- **Salvage 조건**: (a) Score-level vs feature-level transfer의 mechanism-level 차이를 명시적으로 실험으로 증명, (b) Video temporal consistency에서 score-level이 필요한 이유 demonstrate, (c) Architecture-agnostic student (LDM backbone 재사용 없음)
- **Math HARD GATE**: 제안된 `J = ∂D/∂z` (decoder Jacobian)는 방향이 틀림. Chain rule for pixel→latent score projection은 encoder Jacobian `(∂E/∂x)^T`가 필요. 이것이 "Diff-Instruct with encoder in forward pass"와 동치이면 Pattern 2 → ABANDON.
- **키워드**: latent-to-pixel transfer, pixel video generation LDM teacher, cross-space score distillation video, VAE-free pixel video latent knowledge

---

*Last updated: 2026-05-19 KST*
