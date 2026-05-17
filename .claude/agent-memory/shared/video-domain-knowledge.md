---
name: video-domain-knowledge
description: Pixel-space Video DiT 도메인 지식 - 모든 agent가 참조
metadata:
  type: reference
---

# Pixel-Space Video DiT 도메인 지식 (2026-05-16 수립)

## 핵심 motivation
이미지 영역(JiT/PixelDiT)은 포화. **Video 영역, 특히 pixel-space video DiT는 거의 미탐색** — 이쪽으로 pivot.

---

## 1) 현재 Video Generation 지형도

### A. Latent Video DiT (경쟁자 — 압도적 dominant)
| 모델 | 출시 | 특징 |
|------|------|------|
| **Sora** | 2024, OpenAI | 3D autoencoder + spacetime patches, 1분 video |
| **CogVideoX** | 2024-2025 | open-source 5B, 8×8×8 VAE 압축 |
| **HunyuanVideo** | 2024, Tencent | 13B open, 5초 720p |
| **Wan / Wan2.1** | 2024-2025, Alibaba | T2V/I2V 통합 |
| **Mochi** | 2024 Genmo | 10B open |
| **LTX-Video** | 2025 Lightricks | realtime 2x faster than playback |
| **MovieGen** | 2024 Meta | 30B, 16초 |
| **Open-Sora / Open-Sora-Plan** | 2024- | open Sora-like |
| **Kling / Gen3** | 2024+, 상용 | quality leader |
| **Veo 2/3** | 2024+, Google | quality leader |
| **WAN/SemanticGen/PyramidFlow** | 2024-2025 | semantic 압축 |

**공통점**: 모두 latent (VAE 8×8×8 압축, 1:1024 ~ 1:2048 pixel-to-token), spacetime patches, full attention or window attention.

### B. Pixel-Space Video Diffusion (✅ 거의 비어있음 — opportunity!)
| 모델 | 출시 | 비고 |
|------|------|------|
| **Video Diffusion Models** (Ho et al.) | 2022 | 원조 pixel video diffusion, 작은 해상도 |
| **Imagen Video** | 2022 Google | Cascaded pixel diffusion, 7 sub-models, 1280×768@24fps |
| **Make-A-Video** | 2022 Meta | mostly latent |
| **Tuna-2** (arxiv:2604.24763) | 2026 | encoder 없이 raw pixel video token 처리 (multimodal) |

**관찰**: 2022 이후 거의 모든 video model이 latent로 갔음. 이유: pixel-space video는 token explosion (시간×공간) 때문에 학습/추론 비용 매우 큼.

**미해결 질문**: PixelDiT/JiT의 image-side 성공(2025-2026)이 video에도 적용 가능한가?

### C. Video AR / Discrete (또 다른 경쟁자)
- VideoPoet (Google, 2024) — discrete token AR
- MAGVIT-v2 (2024) — video tokenizer + AR
- W.A.L.T. (2024) — windowed AR

---

## 2) Video Diffusion 가속 Family (포화 — 자동 NO-GO 후보)

| 논문 | 연도 | 가속 |
|------|------|------|
| **PAB** (Pyramid Attention Broadcast) | 2024 | 1.66× on Open-Sora |
| **AdaCache** (ICCV 2025) | 2024 | **2.61×** on Open-Sora 720p-2s |
| **TaoCache** (2508.08978) | 2025 | structure-maintained |
| **BWCache** (2509.13789) | 2025 | 1.61× block-wise |
| **MixCache** (2508.12691) | 2025 | mixture-of-cache, adaptive hybrid |
| **ProfilingDiT** | 2025 | profile-based adaptive cache |

→ **Video caching family는 image와 똑같이 포화**. 새 caching idea 자동 NO-GO.

---

## 3) Video Diffusion 평가 지표

### 표준 metric
- **FVD (Fréchet Video Distance)** — FID의 video 버전, I3D features
- **VBench** — comprehensive benchmark suite (16+ dimensions)
- **CLIPSIM** — text-video alignment
- **IS-V** (Inception Score for video)
- **LPIPS / PSNR / SSIM** — per-frame quality
- **Temporal consistency** — flicker, optical flow consistency

### 표준 데이터셋
- **UCF-101** (작음, 옛날) — 13K videos, 101 classes
- **Kinetics-600/700** — 500K+, 600/700 actions
- **WebVid-10M** — 10M text-video
- **Panda-70M** — large-scale
- **HowTo100M** — 100M
- **OpenVid-1M** — 최근 open dataset
- **MiraData** — quality-filtered

### Resolution × Frames 표준
- 256×256 × 16 frames (작은 PoC)
- 512×512 × 16 frames (중간)
- 720×1280 × 49-129 frames (full quality)
- pixel video는 후자 거의 불가능 → smaller scale가 현실적

---

## 4) Pixel-Space Video 특유의 도전

| 도전 | 설명 |
|------|------|
| **Token explosion** | 256×256×16f = 1M+ tokens (이미지 16K 대비 64×) |
| **Memory** | full attention O((HWT)²) → 1M² = 1T 메모리, 사실상 불가능 |
| **Temporal consistency** | frame 간 flicker/inconsistency |
| **Training data** | text-video pair는 image보다 적음 + low quality |
| **Compute** | 4×A100/B200으로 1 frame당 분 단위 |

→ Latent space가 dominant인 이유 명백. **Pixel-space는 이 비용을 어떻게 해결하느냐가 관건.**

---

## 5) Pixel-Space Video DiT가 진짜로 paper 가능한 방향

### 🎯 Truly underexplored
1. **Pixel-space video DiT의 첫 baseline** — image PixelDiT를 video로 확장한 첫 paper
2. **Cascaded pixel video** — Imagen Video 2022 → 2026 modern transformer 버전
3. **Temporal token compression in pixel space** — VAE 없이 frame diff/motion compression
4. **Pixel-space video distillation** — latent video 모델에서 pixel video로 distill
5. **Frame-adaptive resolution** — motion 많은 frame은 high-res, 적은 frame은 low-res
6. **Window-only attention for pixel video** — full attention 불가하니 spatial-temporal window만

### 🔴 자동 NO-GO 영역 (video도 포함)
- Caching family (AdaCache/PAB/TaoCache/BWCache/MixCache/ProfilingDiT 등)
- Latent space efficiency (이미 dominant, 새 게 들어갈 자리 없음)
- 단순 image method를 video로 변환만 (model swap only)

---

## 6) 새 family checklist (video 특화)

새 video idea 평가 시:
- [ ] Caching family (위 6+ video caching paper)와 충돌 안 함
- [ ] Latent video DiT 가속과 mechanism 다름
- [ ] Sora/CogVideoX/Wan과 paradigm 다름 (단순 더 빠른 latent video DiT면 무의미)
- [ ] Pixel-space의 본질적 challenge (token explosion) 다룸
- [ ] FVD/VBench로 평가 가능한 결과 산출 가능
- [ ] 학습 비용 합리적 (B200 4개로 가능한 PoC)

---

## 7) 우선순위 (paper-가능성 × 임팩트)

### Tier 1 (높은 가능성)
- Pixel-space video DiT baseline (첫 paper potential — 단, Tuna-2 등 web 확인 필수)
- Temporal token compression (motion-aware)
- Hybrid latent + pixel (latent for early steps, pixel for refinement) — but check HART 류

### Tier 2 (중간)
- Frame-adaptive computation (motion-conditional)
- Cascaded pixel video modernization
- Window-only attention design

### Tier 3 (낮음 / 위험)
- 단순 image method의 video extension
- Latent video DiT 가속 (포화)
- Caching family 변형 (포화)

---

## 8) Image 영역 reference (legacy, 참고용)

기존 JiT/PixelDiT image 영역 prior art는 여전히 유효:
- JiT (2511.13720), PixelDiT (2511.20645), DiP (2511.18822)
- EPG (2510.12586), DeCo (2511.19365), PixelGen (2602.02493)
- PixelFlow (2504.07963), FREPix (2605.06421), HDiT (2401.11605)
- BDPM (2501.13915), HART (2410.10812)

→ Video idea에서 "image method의 video extension"이면 위 image paper와 직접 충돌

[[caching-family-trap]] / [[sacrificed-ideas]] 도 video에 그대로 적용
