---
name: domain-pivot-2026-05-16
description: Pixel-Space Image DiT → Pixel-Space Video Generation pivot 결정 (2026-05-16)
metadata:
  type: project
---

# Domain Pivot: Image PixelDiT → Pixel-Space Video Generation

## 결정 배경
이전 작업 ([[sacrificed-ideas]], [[caching-family-trap]]) 으로 image PixelDiT 영역의 포화 확인.

**Image PixelDiT inference 가속**:
- 12개 idea 2 batches 모두 paper 가능성 낮음
- caching family 16+ papers
- DSTP/BP-Hybrid/SNR-Scaling 모두 web search로 prior art 발견
- 결론: 영역 자체가 saturated, 새 진입 매우 어려움

## 새 도메인: Pixel-Space Video Generation

### 왜 이쪽?
1. Latent video DiT 매우 dominant (Sora/CogVideoX/Wan/HunyuanVideo/Mochi/Veo) — 빅테크 영역
2. **Pixel-space video 거의 비어있음** — Imagen Video (2022) 이후 사실상 없음, Tuna-2 (2026) 정도
3. Image PixelDiT (JiT/PixelDiT) 성공이 video에 generalize 되는지 unanswered
4. B200 4× 환경에 적합한 scale (작은 PoC 가능)

### 도메인 지식
모든 video 작업은 [[video-domain-knowledge]] 참조.

### 변경 사항
- 모든 agent에 "DOMAIN PIVOT (2026-05-16)" 섹션 추가
- 공통 [[video-domain-knowledge]] 메모리 참조
- 기존 image domain은 legacy reference로 유지 (video idea의 baseline/대조용)

### 우선 ideation 방향
- Pixel-space video DiT의 첫 principled baseline
- Temporal token compression (motion-aware)
- Cascaded pixel video 현대화 (Imagen Video → 2026 transformer)
- Frame-adaptive computation
- Window-only spatiotemporal attention
- Latent → pixel video distillation

### 자동 회피
- Video caching family (AdaCache, PAB, TaoCache, BWCache, MixCache, ProfilingDiT)
- 단순 image method의 video 확장 (incremental)
- Latent space efficiency (포화)

### Next step
새 video ideation round: jit-idea-generator로 video-specific 12개 idea 생성 → validator → paper-feasibility-checker (web search 강제).
