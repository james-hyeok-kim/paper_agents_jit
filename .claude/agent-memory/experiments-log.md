---
name: experiments-log
description: 실험 실행 결과 로그 (날짜순)
metadata:
  type: feedback
---

# Experiment Execution Log

## 2026-05-15 — RATBA Sanity Probe

**슬러그:** ratba  
**아이디어:** jit-b-16 (256 tokens) + jit-b-32 (64 tokens) 캐스케이드 샘플링  
**결과:** FAILED (NO-GO)  
**이유:**
- jit-b-32 체크포인트의 `pos_embed=[1, 256, 768]` — 이 모델은 img_size=512로 학습됨 (512/32=16, 16²=256 tokens)
- 모든 JiT 체크포인트가 256 tokens 사용 (코스-파인 캐스케이드 불가)
- FID 측정도 무효 (torchvision Inception V3 ≠ torch_fidelity TF-FID Inception V3 extractor mismatch)

**타이밍 (유효):** jit-b-16, 341 ms/img, B200, batch=16, 10 Euler steps, CFG=3.0  
**샘플 저장:** `/data/jameskimh/ratba/sanity_jit-b-16/` (1000장, 유효)

---

## 2026-05-15 — DSTP Step-Skip PoC

**슬러그:** dstp  
**아이디어:** PixelDiT-XL patch_blocks를 K step마다 한 번만 실행하고 cached s 재사용  
**결과:** GO  

**GPU:** NVIDIA B200  
**모델:** PixelDiT-XL (patch_depth=26, pixel_depth=4, hidden=1152, patch_size=16)  
**설정:** 256×256, 20 steps FlowDPMSolver, CFG=3.25, 128장  

| 설정 | speedup | IS drop |
|---|---|---|
| K=2 periodic | 1.37× | 0.212 |
| K=3 periodic | 2.12× | 0.272 |
| K=2 t_based | 1.28× | 0.026 |
| K=3 t_based | 1.53× | 0.095 |

**추천 설정 (GO):** K=3 t_based (1.53×, IS drop 0.095, 시각 품질 baseline과 동일)  
**K=3 periodic (NO GO):** 2.12× speedup이나 시각적 composition 손실 심각 (배경 객체 소실)  

**프로파일:**
- patch_blocks: ~87% of total runtime
- pixel_blocks: ~13% of total runtime
- Token-level pruning은 실패 (gather/scatter overhead > savings, 0.785× = 27% 느림)

**시각 검증 (2026-05-15 추가):**
- K3_periodic: 00000/00006/00010에서 구성 손실 확인. IS-at-20-steps은 이 손상을 탐지 못함
- K3_tbased: 4개 인덱스 전체 baseline과 시각적 동일. **실질적 GO 설정**
- K2_periodic: 00006 배경 소실 확인 (K3_periodic과 동일 패턴)
- IS drop 임계값(≤5.0)은 단독으로 품질 보증 불충분 — 시각 검증 필수

**코드:** `/home/jovyan/workspace/paper_agents_jit/experiments/dstp/dstp_sampler.py`  
**결과 JSON:** `/home/jovyan/workspace/paper_agents_jit/experiments/dstp/results.json`  
**샘플:** `/data/jameskimh/dstp/{baseline,K2_periodic,K3_periodic,K2_tbased,K3_tbased}/`

**주의사항:**
- B200 타이밍 std가 큼 (~100+ ms/img) — median 기준 판단
- IS 절대값 ~11은 20 steps 때문 (canonical 100 steps 사용 시 높아짐)
- FID 측정 보류 (ImageNet val reference set 없음)

---

## 공통 환경 노트

**FID 측정 이슈:**
- `torch_fidelity==0.3.0`은 `fid_statistics_file` 파라미터 미지원
- 사전계산 `.npz` stats는 TF-FID Inception V3 사용 → torchvision Inception V3와 다름
- 유효한 FID 측정법: `torch_fidelity.calculate_metrics(input1=gen_dir, input2=imagenet_val_dir, fid=True, isc=True)`
- ImageNet val 이미지가 없으면 IS만 사용

**체크포인트 경로:**
- JiT: `/data/jameskimh/james_jit_pretrained/jit-h-16/{variant}/checkpoint-last.pth`
- PixelDiT-XL: `/data/jameskimh/pixeldit_pretrained/imagenet256_pixeldit_xl_epoch320.ckpt` (Lightning ckpt, ema_denoiser. prefix)

**모델 구조 확인:**
- JiT 모든 체크포인트: `pos_embed=[1, 256, 768]` (256 tokens)
- PixelDiT-XL: patch_size=16, patch_depth=26, pixel_depth=4, hidden=1152, in_channels=3 (pixel space, no VAE)
