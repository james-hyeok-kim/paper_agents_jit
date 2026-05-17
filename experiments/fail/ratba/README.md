# RATBA: 실험 결과 요약

**상태: FAILED (NO-GO)**

## 실험 아이디어

JiT-B/16 (256 tokens, 미세) + JiT-B/32 (64 tokens, 거친) 두 모델을 이용한 캐스케이드 샘플링:
- 초기 고노이즈 스텝은 저해상도 토큰(b-32)으로 빠르게 처리
- 후반 저노이즈 스텝에서 고해상도 토큰(b-16)으로 전환하여 품질 복원
- 목표: 1.5× 이상 속도 향상, FID gap < 5.0

## STEP 1: Sanity Probe 결과

### JiT-B/16 생성 결과

| 항목 | 값 |
|---|---|
| GPU | NVIDIA B200 |
| 해상도 | 256×256 |
| 샘플 수 | 1,000장 |
| 배치 크기 | 16 |
| Euler steps | 10 |
| CFG | 3.0 |
| 생성 시간 | 341.1s |
| 속도 | 2.93 img/s, 341 ms/img |

**FID 측정 결과 (무효):**
- 측정된 FID: 175.38 (IS: 50.18)
- 무효 이유: `torchvision.models.inception_v3` (ImageNet-pretrained)와 사전계산된 `.npz` 통계 (torch_fidelity TF-FID Inception V3)가 서로 다른 특징 공간을 사용함
- 유효한 FID 측정은 `torch_fidelity.calculate_metrics(input1=gen_dir, input2=imagenet_val_dir)` 형태로 동일한 extractor를 양쪽에 사용해야 함

## STEP 1: 치명적 발견 - RATBA 불가능

**jit-b-32 체크포인트 shape mismatch:**
```
RuntimeError: size mismatch for net.pos_embed:
  checkpoint shape: torch.Size([1, 256, 768])   ← 256 tokens!
  model expects:    torch.Size([1, 64, 768])     ← expected for 256^2 / 32^2
```

**근본 원인 분석:**
- `JiT-B/32`는 patch_size=32로 학습되었지만, **img_size=512** 기준으로 학습됨
- 512 / 32 = 16 → 16×16 = **256 tokens** (256² 이미지에서 기대하는 64개가 아님)
- 사용 가능한 모든 체크포인트 (`jit-b-16`, `jit-b-32`, `jit-l-16`, `jit-l-32`, `jit-h-16`, `jit-h-32`) 모두 `pos_embed=[1, 256, N]` 형태

| 체크포인트 | 실제 img_size | 토큰 수 |
|---|---|---|
| jit-b-16 | 256 (16×16 grid) | 256 |
| jit-b-32 | 512 (16×16 grid) | 256 |
| jit-l-16 | 256 (16×16 grid) | 256 |
| jit-l-32 | 512 (16×16 grid) | 256 |

**결론:** 토큰 수가 다른 두 체크포인트가 존재하지 않으므로 캐스케이드 샘플링 불가능.

## 판정: FAILED (NO-GO)

- 캐스케이드를 위해 필요한 "저해상도 토큰 모델"이 없음
- 대안 실험: **DSTP** (PixelDiT-XL attention entropy 기반 토큰 프루닝)으로 전환

---

*실험 날짜: 2026-05-15*
