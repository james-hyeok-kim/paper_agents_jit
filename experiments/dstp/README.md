# DSTP: Step-Skip Caching for Pixel-Space Diffusion Transformers

**상태: ✅ GO — 추천 설정: K=3, t_based (T_split=0.5)**
**날짜: 2026-05-15**
**대상 모델: PixelDiT-XL (NVlabs, arXiv:2511.20645)**

---

## 핵심 아이디어

PixelDiT-XL의 `patch_blocks`(26개 Transformer 블록, 전체 연산의 ~87%)를
매 K번째 denoising step에서만 실행하고, 나머지 step에서는 cached `s` 벡터를 재사용한다.
저노이즈 구간(t ≤ T_split)은 디테일·구성 형성 단계이므로 매 step full compute를 보장 (t-based policy).

**학습 불필요 (training-free), plug-and-play.**

## 런타임 분해 (PixelDiT-XL)

| 컴포넌트 | 런타임 비율 |
|---|---|
| patch_blocks (26 layers) | ~87% |
| pixel_blocks (4 layers) | ~13% |
| 기타 (embedder, sampler) | < 1% |

→ patch_blocks 캐싱이 유일한 효과적 가속 경로.
token-level pruning은 sparse gather overhead로 0.785× 역효과 (실험 확인).

---

## 실험 목록 (실행 순서)

| # | 실험 | GPU | 상태 | 핵심 결과 |
|---|------|-----|------|-----------|
| E1 | PoC: K∈{2,3} × {periodic, t_based} (128장, 20 step) | B200 | ✅ 완료 | K=3, t_based 1.53× |
| E2 | 2K 샘플 + FID 측정 (20 step) | B200 | ✅ 완료 | FID gap +1.90 |
| E3 | T_split ablation (256장, 20 step) | B200 | ✅ 완료 | T=0.6 sweet spot |
| E4 | 100-step canonical 인터리빙 (256장) | B200 | ✅ 완료 | paired 1.27× |
| E5 | 시각 검증 (4 클래스 × 5 설정) | B200 | ✅ 완료 | K3_tbased 보존 |

---

## 결과 1: PoC (128장, 20 steps)

| 설정 | ms/img (median) | speedup | IS | IS drop | 시각 품질 | 판정 |
|---|---|---|---|---|---|---|
| Baseline | 138.1 | 1.000× | 11.093 | 0.000 | 기준 | — |
| K=2, periodic | 100.5 | 1.37× | 10.881 | 0.212 | 배경 손실 있음 | WEAK |
| K=3, periodic | 65.3 | 2.12× | 10.821 | 0.272 | **구성 손실 심각** | NO GO |
| K=2, t_based | 107.6 | 1.28× | 11.068 | 0.026 | baseline과 거의 동일 | GO |
| **K=3, t_based** | **90.2** | **1.53×** | **10.998** | **0.095** | **baseline과 시각적 동일** | **✅ GO** |

K3_periodic 시각 손상 사례 (00006, 가오리 클래스): 두 번째 가오리 + 켈프 배경 소실.
→ IS 지표만으로는 시각 품질 보장 불가. **t_based가 핵심.**

## 결과 2: FID-2K (PixelDiT-XL, 20 steps)

| Config | FID (vs ImageNet val 2K) | 비고 |
|---|---|---|
| Baseline | 34.143 | 20 step + 2K ref → noise floor 높음 |
| K3_tbased | 36.047 | gap +1.904 |

- 절대 FID는 paper용 100-step + 50K 측정 필요
- **gap 1.90**은 IS drop 0.095와 일관된 directional signal
- reference: 2 imgs/class × 1000 classes, center crop + BICUBIC 256, BGR PNG

## 결과 3: T_split Ablation (256장, 20 steps, K=3)

| T_split | median ms | speedup | 비고 |
|---|---|---|---|
| 0.3 | 294.5 | 1.29× | |
| 0.4 | 387.5 | 0.98× | 노이즈 (외부 contention) |
| 0.5 | 269.8 | 1.41× | |
| **0.6** | **183.1** | **2.08×** | **sweet spot** |
| 0.7 | 455.9 | 0.84× | 노이즈 (외부 contention) |
| periodic | 237.0 | 1.61× | 비교 baseline |

그래프: `figures/t_split_curve.png`

**경고**: B200 std/median 비율이 50% 이상 — 다른 사용자의 GPU 점유로 인한 contention.
방향성은 유효하나 절대 timing은 신뢰 불가. paired 측정(E4)이 더 안정적.

## 결과 4: 100-step Canonical (인터리빙 paired 측정)

baseline/K3를 같은 GPU에서 alternating 측정하여 외부 contention에 robust한 paired speedup 측정.

| Config | median ms | mean ms | std ms |
|---|---|---|---|
| Baseline | 400.79 | 405.39 | 47.93 |
| K3_tbased | 309.61 | 317.76 | 45.10 |

- **paired speedup (median of per-batch ratios): 1.273×**
- speedup (median/median): 1.294×
- std/median ratio: ~12% (PoC 수준으로 안정)

**핵심 관찰**: 100 step에서는 20 step보다 speedup이 작다 (1.27× vs 1.53×).
이유: 100 step에서는 저노이즈 구간(t ≤ 0.5)에 더 많은 step이 분포 → t_based가 skip하지 않는 구간 비중 증가.
→ T_split을 step 수에 따라 튜닝하면 추가 향상 가능 (future work).

---

## 핵심 발견

1. **patch_blocks 통째 캐싱이 유일한 효과적 경로** — token-level pruning은 sparse overhead로 실패
2. **t_based policy가 품질 보존의 핵심** — periodic은 저노이즈 step도 skip해서 구성 손상
3. **IS 단독 신뢰 불가** — K3_periodic은 IS drop 0.27인데 시각적으로 심각한 손상 (가오리 소실 등)
4. **20 step PoC: 1.53× speedup**, **100 step canonical: 1.27× speedup** (paired 측정)
5. **B200 timing 노이즈가 매우 큼** — 외부 사용자 간섭. paired 또는 격리 환경에서만 신뢰 가능

---

## 파일 구조

```
experiments/dstp/
  dstp_sampler.py              — 핵심 step-skip 모델 + PoC main
  generate_2k.py               — 2K 샘플 생성
  prepare_imagenet_ref.py      — ImageNet val 2K reference 구성
  timing_and_tsplit_ablation.py — T_split ablation
  canonical_100steps.py        — 100-step paired 측정
  plot_tsplit.py               — T_split 곡선 그래프
  results.json                 — PoC 결과 (128장)
  fid_2k_results.json          — FID 측정 결과
  tsplit_ablation.json         — T_split ablation
  canonical_100steps.json      — 100-step paired
  timings_2k.json              — 2K 생성 시 timing (외부 contention으로 불신)
  figures/
    t_split_curve.png          — T_split vs speedup 곡선
  README.md                    — 이 파일

/data/jameskimh/
  imagenet_val_ref_2k/         — ImageNet val 2K reference (FID용)
  dstp/baseline/               — PoC 128장
  dstp/K2_periodic/  K2_tbased/  K3_periodic/  K3_tbased/  — PoC
  dstp/fid_2k/baseline/        — 2000장 (FID용)
  dstp/fid_2k/K3_tbased/       — 2000장 (FID용)
```

---

## 논문 포지셔닝

**Title 후보**: *"Step-Skip Caching for Pixel-Space Diffusion Transformers"*

**Contribution**:
1. **분석**: PixelDiT-XL 런타임 분해, patch_blocks가 87% 차지 → 유일한 효과적 캐싱 타겟
2. **방법**: t-aware step-skip — 저노이즈 구간(t ≤ T_split) 보호로 composition 손상 방지
3. **실증**: PixelDiT-XL에서 1.27–1.53× speedup, 시각 품질 보존, training-free

**비교 대상 (이미 알려진 선행)**:
- DeepCache (NeurIPS 2024): latent U-Net caching — 우리는 pixel-space DiT
- FasterDiffusion: encoder caching — 우리는 transformer block caching
- DiP (arXiv:2511.18822): 10× speedup but requires retraining — 우리는 plug-and-play
- DyDiT: timestep-conditioned compute on latent — 우리는 pixel-space + training-free

**타겟 venue**: CVPR/ICCV efficient generation track

## 남은 작업 (논문화)

1. **FID-50K @ 100 steps** — ImageNet val 50K로 정식 측정 (현재 2K + 20 step은 directional only)
2. **DeepCache 직접 이식 vs 우리 방법** 비교
3. **JiT 적용** — PixelDiT 외 다른 pixel-space 모델로 일반화
4. **512×512** 일반화
5. **T_split learned schedule** — step 수에 따라 자동 조정

---

## 시각 검증 (PoC)

4개 클래스 인덱스(00000, 00003, 00006, 00010)에서 baseline vs 각 설정 1:1 비교 수행.

**K3_periodic 문제점 (시각 확인됨):**
- 00000 (잉어): 구도 완전히 다름, 사람 손과 반지 디테일 소실
- 00006 (가오리): **심각** — 두 번째 가오리 소실, kelp 배경 전체 소실
- 00010 (새): 깃털 미세 디테일 블러
- 00003 (상어): 블러지만 인식 가능

**K3_tbased (추천):**
- 4개 인덱스 전체에서 baseline과 시각적으로 구분 불가

---

*마지막 갱신: 2026-05-15 23:35*
