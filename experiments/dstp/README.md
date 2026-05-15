# DSTP: Dynamic Step-wise Token Pruning (PixelDiT-XL)

**상태: GO — 추천 설정: K=3, t_based**

## 실험 아이디어

PixelDiT-XL의 `patch_blocks` (26개 Transformer 블록) 연산을 매 K번째 step에서만 실행하고, 나머지 step에서는 캐시된 `s` 벡터를 재사용. **Step-Skip** 기법으로 denoising 속도를 높이면서 품질 저하를 최소화.

## 초기 접근 (Token-level Pruning) → 실패

- gate_msa norm으로 token importance 측정 후 하위 r% 토큰의 patch_blocks skip 시도
- **결과: 0.785× speedup (오히려 27% 느림)**
- 원인: gather/scatter + topk overhead가 attention 절약보다 큼
- B200 + bf16 환경에서 256 token 규모의 sparse gather는 dense 대비 비효율적

## 구조 분석

| 컴포넌트 | 런타임 비율 |
|---|---|
| patch_blocks (26 layers) | ~87% |
| pixel_blocks (4 layers) | ~13% |
| 기타 (embedder, sampler) | < 1% |

→ **patch_blocks가 bottleneck**: 이를 통째로 skip하는 것이 유일한 효과적 접근

## Step-Skip 설계

```python
def forward(self, x, t, y):
    if step_count % K == 0 or cache is None:
        s = patch_blocks(x)  # full computation
        cache = s
    else:
        s = cache  # reuse cached patch features
    out = pixel_blocks(x, s)  # always run
    step_count += 1
    return out
```

두 가지 refresh policy:
- **periodic**: 단순 카운터 (step % K == 0이면 refresh)
- **t_based**: t > T_split 구간(고노이즈)은 K step마다 skip, t <= T_split(저노이즈) 구간은 매 step full compute

## 실험 결과 (PoC)

**설정:** PixelDiT-XL, 256×256, 20 steps (FlowDPMSolver), CFG=3.25, 128장  
**GPU:** NVIDIA B200  
**타이밍 주의:** B200에서 std가 매우 큼 (일부 설정에서 std > median). 아래 수치는 안정 구간 median 기준. 방향성은 FLOP-skip 이론치와 일치하나 정밀한 절대값으로 해석하지 말 것.

| 설정 | ms/img (median) | speedup | IS | IS drop | 시각 품질 | 판정 |
|---|---|---|---|---|---|---|
| Baseline | 138.1 | 1.000× | 11.093 | 0.000 | 기준 | — |
| K=2, periodic | 100.5 | 1.37× | 10.881 | 0.212 | 배경 손실 있음 | WEAK GO |
| K=3, periodic | 65.3 | **2.12×** | 10.821 | 0.272 | **구성 및 배경 손실 심각** | NO GO |
| K=2, t_based | 107.6 | 1.28× | 11.068 | 0.026 | baseline과 거의 동일 | GO |
| **K=3, t_based** | **90.2** | **1.53×** | **10.998** | **0.095** | **baseline과 시각적으로 동일** | **GO (추천)** |

## 시각적 품질 검증

4개 클래스 인덱스(00000, 00003, 00006, 00010)에서 baseline vs 각 설정 1:1 비교 수행.

**K3_periodic 문제점 (시각 확인됨):**
- 00000 (잉어): 구도 완전히 다름, 사람 손과 반지 디테일 소실
- 00006 (가오리): **심각** — 두 번째 가오리 소실, kelp 배경 전체 소실
- 00010 (새): 깃털 미세 디테일 블러
- 00003 (상어): 블러지만 인식 가능

**K2_periodic 문제점:**
- 00006 (가오리): K3_periodic과 동일하게 두 번째 가오리 및 배경 소실

**K3_tbased (추천):**
- 4개 인덱스 전체에서 baseline과 시각적으로 구분 불가
- 저노이즈 구간(t ≤ 0.5) = 디테일 형성 단계에서 full compute 보장, 구성 손실 없음

**결론:** IS 20-steps/128장 지표는 K3_periodic의 시각 품질 저하를 탐지하지 못함. IS drop 기준만으로는 부족하며, 시각 검증이 필수.

## 주요 발견

1. **K=3 t_based가 실질적 GO** (1.53×, 시각 품질 보존)
2. **K=3 periodic은 speedup(2.12×)은 크나 시각 품질 손상** — 논문 기본 설정으로 부적합
3. Token-level pruning과 달리 step-skip은 오버헤드 없음 (분기 + 캐시 재사용만)
4. IS 20 steps/128장 지표는 신뢰도 낮음 — FID-50K (100 steps) 측정 필요

## FID 측정 현황

- **FID 측정 보류**: ImageNet validation reference set 없음
- torch_fidelity 0.3.0 버전은 `fid_statistics_file` 미지원
- 유효한 FID = `torch_fidelity.calculate_metrics(input1=gen_dir, input2=imagenet_val_dir, fid=True)`
- 현재 IS만으로 방향 판단; 논문 최종 수치는 FID-50K (100 steps) 필요

## 다음 단계

1. **FID-2K 측정**: ImageNet validation 2K장 reference 확보 후 K3_tbased vs baseline 비교
2. **100 steps canonical 측정**: 공식 샘플러 설정으로 재측정 (IS ~11 → 더 높아질 것)
3. **K3_tbased 대용량 검증**: 1K+ 샘플로 IS/FID 안정적 측정
4. **T_split 튜닝**: T_split=0.3/0.7 조건 추가 실험 → 최적 threshold 탐색

---

*실험 날짜: 2026-05-15*  
*실험 코드: `/home/jovyan/workspace/paper_agents_jit/experiments/dstp/dstp_sampler.py`*  
*결과 JSON: `/home/jovyan/workspace/paper_agents_jit/experiments/dstp/results.json`*  
*샘플 경로: `/data/jameskimh/dstp/{baseline,K2_periodic,K3_periodic,K2_tbased,K3_tbased}/`*
