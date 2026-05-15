# JiT 적용 시도 (일반화 검증)

**핵심 발견: JiT는 monolithic 구조라 step-skip caching 이득이 PixelDiT보다 훨씬 작다.**

이는 DSTP가 PixelDiT의 dual-level (patch_blocks 87% + pixel_blocks 13%) 구조에 특화된 방법임을 시사한다.
JiT는 32(H/16) 또는 12(B/16) 개의 monolithic block만 있어, 전체 forward를 캐싱하는 것 외 분기가 없다.

## 실험 결과 (JiT-B/16, 50 step, 64장, B200)

### v1 (t_based 부등호 잘못)
| Config | median ms | speedup | 비고 |
|---|---|---|---|
| Baseline | 86.01 | 1.000× | |
| K3_tbased | 77.83 | 1.105× | t_based 방향 오류로 거의 효과 없음 |
| K3_periodic | 33.75 | 2.549× | 품질 미검증 (PoC) |

### v2 (t_based 수정: JiT는 t=0이 노이즈)
| Config | median ms | std | speedup | 비고 |
|---|---|---|---|---|
| Baseline | 48.17 | 5.34 | 1.000× | |
| K3_tbased | 325.75 | **198.4** | 0.15× | std 60% — torch.compile cache miss 의심 |
| K3_periodic | 42.50 | 22.45 | 1.13× | speedup 미미 |

## 분석

### v2 K3_tbased가 비정상적으로 느린 이유 (가설)
1. JiT-B/16의 JiTBlock은 `@torch.compile` 데코레이터 사용
2. 캐시 wrapper의 `_should_refresh` 분기가 compile graph를 깨뜨림
3. 매번 다른 분기를 타면서 recompilation cache miss 발생
4. std=198ms (median 60%)가 강한 증거

### PixelDiT vs JiT 구조 차이
- **PixelDiT-XL**: patch_blocks (26 layers, 87% 시간) + pixel_blocks (4 layers, 13% 시간)
  → patch_blocks만 캐싱하고 pixel_blocks는 매 step 실행 → texture refinement 보존
- **JiT-B/16**: 12 monolithic blocks (단일 trunk)
  → 전체 forward 캐싱이 유일 옵션 → low-noise step에서 detail 손실 위험 (또는 위처럼 compile miss)

### 논문 시사점
DSTP의 효과는 **단순 step caching이 아니라 architecture-aware caching**에서 온다.
PixelDiT는 patch_blocks/pixel_blocks 분할 덕분에 patch만 cache하고 pixel은 detail refinement에 활용 가능 →
이것이 quality 보존의 핵심 메커니즘.

JiT 류 monolithic 모델에는 직접 적용 어렵고, 다른 접근 필요:
- N개 block만 cache하고 마지막 N개 block은 항상 실행 (DeepCache style — JiT용 추가 실험 필요)
- Block-level partial caching (JiT-H/16 32 blocks를 24+8로 분할)

## 파일 구조
```
experiments/jit_dstp/
  jit_stepskip.py       — JiT용 step-skip wrapper + euler 샘플러
  results.json          — v2 (수정본) 결과
  jit_stepskip.log      — v1 로그
  jit_stepskip_v2.log   — v2 로그
  README.md             — 이 파일
/data/jameskimh/jit_dstp/
  JiT-B_16/             — v1 시각 샘플
  JiT-B_16_v2/          — v2 시각 샘플
```
