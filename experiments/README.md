# JiT Inference Efficiency — Experiments

JiT (Just-in-Time) DiT 가속 연구 실험 모음. B200 GPU 환경.

## Folder Structure

실험은 결과 상태별로 분류:

```
experiments/
├── README.md          ← this file
├── fail/              ← polished off / abandoned ideas
│   └── ratba/         🔴 FAILED (NO-GO, sanity probe)
├── wip/               ← work-in-progress experiments
│   └── jit_dstp/      ⚠️ JiT 일반화 시도 — monolithic 구조라 PixelDiT 대비 이득 작음
└── success/           ← published / publishable experiments
    └── dstp/          ✅ GO — Step-Skip Caching for PixelDiT-XL (K=3, t_based, ~3× speedup)
```

각 실험 폴더 내부:
```
{name}/
├── README.md          # 실험 목적, 방법, 결과 요약
├── *.py               # 실험 스크립트 (PoC, ablation, full)
├── *.log              # 실행 로그
├── *.json             # 측정값 (FID, latency 등)
└── figures/           # 시각화
```

## 아이디어 명세 위치

각 아이디어의 상세 명세, 문헌 검토, validation 결과:
- `.claude/agent-memory/jit-idea-*/`

전체 idea 상태:
- `.claude/agent-memory/jit-doc-organizer/` (있다면)

## Top-level docs

- `SESSION_*.md` — 세션 요약
- `docs/{fail,wip,success}/` — 아이디어별 정리 문서 (동일 분류)
