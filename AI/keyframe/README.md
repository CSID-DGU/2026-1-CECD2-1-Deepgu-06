# AI Pipeline (TSM + VLM)

TSM 기반 1차 필터링 → Keyframe 추출 → InternVL2 판정으로 이어지는 이상행동 감지 파이프라인.

## 데이터 흐름

```
입력 영상
  → clip_generator.py     : 16 frame 클립으로 분할 (stride 8)
  → models/tsm/           : ResNet + TSM으로 클립별 이상행동 확률 산출
  → scorer.py / filter.py : 확률 → 점수 변환, threshold(0.4) 필터링
  → sampler.py            : 후보 클립에서 keyframe 선택
  → models/vlm/           : InternVL2로 이상행동 최종 판정
  → results.json          : 결과 저장
```

## Keyframe 선택기

| 방식 | 클래스 | 설명 |
|------|--------|------|
| Phase 1 | `KeyframeSampler(model_path=None)` | K-means 시간 구간 + Motion proxy |
| Phase 2 | `KeyframeSampler(model_path=...)` | BiGRU FrameScorer (pseudo-label 학습) |
| PGL-SUM | `PGLSumSampler(model_path=...)` | PGL-SUM + Anomaly Proxy |

## 디렉토리 구조

```
ai_pipeline/
├── pipeline/
│   ├── main_pipeline.py       # 전체 파이프라인 진입점
│   ├── clip_generator.py      # 영상 → 클립 분할
│   ├── sampler.py             # Keyframe 선택 (Phase1/2, PGL-SUM)
│   ├── scorer.py              # 이상행동 점수 변환
│   └── filter.py              # threshold 필터링
├── models/
│   ├── tsm/                   # ResNet + TSM 추론
│   ├── vlm/                   # InternVL2 추론
│   ├── feature_extractor.py   # ResNet-50 frame feature 추출
│   └── frame_selector.py      # BiGRU FrameScorer (Phase 2)
├── scripts/
│   ├── prepare_data.py        # 비디오 → clip → feature → pseudo-label
│   ├── generate_pseudo_labels.py
│   ├── train_frame_selector.py  # Phase 2 학습
│   ├── train_pglsum_anomaly.py
│   ├── evaluate_selector.py   # Phase1 / Phase2 / PGL-SUM 3-way 비교
│   ├── eval_clip_score.py
│   ├── eval_temporal.py
│   ├── eval_video_level.py
│   ├── eval_vlm.py
│   └── eval_vlm_video.py
└── run_all.sh                 # 데이터 준비 → 학습 → 평가 일괄 실행
```

## 실행

```bash
# 전체 파이프라인 (데이터 준비 → Phase 2 학습 → 3-way 평가)
bash run_all.sh

# 개별 실행
python scripts/prepare_data.py
python scripts/train_frame_selector.py
python scripts/evaluate_selector.py
```

## 데이터셋

- UCF-Crime, XD-Violence 사용
- 영상: `data/videos/anomaly/`, `data/videos/normal/` 아래 배치
- 대용량 데이터(videos/, Data/)는 git-ignored
