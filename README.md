# 2026-1-CECD2-1-Deepgu-06
2026-1 종합설계2 딥지우팀 — 영상 이상행동 감지 시스템

## 전체 모델 파이프라인

```
실시간 영상
  → clip_generator.py     : 영상을 고정 길이 클립(16 frame, stride 8)으로 분할
  → TSM.predict()         : 클립별 이상행동 확률 산출 (ResNet + Temporal Shift Module)
  → scorer.py / filter.py : 확률 → 점수 변환 및 threshold 필터링
  → sampler.py            : 클립에서 keyframe 선택 (아래 참고)
  → InternVL2.predict()   : 선택된 keyframe으로 VLM 최종 이상행동 판정
  → results.json          : 결과 저장
```

## Keyframe 추출

이상행동 판정 전 클립에서 대표 frame을 선택하는 단계.
세 가지 방식을 구현하고 성능을 비교한다.

| 방식 | 클래스 | 설명 | 학습 필요 |
|------|--------|------|-----------|
| Phase 1 | `KeyframeSampler(model_path=None)` | K-means 시간 구간 + Motion proxy | 불필요 |
| Phase 2 | `KeyframeSampler(model_path=...)` | BiGRU FrameScorer (pseudo-label 학습) | 필요 |
| PGL-SUM | `PGLSumSampler(model_path=...)` | PGL-SUM + Anomaly Proxy (UCF-Crime scratch 학습) | 필요 |

### Phase 1 — Temporal Clustering + Motion
K-means로 clip을 N구간으로 나눠 시간적 커버리지를 확보하고, 각 구간에서 velocity·acceleration을 결합한 motion score가 높은 frame을 선택한다.

### Phase 2 — BiGRU FrameScorer
Phase 1 + InternVL2로 생성한 pseudo-label로 BiGRU 기반 FrameScorer를 학습한다. VLM이 잘 이해하는 frame을 직접 예측하도록 학습된다.

### PGL-SUM (메인 선택기)
PGL-SUM (Global + Local Attention for Video Summarization) 구조에 Anomaly Proxy supervision을 결합해 UCF-Crime 데이터셋 2,150개 영상으로 scratch 학습한다.

```
학습 데이터 : UCF-Crime ResNet-50 2048-dim feature (h5)
Supervision : motion proxy gtscore (frame-level annotation 대용)
체크포인트  : PGL-SUM/Summaries/UCF/models/split0/best_model.pth
```

## 모듈 구성

| 디렉토리 | 역할 |
|----------|------|
| `AI/ai_pipeline/` | TSM + VLM 기반 이상행동 탐지 파이프라인 |
| `PGL-SUM/` | PGL-SUM 비디오 요약 모델 (keyframe selector 백본) |
| `MEDIA/` | 영상 스트리밍 서버 (FastAPI + FFmpeg, Docker 배포) |

## 실행

```bash
# AI 파이프라인
cd AI/ai_pipeline
bash run_all.sh          # 데이터 준비 → Phase 2 학습 → 3-way 평가

# PGL-SUM 학습
cd PGL-SUM
python model/main.py --split_index 0 --n_epochs 200 --batch_size 20 --video_type UCF

# MEDIA 서버
cd MEDIA
uvicorn app.main:app --host 0.0.0.0 --port 9000
```
