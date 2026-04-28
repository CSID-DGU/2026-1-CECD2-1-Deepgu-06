
# Deepgu — VLM 기반 이상행동 분석 플랫폼
2026-1 종합설계2 딥지팀 — 영상 이상행동 감지 시스템

=======


> **"CCTV 영상에서 이상행동을 실시간으로 감지합니다."**  
> HLS 스트림 영상을 슬라이딩 윈도우로 분절하고, TSM과 InternVL2-8B(VLM)로  
> 이상행동을 탐지·분류하여 관제 시스템에 제공하는 영상 분석 플랫폼입니다.


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

### Phase 2 — BiGRU FrameScorer (메인 선택기)
Phase 1 + InternVL2로 생성한 pseudo-label로 BiGRU 기반 FrameScorer를 학습한다. VLM이 잘 이해하는 frame을 직접 예측하도록 학습된다.

### PGL-SUM 
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


<p>
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Python%203.11-009688?style=flat&logo=fastapi&logoColor=white">
  <img alt="VLM" src="https://img.shields.io/badge/VLM-InternVL2--8B-FF69B4?style=flat">
  <img alt="MySQL" src="https://img.shields.io/badge/MySQL-8.0-4479A1?style=flat&logo=mysql&logoColor=white">
  <img alt="AWS" src="https://img.shields.io/badge/AWS-EC2-FF9900?style=flat&logo=amazonaws&logoColor=white">
  <img alt="Vercel" src="https://img.shields.io/badge/Frontend-Vercel-000000?style=flat&logo=vercel&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-배포-2496ED?style=flat&logo=docker&logoColor=white">
</p>

---

## 👀 주요 기능

| 기능 | 설명 |
|------|------|
| **카메라 관리** | CCTV 카메라 등록·수정·삭제 및 상태 관리 |
| **스트림 제어** | HLS 스트리밍 시작·중지 및 세션 이력 조회 |
| **이상행동 탐지** | TSM 기반 1차 필터링 → VLM(InternVL2-8B) 정밀 분석 |
| **결과 제공** | 이상행동 탐지 결과를 JSON으로 반환 |

---

## 🏗️ 시스템 아키텍처

<img width="1430" height="1320" alt="image" src="https://github.com/user-attachments/assets/c6ce8e41-7e17-4721-b3fe-c0a1f659bf51" />

### 기술 스택

**Frontend**
- Vercel 배포

**Backend**
- FastAPI (Python 3.11, port 8000)
- MySQL (cameras · stream_sessions 테이블)
- Docker

**AI Pipeline**
- TSM (Temporal Segment Networks) — 1차 이상행동 스코어링
- InternVL2-8B — VLM 기반 정밀 분류 (CUDA GPU 필요)
- 슬라이딩 윈도우: 16프레임, stride 8
- 이상행동 임계값: 0.4
- 대표 프레임 추출: 클립당 4프레임

**Infra**
- AWS EC2
- Docker
- HLS Media Server (포트 9000)

---

## 📁 프로젝트 구조

```
├── BE/
│   ├── app/
│   │   ├── main.py              # 앱 진입점, 라우터 등록
│   │   ├── api/
│   │   │   ├── camera.py        # 카메라 CRUD
│   │   │   └── stream.py        # 스트림 시작/중지/상태
│   │   ├── services/
│   │   │   ├── camera_service.py
│   │   │   └── stream_service.py
│   │   ├── models/              # SQLAlchemy ORM
│   │   ├── schemas/             # Pydantic DTO
│   │   ├── clients/
│   │   │   └── media_server_client.py  # HLS 서버 연동
│   │   └── core/
│   │       ├── config.py        # 환경변수 (pydantic-settings)
│   │       ├── database.py
│   │       ├── enums.py         # 상태 머신 enum
│   │       └── exceptions.py
│   ├── requirements.txt
│   └── Dockerfile
│
└── AI/
    └── ai_pipeline/
        ├── pipeline/
        │   ├── main_pipeline.py   # 파이프라인 오케스트레이터
        │   ├── clip_generator.py  # 슬라이딩 윈도우 분절
        │   ├── scorer.py          # 이상행동 스코어링
        │   ├── filter.py          # 임계값 필터링
        │   └── sampler.py         # 대표 프레임 추출
        ├── models/
        │   ├── tsm/inference.py   # TSM 추론
        │   └── vlm/inference.py   # InternVL2-8B 추론
        └── scripts/               # 테스트 스크립트
```

---

## 🚀 실행 방법

### 사전 요구사항

- Python 3.11+
- Docker
- CUDA GPU (AI 파이프라인 VLM 추론용)
- MySQL 서버
- HLS Media Server (포트 9000)

### 백엔드

```bash
cd BE

# 환경변수 설정
cp .env.example .env
# .env 파일에 DB, Media Server 주소 설정

# 로컬 실행
pip install -r requirements.txt
python -m uvicorn app.main:app --reload

# Docker 실행
docker build -t deepgu-be .
docker run -p 8000:8000 --env-file .env deepgu-be
```

### AI 파이프라인

```bash
cd AI/ai_pipeline

pip install -r requirements.txt

# 전체 파이프라인 실행
python pipeline/main_pipeline.py

# 개별 테스트
python scripts/test_clip_generator.py
python scripts/test_pipeline.py
```

---

## ⚙️ 환경변수

`BE/.env`에 아래 값을 설정하세요.

```env
APP_NAME=deepgu
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=8000

DB_HOST=<DB_HOST>
DB_PORT=3306
DB_NAME=<DB_NAME>
DB_USER=<DB_USER>
DB_PASSWORD=<DB_PASSWORD>

MEDIA_SERVER_BASE_URL=http://172.31.4.225:9000
```

---

## 📡 API

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/cameras` | 카메라 등록 |
| GET | `/api/cameras` | 카메라 목록 조회 |
| GET | `/api/cameras/{cameraId}` | 카메라 단건 조회 |
| PATCH | `/api/cameras/{cameraId}` | 카메라 수정 |
| DELETE | `/api/cameras/{cameraId}` | 카메라 삭제 |
| POST | `/api/cameras/{camera_id}/stream/start` | 스트림 시작 |
| POST | `/api/cameras/{camera_id}/stream/stop` | 스트림 중지 |
| GET | `/api/cameras/{camera_id}/stream` | 스트림 상태 조회 |
| GET | `/api/cameras/{camera_id}/stream/sessions` | 세션 이력 (페이지네이션) |
| GET | `/health` | 생존 확인 |
| GET | `/ready` | 준비 확인 (DB 연결 체크) |

---

## 🔄 설계 패턴

- **서비스 레이어**: 모든 비즈니스 로직은 `*_service.py`에 위치, API 핸들러는 파싱·위임만 담당
- **의존성 주입**: DB 세션은 FastAPI `Depends(get_db)`로 주입
- **상태 머신**: `CameraStatus` / `StreamSessionStatus` enum으로 상태 전이 제어
- **보상 트랜잭션**: DB 저장 실패 시 미디어 서버 중지로 상태 불일치 방지

---

## 🤝 기여 방법

1. 이 레포를 Fork 합니다.
2. 새 브랜치를 생성합니다. (`git checkout -b feature/기능명`)
3. 변경사항을 커밋합니다. (`git commit -m "feat: 기능 설명"`)
4. 브랜치에 Push합니다. (`git push origin feature/기능명`)
5. Pull Request를 생성합니다.

---

## 📄 라이센스

This project is licensed under the MIT License.

