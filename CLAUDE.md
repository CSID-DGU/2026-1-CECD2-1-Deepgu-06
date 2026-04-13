# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Deepgu** — a video anomaly detection system (2026-1 Capstone Design, Team Deep-gu). Detects abnormal behaviors in surveillance video streams using a FastAPI backend + AI inference pipeline.

## Commands

### Backend (BE/)

```bash
cd BE

# Install dependencies
pip install -r requirements.txt

# Run development server
python -m uvicorn app.main:app --reload

# Run production server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Docker build and run
docker build -t deepgu-be .
docker run -p 8000:8000 --env-file .env deepgu-be
```

### AI Pipeline (AI/)

```bash
cd AI/ai_pipeline

# Run full pipeline on a video
python pipeline/main_pipeline.py

# Test scripts
python scripts/test_clip_generator.py
python scripts/test_pipeline.py
```

## Architecture

### System Components

**BE/** — FastAPI backend (Python 3.11, port 8000)
- `app/main.py` — app entry point with router registration and lifespan
- `app/api/` — REST endpoints: `camera.py` (CRUD), `stream.py` (start/stop/status)
- `app/services/` — business logic: `camera_service.py`, `stream_service.py`
- `app/models/` — SQLAlchemy ORM: `cameras`, `stream_sessions` tables
- `app/schemas/` — Pydantic request/response DTOs
- `app/clients/media_server_client.py` — async HTTP client for external HLS media server
- `app/core/` — `config.py` (Settings via pydantic-settings), `database.py`, `enums.py`, `exceptions.py`

**AI/** — Inference pipeline (prototype stage)
- `pipeline/main_pipeline.py` — orchestrator
- `pipeline/clip_generator.py` — sliding window video segmentation (16 frames, stride 8)
- `models/tsm/inference.py` — TSM (Temporal Segment Networks) — **currently a dummy returning random probabilities**
- `models/vlm/inference.py` — InternVL2-8B vision-language model (requires CUDA)
- `pipeline/scorer.py` + `pipeline/filter.py` — anomaly scoring and candidate filtering (threshold 0.4)
- `pipeline/sampler.py` — extracts 4 representative frames per candidate clip

### Data Flow

```
Client → FastAPI (BE) → MySQL DB
                   ↓
          Media Server Client → External HLS Server (port 9000)
                                        ↓
                                  HLS Stream URL returned to client
```

```
Video File → Clip Generator → TSM Inference → Scorer/Filter → Frame Sampler → VLM Inference → JSON output
```

### Key Design Patterns

- **Service layer**: All business logic in `*_service.py`; API handlers only parse/validate and delegate
- **Dependency injection**: DB sessions injected via FastAPI `Depends(get_db)`
- **Pydantic schemas** separate from SQLAlchemy models; schemas in `app/schemas/`
- **Custom exception handling**: `AppException` in `core/exceptions.py` maps to HTTP status codes
- **State machine**: Camera and stream session statuses tracked via `CameraStatus` / `StreamSessionStatus` enums — transitions enforced in service layer
- **Compensation logic**: If DB save fails after media server starts, the service attempts to stop the media server to prevent state inconsistency

### External Dependencies

- **MySQL** — connection details in `BE/.env` (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`)
- **HLS Media Server** — configured via `MEDIA_SERVER_BASE_URL` (default `http://172.31.4.225:9000`)
- **CUDA GPU** — required for AI pipeline VLM inference (uses `cuda:1`)

## Configuration

All backend config is in `BE/.env` and loaded via `app/core/config.py` (pydantic-settings):

```
APP_NAME, APP_ENV, APP_HOST, APP_PORT
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
MEDIA_SERVER_BASE_URL
```

## API Surface

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/cameras` | Create camera |
| GET | `/api/cameras` | List cameras |
| GET | `/api/cameras/{cameraId}` | Get camera |
| PATCH | `/api/cameras/{cameraId}` | Update camera |
| DELETE | `/api/cameras/{cameraId}` | Delete camera |
| POST | `/api/cameras/{camera_id}/stream/start` | Start stream |
| POST | `/api/cameras/{camera_id}/stream/stop` | Stop stream |
| GET | `/api/cameras/{camera_id}/stream` | Get stream status |
| GET | `/api/cameras/{camera_id}/stream/sessions` | List sessions (paginated) |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe (checks DB) |
