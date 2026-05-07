import json
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.core.config import settings
from app.core.database import get_db
from app.core.exceptions import AppException
from app.schemas.event import EventCreateRequest, EventItem
from app.schemas.stream import StreamCallbackRequest
from app.services.event_service import EventService
from app.services.stream_service import StreamService
from app.utils.response import success_response

router = APIRouter(prefix="/internal", tags=["internal"])


@router.post("/streams/callback")
def stream_callback(
    request: Request,
    payload: StreamCallbackRequest,
    db: Session = Depends(get_db),
):
    if settings.callback_secret:
        received = request.headers.get("X-Callback-Secret", "")
        if received != settings.callback_secret:
            raise AppException(
                status_code=401,
                code="UNAUTHORIZED",
                message="invalid callback secret",
            )

    service = StreamService(db)
    service.handle_callback(payload)
    return {"success": True}


def _check_callback_secret(request: Request) -> None:
    if settings.callback_secret:
        received = request.headers.get("X-Callback-Secret", "")
        if received != settings.callback_secret:
            raise AppException(status_code=401, code="UNAUTHORIZED", message="invalid callback secret")


@router.post("/events", status_code=201)
async def create_event(
    request: Request,
    camera_id: str = Form(...),
    detected_at: datetime = Form(...),
    anomaly_type: str = Form(...),
    confidence: float = Form(...),
    description: str | None = Form(None),
    video: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    _check_callback_secret(request)

    payload = EventCreateRequest(
        camera_id=camera_id,
        detected_at=detected_at,
        anomaly_type=anomaly_type,
        confidence=confidence,
        description=description,
    )

    video_bytes = await video.read() if video else None
    event = EventService.create(db, payload, video_bytes)
    return success_response(EventItem.model_validate(event).model_dump())


@router.post("/event-payloads", status_code=201)
async def create_event_payloads(
    request: Request,
    db: Session = Depends(get_db),
):
    _check_callback_secret(request)

    form = await request.form()
    raw_payload = form.get("payload")
    if not raw_payload:
        raise AppException(status_code=400, code="INVALID_PAYLOAD", message="payload field is required")

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        raise AppException(status_code=400, code="INVALID_PAYLOAD", message="payload must be valid JSON")

    events = payload.get("events", [])
    if not isinstance(events, list) or not events:
        return success_response({"created": [], "count": 0})

    created = []
    for index, event_data in enumerate(events):
        if not isinstance(event_data, dict):
            continue

        detected_at_raw = event_data.get("started_at") or payload.get("generated_at")
        if not detected_at_raw:
            raise AppException(
                status_code=400,
                code="INVALID_EVENT",
                message=f"event[{index}] is missing started_at/generated_at",
            )

        try:
            detected_at = datetime.fromisoformat(str(detected_at_raw))
        except ValueError:
            raise AppException(
                status_code=400,
                code="INVALID_EVENT",
                message=f"event[{index}] has invalid started_at",
            )

        camera_id = (
            event_data.get("stream_id")
            or payload.get("stream_id")
            or form.get("camera_id")
            or str(payload.get("cctv_id") or "")
        )
        if not camera_id:
            raise AppException(
                status_code=400,
                code="INVALID_EVENT",
                message=f"event[{index}] is missing camera identifier",
            )

        create_payload = EventCreateRequest(
            camera_id=str(camera_id),
            detected_at=detected_at,
            anomaly_type=str(event_data.get("label") or "fight"),
            confidence=float(event_data.get("confidence") or 0.0),
            description=event_data.get("description"),
        )

        video_field_name = f"event_{index}_video"
        upload = form.get(video_field_name)
        video_bytes = await upload.read() if isinstance(upload, (UploadFile, StarletteUploadFile)) else None
        thumbnail_bytes_list = []
        thumb_index = 1
        while True:
            thumb_field_name = f"event_{index}_thumb_{thumb_index}"
            thumb_upload = form.get(thumb_field_name)
            if not isinstance(thumb_upload, (UploadFile, StarletteUploadFile)):
                break
            thumbnail_bytes_list.append(await thumb_upload.read())
            thumb_index += 1

        ended_at_raw = event_data.get("ended_at")
        ended_at = None
        if ended_at_raw:
            try:
                ended_at = datetime.fromisoformat(str(ended_at_raw))
            except ValueError:
                ended_at = None

        created_event = EventService.create_from_event_payload(
            db,
            create_payload,
            ai_event_id=event_data.get("event_id"),
            ended_at=ended_at,
            duration_sec=float(event_data.get("duration_sec") or 0.0) or None,
            video_bytes=video_bytes,
            thumbnail_bytes_list=thumbnail_bytes_list,
        )
        created.append(
            {
                "db_event": EventItem.model_validate(created_event).model_dump(),
                "event_id": event_data.get("event_id"),
                "started_at": event_data.get("started_at"),
                "ended_at": event_data.get("ended_at"),
                "duration_sec": event_data.get("duration_sec"),
                "thumbnail_count": len(thumbnail_bytes_list),
            }
        )

    return success_response({"created": created, "count": len(created)})
