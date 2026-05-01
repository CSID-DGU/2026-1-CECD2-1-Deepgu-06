from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from sqlalchemy.orm import Session

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
