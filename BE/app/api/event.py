from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.clients.s3_client import generate_presigned_url
from app.core.auth import get_current_user, require_admin
from app.core.database import get_db
from app.core.exceptions import AppException
from app.schemas.event import EventDetail, EventItem, EventListResponse, EventStatusUpdateRequest
from app.services.event_service import EventService
from app.utils.response import success_response

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("", response_model=dict)
def list_events(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    camera_id: str | None = Query(None),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    items, total = EventService.list_events(db, page=page, size=size, status=status, camera_id=camera_id)
    return success_response(
        EventListResponse(
            items=[EventItem.model_validate(e) for e in items],
            total=total,
            page=page,
            size=size,
        ).model_dump()
    )


@router.get("/{event_id}", response_model=dict)
def get_event(
    event_id: int,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    event = EventService.get(db, event_id)
    if not event:
        raise AppException(status_code=404, code="EVENT_NOT_FOUND", message="이벤트를 찾을 수 없습니다.")

    detail = EventDetail.model_validate(event)
    detail.video_url = generate_presigned_url(event.s3_key) if event.s3_key else None
    return success_response(detail.model_dump())


@router.patch("/{event_id}/status", response_model=dict)
def update_event_status(
    event_id: int,
    payload: EventStatusUpdateRequest,
    db: Session = Depends(get_db),
    _=Depends(require_admin),
):
    event = EventService.get(db, event_id)
    if not event:
        raise AppException(status_code=404, code="EVENT_NOT_FOUND", message="이벤트를 찾을 수 없습니다.")
    try:
        updated = EventService.update_status(db, event, payload.status)
    except ValueError:
        raise AppException(status_code=400, code="INVALID_STATUS", message="유효하지 않은 상태값입니다.")
    return success_response(EventItem.model_validate(updated).model_dump())
