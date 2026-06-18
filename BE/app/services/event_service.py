import json
from datetime import datetime

from sqlalchemy.orm import Session

from app.clients import s3_client
from app.core.enums import EventStatus
from app.models.event import Event
from app.schemas.event import EventCreateRequest


class EventService:
    ALLOWED_STATUSES = {s.value for s in EventStatus}

    @staticmethod
    def create(db: Session, payload: EventCreateRequest, video_bytes: bytes | None) -> Event:
        s3_key = None
        if video_bytes:
            s3_key = s3_client.upload_event_video(video_bytes, payload.camera_id, payload.detected_at)

        event = Event(
            camera_id=payload.camera_id,
            detected_at=payload.detected_at,
            anomaly_type=payload.anomaly_type,
            confidence=payload.confidence,
            description=payload.description,
            s3_key=s3_key,
            status=EventStatus.UNREVIEWED.value,
            created_at=datetime.utcnow(),
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event

    @staticmethod
    def create_from_event_payload(
        db: Session,
        payload: EventCreateRequest,
        *,
        ai_event_id: str | None = None,
        ended_at: datetime | None = None,
        duration_sec: float | None = None,
        video_bytes: bytes | None = None,
        thumbnail_bytes_list: list[bytes] | None = None,
    ) -> Event:
        s3_key = None
        if video_bytes:
            s3_key = s3_client.upload_event_video(video_bytes, payload.camera_id, payload.detected_at)

        thumbnail_keys = []
        for image_bytes in thumbnail_bytes_list or []:
            if not image_bytes:
                continue
            thumb_key = s3_client.upload_event_thumbnail(
                image_bytes,
                payload.camera_id,
                payload.detected_at,
            )
            thumbnail_keys.append(thumb_key)

        event = Event(
            camera_id=payload.camera_id,
            ai_event_id=ai_event_id,
            detected_at=payload.detected_at,
            ended_at=ended_at,
            duration_sec=duration_sec,
            anomaly_type=payload.anomaly_type,
            confidence=payload.confidence,
            description=payload.description,
            s3_key=s3_key,
            thumbnail_s3_keys=json.dumps(thumbnail_keys, ensure_ascii=False) if thumbnail_keys else None,
            status=EventStatus.UNREVIEWED.value,
            created_at=datetime.utcnow(),
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event

    @staticmethod
    def list_events(
        db: Session,
        page: int = 1,
        size: int = 20,
        status: str | None = None,
        camera_id: str | None = None,
        allowed_camera_ids: list[str] | None = None,
    ) -> tuple[list[Event], int]:
        query = db.query(Event)
        if allowed_camera_ids is not None:
            query = query.filter(Event.camera_id.in_(allowed_camera_ids))
        if status:
            query = query.filter(Event.status == status)
        if camera_id:
            query = query.filter(Event.camera_id == camera_id)
        total = query.count()
        items = query.order_by(Event.detected_at.desc()).offset((page - 1) * size).limit(size).all()
        return items, total

    @staticmethod
    def get(db: Session, event_id: int) -> Event | None:
        return db.get(Event, event_id)

    @staticmethod
    def update_status(db: Session, event: Event, status: str) -> Event:
        if status not in EventService.ALLOWED_STATUSES:
            raise ValueError("INVALID_STATUS")
        event.status = status
        db.commit()
        db.refresh(event)
        return event
