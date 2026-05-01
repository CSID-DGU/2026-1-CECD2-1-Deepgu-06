from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EventCreateRequest(BaseModel):
    camera_id: str
    detected_at: datetime
    anomaly_type: str
    confidence: float
    description: str | None = None


class EventStatusUpdateRequest(BaseModel):
    status: str


class EventItem(BaseModel):
    id: int
    camera_id: str
    detected_at: datetime
    anomaly_type: str
    confidence: float
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class EventDetail(EventItem):
    description: str | None
    video_url: str | None = None


class EventListResponse(BaseModel):
    items: list[EventItem]
    total: int
    page: int
    size: int
