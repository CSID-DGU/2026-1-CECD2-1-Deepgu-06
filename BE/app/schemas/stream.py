from datetime import datetime

from pydantic import BaseModel, ConfigDict


class StreamStartRequest(BaseModel):
    force_restart: bool = False


class StreamControlResponse(BaseModel):
    camera_id: str
    status: str
    hls_url: str | None = None
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    session_id: int | None = None
    message: str | None = None


class StreamSessionItem(BaseModel):
    session_id: int
    status: str
    hls_url: str | None
    started_at: datetime
    stopped_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class StreamStatusResponse(BaseModel):
    camera_id: str
    camera_status: str
    current_session: StreamSessionItem | None


class MediaStartRequest(BaseModel):
    camera_id: str
    stream_key: str


class MediaStopRequest(BaseModel):
    camera_id: str


class MediaStartData(BaseModel):
    status: str
    hls_url: str


class MediaStopData(BaseModel):
    status: str


class MediaResponse(BaseModel):
    success: bool
    data: dict