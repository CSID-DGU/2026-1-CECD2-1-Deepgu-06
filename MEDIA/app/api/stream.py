from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.hls_service import (
    get_stream_status,
    list_streams,
    start_hls_stream,
    stop_hls_stream,
)

router = APIRouter(prefix="/streams", tags=["streams"])


class StartStreamRequest(BaseModel):
    camera_id: str = Field(..., description="카메라 식별자")
    input_url: str = Field(..., description="RTSP 또는 RTMP 입력 주소")


@router.post("/start")
def start_stream(request: StartStreamRequest):
    try:
        return start_hls_stream(
            camera_id=request.camera_id,
            input_url=request.input_url,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{camera_id}/stop")
def stop_stream(camera_id: str):
    try:
        return stop_hls_stream(camera_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{camera_id}/status")
def stream_status(camera_id: str):
    try:
        return get_stream_status(camera_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
def stream_list():
    return list_streams()