from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.core.config import settings
from app.services.hls_service import (
    get_stream_status,
    list_streams,
    start_hls_stream,
    stop_hls_stream,
)

router = APIRouter(prefix="/streams", tags=["streams"])


class StartStreamRequest(BaseModel):
    stream_key: str = Field(..., description="RTMP stream key")


@router.post("/{camera_id}/start")
def start_stream(camera_id: str, request: StartStreamRequest):
    try:
        input_url = (
    f"rtmp://{settings.rtmp_host}:"
    f"{settings.rtmp_port}/"
    f"{settings.rtmp_app}/"
    f"{request.stream_key}"
)

        return start_hls_stream(
            camera_id=camera_id,
            input_url=input_url,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{camera_id}/stop")
def stop_stream(camera_id: str):
    try:
        return stop_hls_stream(camera_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{camera_id}/status")
def stream_status(camera_id: str):
    try:
        return get_stream_status(camera_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
def stream_list():
    return list_streams()