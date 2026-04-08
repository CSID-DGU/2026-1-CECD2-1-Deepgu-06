from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.common import ApiResponse
from app.services.hls_service import (
    get_stream_status,
    list_streams,
    start_hls_stream,
    stop_hls_stream,
)

router = APIRouter(prefix="/streams", tags=["streams"])


class StartStreamRequest(BaseModel):
    stream_key: str = Field(..., description="RTMP stream key")


@router.post("/{camera_id}/start", response_model=ApiResponse[dict])
def start_stream(camera_id: str, request: StartStreamRequest):
    try:
        input_url = (
            f"rtmp://{settings.rtmp_host}:"
            f"{settings.rtmp_port}/"
            f"{settings.rtmp_app}/"
            f"{request.stream_key}"
        )

        data = start_hls_stream(
            camera_id=camera_id,
            input_url=input_url,
        )

        return ApiResponse(success=True, data=data)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{camera_id}/stop", response_model=ApiResponse[dict])
def stop_stream(camera_id: str):
    try:
        data = stop_hls_stream(camera_id)
        return ApiResponse(success=True, data=data)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{camera_id}", response_model=ApiResponse[dict])
def stream_status(camera_id: str):
    try:
        data = get_stream_status(camera_id)
        return ApiResponse(success=True, data=data)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=ApiResponse[dict])
def stream_list():
    try:
        data = list_streams()
        return ApiResponse(success=True, data=data)

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))