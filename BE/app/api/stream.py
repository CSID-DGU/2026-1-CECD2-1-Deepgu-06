from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.common import ApiResponse
from app.schemas.stream import (
    StreamControlResponse,
    StreamStartRequest,
    StreamSessionItem,
    StreamStatusResponse,
)
from app.services.stream_service import StreamService

router = APIRouter(prefix="/api/cameras", tags=["streams"])


@router.post(
    "/{camera_id}/stream/start",
    response_model=ApiResponse[StreamControlResponse],
    status_code=status.HTTP_200_OK,
)
async def start_stream(
    camera_id: str,
    _: StreamStartRequest,
    db: AsyncSession = Depends(get_db),
):
    service = StreamService(db)
    data = await service.start_stream(camera_id)
    return ApiResponse(success=True, data=data)


@router.post(
    "/{camera_id}/stream/stop",
    response_model=ApiResponse[StreamControlResponse],
    status_code=status.HTTP_200_OK,
)
async def stop_stream(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
):
    service = StreamService(db)
    data = await service.stop_stream(camera_id)
    return ApiResponse(success=True, data=data)


@router.get(
    "/{camera_id}/stream",
    response_model=ApiResponse[StreamStatusResponse],
    status_code=status.HTTP_200_OK,
)
async def get_stream_status(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
):
    service = StreamService(db)
    data = await service.get_stream_status(camera_id)
    return ApiResponse(success=True, data=data)


@router.get(
    "/{camera_id}/stream/sessions",
    response_model=ApiResponse[list[StreamSessionItem]],
    status_code=status.HTTP_200_OK,
)
async def get_stream_sessions(
    camera_id: str,
    page: int = Query(default=0, ge=0),
    size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    service = StreamService(db)
    data = await service.get_stream_sessions(camera_id, page, size)
    return ApiResponse(success=True, data=data)