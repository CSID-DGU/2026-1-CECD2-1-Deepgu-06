from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.core.database import get_db
from app.models.camera_assignment import CameraAssignment
from app.models.user import User
from app.schemas.common import ApiResponse
from app.schemas.stream import (
    StreamControlResponse,
    StreamSessionItem,
    StreamStartRequest,
    StreamStatusResponse,
)
from app.services.camera_service import CameraService
from app.services.stream_service import StreamService

router = APIRouter(prefix="/api/cameras", tags=["streams"])


def _check_camera_access(camera_id: str, current_user: User, db: Session) -> None:
    """관리자가 아닌 경우 할당된 카메라인지 확인한다."""
    if current_user.role == "ADMIN":
        return
    camera = CameraService.get_camera_by_camera_id(db, camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CAMERA_NOT_FOUND")
    assignment = (
        db.query(CameraAssignment)
        .filter(CameraAssignment.user_id == current_user.id, CameraAssignment.camera_id == camera.id)
        .first()
    )
    if not assignment:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="ACCESS_DENIED")


@router.post(
    "/{camera_id}/stream/start",
    response_model=ApiResponse[StreamControlResponse],
    status_code=status.HTTP_200_OK,
)
async def start_stream(
    camera_id: str,
    request: StreamStartRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _check_camera_access(camera_id, current_user, db)
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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _check_camera_access(camera_id, current_user, db)
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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _check_camera_access(camera_id, current_user, db)
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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _check_camera_access(camera_id, current_user, db)
    service = StreamService(db)
    data = await service.get_stream_sessions(camera_id, page, size)
    return ApiResponse(success=True, data=data)
