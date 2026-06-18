from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.auth import require_admin
from app.core.database import get_db
from app.models.camera import Camera
from app.models.camera_assignment import CameraAssignment
from app.models.user import User
from app.schemas.user import AssignCameraRequest, UserResponse
from app.services.user_service import UserService
from app.utils.response import success_response

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("")
def get_users(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    service = UserService(db)
    users = service.get_list()
    return success_response(data=[UserResponse.model_validate(u) for u in users])


@router.patch("/{userId}/approve")
def approve_user(
    userId: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    service = UserService(db)
    user = service.approve(userId)
    return success_response(data=UserResponse.model_validate(user))


@router.delete("/{userId}")
def delete_user(
    userId: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    service = UserService(db)
    service.delete(userId)
    return success_response(data={"userId": userId, "deleted": True})


@router.post("/cameras/{cameraId}/assign")
def assign_camera(
    cameraId: str,
    payload: AssignCameraRequest,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    service = UserService(db)
    service.assign_camera(cameraId, payload.user_id)
    return success_response(data={"cameraId": cameraId, "userId": payload.user_id, "assigned": True})


@router.delete("/cameras/{cameraId}/assign/{userId}")
def unassign_camera(
    cameraId: str,
    userId: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    service = UserService(db)
    service.unassign_camera(cameraId, userId)
    return success_response(data={"cameraId": cameraId, "userId": userId, "unassigned": True})


@router.get("/{userId}/cameras")
def get_user_cameras(
    userId: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    cameras = (
        db.query(Camera)
        .join(CameraAssignment, CameraAssignment.camera_id == Camera.id)
        .filter(CameraAssignment.user_id == userId)
        .all()
    )
    return success_response(data=[{"cameraId": c.camera_id, "name": c.name} for c in cameras])
