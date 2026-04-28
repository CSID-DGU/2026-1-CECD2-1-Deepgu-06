from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.auth import get_current_user, require_admin
from app.core.database import get_db
from app.models.camera_assignment import CameraAssignment
from app.models.user import User
from app.schemas.camera import CameraCreateRequest, CameraUpdateRequest
from app.services.camera_service import CameraService
from app.utils.response import success_response


router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


def serialize_camera_list_item(camera):
    return {
        "id": camera.id,
        "cameraId": camera.camera_id,
        "name": camera.name,
        "location": camera.location,
        "status": camera.status,
        "active": camera.active,
    }


def serialize_camera_detail(camera):
    return {
        "id": camera.id,
        "cameraId": camera.camera_id,
        "name": camera.name,
        "location": camera.location,
        "streamKey": camera.stream_key,
        "description": camera.description,
        "status": camera.status,
        "active": camera.active,
        "createdAt": camera.created_at,
        "updatedAt": camera.updated_at,
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def create_camera(
    payload: CameraCreateRequest,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    try:
        camera = CameraService.create_camera(db, payload)
        return success_response(data=serialize_camera_detail(camera))
    except ValueError as e:
        if str(e) == "CAMERA_ID_ALREADY_EXISTS":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="CAMERA_ID_ALREADY_EXISTS")
        if str(e) == "STREAM_KEY_ALREADY_EXISTS":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="STREAM_KEY_ALREADY_EXISTS")
        raise


@router.get("")
def get_cameras(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role == "ADMIN":
        cameras = CameraService.get_camera_list(db)
    else:
        assignments = db.query(CameraAssignment).filter(CameraAssignment.user_id == current_user.id).all()
        camera_ids = [a.camera_id for a in assignments]
        cameras = CameraService.get_camera_list_by_ids(db, camera_ids)

    return success_response(data=[serialize_camera_list_item(c) for c in cameras])


@router.get("/{cameraId}")
def get_camera(
    cameraId: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    camera = CameraService.get_camera_by_camera_id(db, cameraId)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CAMERA_NOT_FOUND")

    if current_user.role != "ADMIN":
        assignment = (
            db.query(CameraAssignment)
            .filter(CameraAssignment.user_id == current_user.id, CameraAssignment.camera_id == camera.id)
            .first()
        )
        if not assignment:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="ACCESS_DENIED")

    return success_response(data=serialize_camera_detail(camera))


@router.patch("/{cameraId}")
def update_camera(
    cameraId: str,
    payload: CameraUpdateRequest,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    camera = CameraService.get_camera_by_camera_id(db, cameraId)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CAMERA_NOT_FOUND")

    try:
        updated_camera = CameraService.update_camera(db, camera, payload)
        return success_response(data=serialize_camera_detail(updated_camera))
    except ValueError as e:
        if str(e) == "STREAM_KEY_ALREADY_EXISTS":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="STREAM_KEY_ALREADY_EXISTS")
        if str(e) == "INVALID_REQUEST":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="INVALID_REQUEST")
        if str(e) == "CANNOT_UPDATE_STREAM_KEY_WHILE_RUNNING":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="CANNOT_UPDATE_STREAM_KEY_WHILE_RUNNING")
        raise


@router.delete("/{cameraId}")
def delete_camera(
    cameraId: str,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    camera = CameraService.get_camera_by_camera_id(db, cameraId)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CAMERA_NOT_FOUND")

    try:
        CameraService.delete_camera(db, camera)
        return success_response(data={"cameraId": cameraId, "deleted": True})
    except ValueError as e:
        if str(e) == "CAMERA_RUNNING_CANNOT_DELETE":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="CAMERA_RUNNING_CANNOT_DELETE")
        raise
