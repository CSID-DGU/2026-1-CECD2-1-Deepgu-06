from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.database import get_db
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
):
    try:
        camera = CameraService.create_camera(db, payload)
        return success_response(data=serialize_camera_detail(camera))
    except ValueError as e:
        if str(e) == "CAMERA_ID_ALREADY_EXISTS":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="CAMERA_ID_ALREADY_EXISTS",
            )
        if str(e) == "STREAM_KEY_ALREADY_EXISTS":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="STREAM_KEY_ALREADY_EXISTS",
            )
        raise


@router.get("")
def get_cameras(db: Session = Depends(get_db)):
    cameras = CameraService.get_camera_list(db)
    return success_response(
        data=[serialize_camera_list_item(camera) for camera in cameras]
    )


@router.get("/{cameraId}")
def get_camera(cameraId: str, db: Session = Depends(get_db)):
    camera = CameraService.get_camera_by_camera_id(db, cameraId)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CAMERA_NOT_FOUND",
        )

    return success_response(data=serialize_camera_detail(camera))


@router.patch("/{cameraId}")
def update_camera(
    cameraId: str,
    payload: CameraUpdateRequest,
    db: Session = Depends(get_db),
):
    camera = CameraService.get_camera_by_camera_id(db, cameraId)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CAMERA_NOT_FOUND",
        )

    try:
        updated_camera = CameraService.update_camera(db, camera, payload)
        return success_response(data=serialize_camera_detail(updated_camera))
    except ValueError as e:
        if str(e) == "STREAM_KEY_ALREADY_EXISTS":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="STREAM_KEY_ALREADY_EXISTS",
            )
        raise


@router.delete("/{cameraId}")
def delete_camera(cameraId: str, db: Session = Depends(get_db)):
    camera = CameraService.get_camera_by_camera_id(db, cameraId)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CAMERA_NOT_FOUND",
        )

    try:
        CameraService.delete_camera(db, camera)
        return success_response(
            data={
                "cameraId": cameraId,
                "deleted": True,
            }
        )
    except ValueError as e:
        if str(e) == "CAMERA_RUNNING_CANNOT_DELETE":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="CAMERA_RUNNING_CANNOT_DELETE",
            )
        raise