from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.camera import Camera
from app.schemas.camera import CameraCreateRequest, CameraUpdateRequest


class CameraService:
    @staticmethod
    def get_camera_by_camera_id(db: Session, camera_id: str) -> Camera | None:
        return db.query(Camera).filter(Camera.camera_id == camera_id).first()

    @staticmethod
    def get_camera_by_stream_key(db: Session, stream_key: str) -> Camera | None:
        return db.query(Camera).filter(Camera.stream_key == stream_key).first()

    @staticmethod
    def create_camera(db: Session, payload: CameraCreateRequest) -> Camera:
        existing_camera_id = CameraService.get_camera_by_camera_id(db, payload.cameraId)
        if existing_camera_id:
            raise ValueError("CAMERA_ID_ALREADY_EXISTS")

        existing_stream_key = CameraService.get_camera_by_stream_key(db, payload.streamKey)
        if existing_stream_key:
            raise ValueError("STREAM_KEY_ALREADY_EXISTS")

        now = datetime.utcnow()

        camera = Camera(
            camera_id=payload.cameraId,
            name=payload.name,
            location=payload.location,
            stream_key=payload.streamKey,
            description=payload.description,
            status="INACTIVE",
            active=True,
            created_at=now,
            updated_at=now,
        )

        try:
            db.add(camera)
            db.commit()
            db.refresh(camera)
            return camera
        except IntegrityError:
            db.rollback()
            raise ValueError("DATABASE_INTEGRITY_ERROR")

    @staticmethod
    def get_camera_list(db: Session) -> list[Camera]:
        return db.query(Camera).order_by(Camera.id.desc()).all()

    @staticmethod
    def update_camera(db: Session, camera: Camera, payload: CameraUpdateRequest) -> Camera:
        update_fields = payload.model_dump(exclude_unset=True)

        if not update_fields:
            raise ValueError("INVALID_REQUEST")

        if "name" in update_fields and update_fields["name"] is None:
            raise ValueError("INVALID_REQUEST")

        if "streamKey" in update_fields and update_fields["streamKey"] is None:
            raise ValueError("INVALID_REQUEST")

        if "streamKey" in update_fields:
            duplicated_stream_key = (
                db.query(Camera)
                .filter(
                    Camera.stream_key == update_fields["streamKey"],
                    Camera.id != camera.id,
                )
                .first()
            )
            if duplicated_stream_key:
                raise ValueError("STREAM_KEY_ALREADY_EXISTS")

        if "name" in update_fields:
            camera.name = update_fields["name"]

        if "location" in update_fields:
            camera.location = update_fields["location"]

        if "streamKey" in update_fields:
            if camera.status == "RUNNING":
                raise ValueError("CANNOT_UPDATE_STREAM_KEY_WHILE_RUNNING")
            camera.stream_key = update_fields["streamKey"]

        if "description" in update_fields:
            camera.description = update_fields["description"]

        camera.updated_at = datetime.utcnow()

        try:
            db.commit()
            db.refresh(camera)
            return camera
        except IntegrityError:
            db.rollback()
            raise ValueError("DATABASE_INTEGRITY_ERROR")

    @staticmethod
    def delete_camera(db: Session, camera: Camera) -> None:
        if camera.status == "RUNNING":
            raise ValueError("CAMERA_RUNNING_CANNOT_DELETE")

        try:
            db.delete(camera)
            db.commit()
        except IntegrityError:
            db.rollback()
            raise ValueError("DATABASE_INTEGRITY_ERROR")