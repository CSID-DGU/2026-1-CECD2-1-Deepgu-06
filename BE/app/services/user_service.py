from datetime import datetime, UTC

from sqlalchemy.orm import Session

from app.core.auth import hash_password
from app.core.exceptions import AppException
from app.models.camera import Camera
from app.models.camera_assignment import CameraAssignment
from app.models.user import User
from app.schemas.user import RegisterRequest


class UserService:
    def __init__(self, db: Session):
        self.db = db

    def register(self, payload: RegisterRequest) -> User:
        if self.db.query(User).filter(User.email == payload.email).first():
            raise AppException(status_code=409, code="EMAIL_ALREADY_EXISTS", message="이미 사용 중인 이메일입니다.")

        now = datetime.now(UTC).replace(tzinfo=None)
        user = User(
            email=payload.email,
            password_hash=hash_password(payload.password),
            name=payload.name,
            role="USER",
            status="PENDING",
            created_at=now,
            updated_at=now,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def approve(self, user_id: int) -> User:
        user = self._get_user_or_404(user_id)
        if user.status == "ACTIVE":
            raise AppException(status_code=409, code="ALREADY_APPROVED", message="이미 승인된 사용자입니다.")

        user.status = "ACTIVE"
        user.updated_at = datetime.now(UTC).replace(tzinfo=None)
        self.db.commit()
        self.db.refresh(user)
        return user

    def delete(self, user_id: int) -> None:
        user = self._get_user_or_404(user_id)
        self.db.delete(user)
        self.db.commit()

    def get_list(self) -> list[User]:
        return self.db.query(User).order_by(User.id.desc()).all()

    def assign_camera(self, camera_id_str: str, user_id: int) -> CameraAssignment:
        camera = self.db.query(Camera).filter(Camera.camera_id == camera_id_str).first()
        if not camera:
            raise AppException(status_code=404, code="CAMERA_NOT_FOUND", message="존재하지 않는 카메라입니다.")

        user = self._get_user_or_404(user_id)

        existing = (
            self.db.query(CameraAssignment)
            .filter(CameraAssignment.user_id == user.id, CameraAssignment.camera_id == camera.id)
            .first()
        )
        if existing:
            raise AppException(status_code=409, code="ALREADY_ASSIGNED", message="이미 할당된 카메라입니다.")

        now = datetime.now(UTC).replace(tzinfo=None)
        assignment = CameraAssignment(user_id=user.id, camera_id=camera.id, created_at=now)
        self.db.add(assignment)
        self.db.commit()
        return assignment

    def unassign_camera(self, camera_id_str: str, user_id: int) -> None:
        camera = self.db.query(Camera).filter(Camera.camera_id == camera_id_str).first()
        if not camera:
            raise AppException(status_code=404, code="CAMERA_NOT_FOUND", message="존재하지 않는 카메라입니다.")

        user = self._get_user_or_404(user_id)

        assignment = (
            self.db.query(CameraAssignment)
            .filter(CameraAssignment.user_id == user.id, CameraAssignment.camera_id == camera.id)
            .first()
        )
        if not assignment:
            raise AppException(status_code=404, code="ASSIGNMENT_NOT_FOUND", message="할당 내역이 없습니다.")

        self.db.delete(assignment)
        self.db.commit()

    def _get_user_or_404(self, user_id: int) -> User:
        user = self.db.get(User, user_id)
        if not user:
            raise AppException(status_code=404, code="USER_NOT_FOUND", message="존재하지 않는 사용자입니다.")
        return user
