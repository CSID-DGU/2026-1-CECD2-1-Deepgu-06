from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.auth import create_access_token, verify_password
from app.core.database import get_db
from app.core.exceptions import AppException
from app.models.user import User
from app.schemas.user import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from app.services.user_service import UserService
from app.utils.response import success_response

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    service = UserService(db)
    user = service.register(payload)
    return success_response(data=UserResponse.model_validate(user))


@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()

    if not user or not verify_password(payload.password, user.password_hash):
        raise AppException(status_code=401, code="INVALID_CREDENTIALS", message="이메일 또는 비밀번호가 올바르지 않습니다.")

    if user.status == "PENDING":
        raise AppException(status_code=403, code="PENDING_APPROVAL", message="관리자 승인 대기 중입니다.")

    if user.status == "INACTIVE":
        raise AppException(status_code=403, code="ACCOUNT_INACTIVE", message="비활성화된 계정입니다.")

    token = create_access_token(user.id, user.role)
    return success_response(data=TokenResponse(access_token=token))
