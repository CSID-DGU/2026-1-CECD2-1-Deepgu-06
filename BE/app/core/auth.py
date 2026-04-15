from datetime import datetime, timedelta, UTC

import base64
import hashlib

import bcrypt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.exceptions import AppException
from app.models.user import User

bearer_scheme = HTTPBearer()


def _prepare(password: str) -> bytes:
    """SHA-256 전처리로 bcrypt 72바이트 제한 우회"""
    return base64.b64encode(hashlib.sha256(password.encode()).digest())


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(_prepare(plain), hashed.encode())


def create_access_token(user_id: int, role: str) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": str(user_id), "role": role, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError:
        raise AppException(status_code=401, code="INVALID_TOKEN", message="유효하지 않은 토큰입니다.")


def _get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    payload = _decode_token(credentials.credentials)
    user_id = int(payload.get("sub", 0))

    user = db.get(User, user_id)
    if not user or user.status != "ACTIVE":
        raise AppException(status_code=401, code="UNAUTHORIZED", message="인증이 필요합니다.")
    return user


def get_current_user(user: User = Depends(_get_current_user)) -> User:
    return user


def require_admin(user: User = Depends(_get_current_user)) -> User:
    if user.role != "ADMIN":
        raise AppException(status_code=403, code="FORBIDDEN", message="관리자 권한이 필요합니다.")
    return user
