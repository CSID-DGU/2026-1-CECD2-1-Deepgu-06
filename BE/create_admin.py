"""
최초 관리자 계정 생성 스크립트
사용법: python create_admin.py
"""
import sys
from datetime import datetime, UTC

sys.path.append(".")

from app.core.auth import hash_password
from app.core.database import SessionLocal
from app.models.user import User


def create_admin(email: str, password: str, name: str) -> None:
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            print(f"이미 존재하는 이메일입니다: {email}")
            return

        now = datetime.now(UTC).replace(tzinfo=None)
        admin = User(
            email=email,
            password_hash=hash_password(password),
            name=name,
            role="ADMIN",
            status="ACTIVE",
            created_at=now,
            updated_at=now,
        )
        db.add(admin)
        db.commit()
        print(f"관리자 계정 생성 완료: {email}")
    finally:
        db.close()


if __name__ == "__main__":
    email = input("이메일: ").strip()
    password = input("비밀번호: ").strip()
    name = input("이름: ").strip()
    create_admin(email, password, name)
