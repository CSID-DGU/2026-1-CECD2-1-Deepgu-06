from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class CameraAssignment(Base):
    __tablename__ = "camera_assignments"
    __table_args__ = (
        UniqueConstraint("user_id", "camera_id", name="uq_user_camera"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    camera_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    user = relationship("User", back_populates="assignments")
    camera = relationship("Camera", back_populates="assignments")
