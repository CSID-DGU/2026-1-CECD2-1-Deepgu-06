from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.exceptions import AppException
from app.schemas.stream import StreamCallbackRequest
from app.services.stream_service import StreamService

router = APIRouter(prefix="/internal", tags=["internal"])


@router.post("/streams/callback")
def stream_callback(
    request: Request,
    payload: StreamCallbackRequest,
    db: Session = Depends(get_db),
):
    if settings.callback_secret:
        received = request.headers.get("X-Callback-Secret", "")
        if received != settings.callback_secret:
            raise AppException(
                status_code=401,
                code="UNAUTHORIZED",
                message="invalid callback secret",
            )

    service = StreamService(db)
    service.handle_callback(payload)
    return {"success": True}
