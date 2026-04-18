from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import app.models  # noqa: F401 — 모든 모델을 SQLAlchemy에 등록

from app.api.auth import router as auth_router
from app.api.camera import router as camera_router
from app.api.internal import router as internal_router
from app.api.stream import router as stream_router
from app.api.user import router as user_router
from app.core.config import settings
from app.core.database import check_db_connection
from app.core.exceptions import add_exception_handlers

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "success": True,
        "message": "server is running",
        "data": {"app": "ok"},
    }


@app.get("/ready")
def readiness_check():
    db_ok = check_db_connection()

    if not db_ok:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "database connection failed",
                "data": {"app": "ok", "database": "fail"},
            },
        )

    return {
        "success": True,
        "message": "server is ready",
        "data": {"app": "ok", "database": "ok"},
    }


app.include_router(auth_router)
app.include_router(camera_router)
app.include_router(stream_router)
app.include_router(user_router)
app.include_router(internal_router)
add_exception_handlers(app)
