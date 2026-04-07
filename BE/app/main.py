from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import check_db_connection

app = FastAPI(title=settings.APP_NAME)


@app.get("/health")
def health_check():
    return {
        "success": True,
        "message": "server is running",
        "data": {
            "app": "ok",
        },
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
                "data": {
                    "app": "ok",
                    "database": "fail",
                },
            },
        )

    return {
        "success": True,
        "message": "server is ready",
        "data": {
            "app": "ok",
            "database": "ok",
        },
    }