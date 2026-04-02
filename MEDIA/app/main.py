from fastapi import FastAPI
from app.api.health import router as health_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.utils.file_utils import ensure_dirs


setup_logging()

app = FastAPI(title=settings.app_name)


@app.on_event("startup")
def on_startup() -> None:
    ensure_dirs(
        [
            settings.incoming_dir,
            settings.hls_dir,
            settings.frames_dir,
            settings.thumbnails_dir,
            settings.clips_dir,
        ]
    )


app.include_router(health_router)