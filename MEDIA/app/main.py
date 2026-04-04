from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.health import router as health_router
from app.api.stream import router as stream_router
from app.core.config import settings
from app.utils.file_utils import ensure_dirs

app = FastAPI(title=settings.app_name)

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
app.include_router(stream_router)

app.mount("/hls", StaticFiles(directory=settings.hls_dir), name="hls")