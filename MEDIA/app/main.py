import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.stream import router as stream_router
from app.core.config import settings
from app.services.hls_service import monitor_processes


@asynccontextmanager
async def lifespan(_app: FastAPI):
    monitor_thread = threading.Thread(target=monitor_processes, daemon=True)
    monitor_thread.start()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(stream_router)
