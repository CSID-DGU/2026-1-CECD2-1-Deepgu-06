from app.core.config import settings


def get_frame_status() -> dict:
    return {
        "frames_dir": settings.frames_dir,
        "message": "Frame extraction placeholder",
        "ready": True,
    }