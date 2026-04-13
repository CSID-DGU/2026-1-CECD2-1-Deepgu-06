from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"


class Settings(BaseSettings):
    app_name: str = "Media Server"
    app_env: str = "local"
    app_host: str = "0.0.0.0"
    app_port: int = 9000

    incoming_dir: str = str(STORAGE_DIR / "incoming")
    hls_dir: str = str(STORAGE_DIR / "hls")
    frames_dir: str = str(STORAGE_DIR / "frames")
    thumbnails_dir: str = str(STORAGE_DIR / "thumbnails")
    clips_dir: str = str(STORAGE_DIR / "clips")

    ffmpeg_binary: str = "ffmpeg"
    hls_time: int = 2
    hls_list_size: int = 6
    segment_filename: str = "segment_%03d.ts"


    rtmp_host: str = "127.0.0.1"
    rtmp_port: int = 1935
    rtmp_app: str = "stream"

    instance_id: str = "media-instance-1"
    callback_secret: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()