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

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()