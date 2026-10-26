from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Deepgu Backend"
    APP_ENV: str = "local"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    media_server_base_url: str = "http://media-server:9000"
    media_server_timeout_seconds: int = 10

    app_base_url: str = "http://localhost:8000"
    callback_secret: str

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    cors_origins: list[str] = ["https://2026-1-cecd-2-1-deepgu-06.vercel.app", "http://localhost:5173"]

    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "ap-northeast-2"
    s3_bucket_name: str = ""
    s3_presigned_expires: int = 3600

    @model_validator(mode="after")
    def validate_secrets(self):
        if not self.jwt_secret_key.strip():
            raise ValueError("JWT_SECRET_KEY must not be empty")
        if self.APP_ENV.lower() == "production" and not self.callback_secret.strip():
            raise ValueError("CALLBACK_SECRET must not be empty in production")
        return self

    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset=utf8mb4"
        )


settings = Settings()
