from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Deepgu Backend"
    app_env: str = "local"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    db_host: str
    db_port: int = 3306
    db_name: str
    db_user: str
    db_password: str

    media_server_base_urlmedia: str = "http://media-server:9000"
    media_server_timeout_seconds: int = 10

    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset=utf8mb4"
        )


settings = Settings()