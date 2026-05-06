from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def initialize_database() -> None:
<<<<<<< HEAD
    try:
        Base.metadata.create_all(bind=engine)
        _ensure_event_columns()
    except Exception as e:
        print(f"[DB] 초기화 실패 (DB 연결 불가): {e}")
=======
    Base.metadata.create_all(bind=engine)
    _ensure_event_columns()
>>>>>>> 9cf5a2e ([Fix] 백엔드 s3저장 로직 변경)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _ensure_event_columns() -> None:
    statements = [
        "ALTER TABLE events ADD COLUMN ai_event_id VARCHAR(120) NULL",
        "ALTER TABLE events ADD COLUMN ended_at DATETIME NULL",
        "ALTER TABLE events ADD COLUMN duration_sec FLOAT NULL",
        "ALTER TABLE events ADD COLUMN thumbnail_s3_keys TEXT NULL",
    ]
    with engine.begin() as connection:
        try:
            existing = {
                row[0]
                for row in connection.execute(text("SHOW COLUMNS FROM events"))
            }
        except Exception:
            return

        mapping = {
            "ai_event_id": statements[0],
            "ended_at": statements[1],
            "duration_sec": statements[2],
            "thumbnail_s3_keys": statements[3],
        }
        for column_name, statement in mapping.items():
            if column_name in existing:
                continue
            connection.execute(text(statement))
