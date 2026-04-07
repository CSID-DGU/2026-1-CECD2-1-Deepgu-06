from datetime import datetime, UTC

from sqlalchemy import desc, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.media_server_client import MediaServerClient
from app.core.enums import CameraStatus, StreamSessionStatus
from app.core.exceptions import AppException
from app.models.camera import Camera
from app.models.stream_session import StreamSession
from app.schemas.stream import (
    MediaStartRequest,
    MediaStopRequest,
    StreamControlResponse,
    StreamSessionItem,
    StreamStatusResponse,
)


class StreamService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.media_client = MediaServerClient()

    async def _get_camera_or_404(self, camera_id: str) -> Camera:
        result = await self.db.execute(
            select(Camera).where(Camera.camera_id == camera_id)
        )
        camera = result.scalar_one_or_none()

        if camera is None:
            raise AppException(
                status_code=404,
                code="CAMERA_NOT_FOUND",
                message="존재하지 않는 카메라입니다.",
            )
        return camera

    async def _get_latest_session(self, camera_pk: int) -> StreamSession | None:
        result = await self.db.execute(
            select(StreamSession)
            .where(StreamSession.camera_id == camera_pk)
            .order_by(desc(StreamSession.id))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def start_stream(self, camera_id: str) -> StreamControlResponse:
        camera = await self._get_camera_or_404(camera_id)

        if not camera.active:
            raise AppException(
                status_code=409,
                code="CAMERA_INACTIVE",
                message="비활성 카메라는 스트리밍할 수 없습니다.",
            )

        if camera.status == CameraStatus.RUNNING.value:
            latest_session = await self._get_latest_session(camera.id)
            return StreamControlResponse(
                camera_id=camera.camera_id,
                status=CameraStatus.RUNNING.value,
                hls_url=latest_session.hls_url if latest_session else None,
                started_at=latest_session.started_at if latest_session else None,
                session_id=latest_session.id if latest_session else None,
                message="Stream is already running",
            )

        media_data = await self.media_client.start_stream(
            MediaStartRequest(
                camera_id=camera.camera_id,
                stream_key=camera.stream_key,
            )
        )

        now = datetime.now(UTC).replace(tzinfo=None)

        session = StreamSession(
            camera_id=camera.id,
            status=StreamSessionStatus.RUNNING.value,
            hls_url=media_data.get("hls_url"),
            started_at=now,
            stopped_at=None,
            created_at=now,
            updated_at=now,
        )

        try:
            camera.status = CameraStatus.RUNNING.value
            camera.updated_at = now

            self.db.add(session)
            await self.db.commit()
            await self.db.refresh(session)
        except SQLAlchemyError:
            await self.db.rollback()

            try:
                await self.media_client.stop_stream(
                    MediaStopRequest(camera_id=camera.camera_id)
                )
            except AppException:
                raise AppException(
                    status_code=500,
                    code="STREAM_STATE_INCONSISTENT",
                    message="media start succeeded but DB save and compensation stop both failed",
                )

            raise AppException(
                status_code=500,
                code="DATABASE_ERROR",
                message="스트리밍 시작 후 DB 저장에 실패했습니다.",
            )

        return StreamControlResponse(
            camera_id=camera.camera_id,
            status=CameraStatus.RUNNING.value,
            hls_url=session.hls_url,
            started_at=session.started_at,
            session_id=session.id,
        )

    async def stop_stream(self, camera_id: str) -> StreamControlResponse:
        camera = await self._get_camera_or_404(camera_id)
        latest_session = await self._get_latest_session(camera.id)

        if camera.status != CameraStatus.RUNNING.value:
            return StreamControlResponse(
                camera_id=camera.camera_id,
                status=CameraStatus.STOPPED.value,
                stopped_at=datetime.now(UTC).replace(tzinfo=None),
                session_id=latest_session.id if latest_session else None,
                hls_url=latest_session.hls_url if latest_session else None,
                message="Stream is already stopped",
            )

        await self.media_client.stop_stream(
            MediaStopRequest(camera_id=camera.camera_id)
        )

        now = datetime.now(UTC).replace(tzinfo=None)

        try:
            camera.status = CameraStatus.STOPPED.value
            camera.updated_at = now

            if latest_session and latest_session.status == StreamSessionStatus.RUNNING.value:
                latest_session.status = StreamSessionStatus.STOPPED.value
                latest_session.stopped_at = now
                latest_session.updated_at = now

            await self.db.commit()
        except SQLAlchemyError:
            await self.db.rollback()
            raise AppException(
                status_code=500,
                code="DATABASE_ERROR",
                message="스트리밍 중지 후 DB 갱신에 실패했습니다.",
            )

        return StreamControlResponse(
            camera_id=camera.camera_id,
            status=CameraStatus.STOPPED.value,
            stopped_at=now,
            session_id=latest_session.id if latest_session else None,
            hls_url=latest_session.hls_url if latest_session else None,
        )

    async def get_stream_status(self, camera_id: str) -> StreamStatusResponse:
        camera = await self._get_camera_or_404(camera_id)
        latest_session = await self._get_latest_session(camera.id)

        current_session = None
        if latest_session:
            current_session = StreamSessionItem(
                session_id=latest_session.id,
                status=latest_session.status,
                hls_url=latest_session.hls_url,
                started_at=latest_session.started_at,
                stopped_at=latest_session.stopped_at,
            )

        return StreamStatusResponse(
            camera_id=camera.camera_id,
            camera_status=camera.status,
            current_session=current_session,
        )

    async def get_stream_sessions(self, camera_id: str, page: int, size: int) -> list[StreamSessionItem]:
        camera = await self._get_camera_or_404(camera_id)

        offset = page * size
        result = await self.db.execute(
            select(StreamSession)
            .where(StreamSession.camera_id == camera.id)
            .order_by(desc(StreamSession.id))
            .offset(offset)
            .limit(size)
        )

        sessions = result.scalars().all()

        return [
            StreamSessionItem(
                session_id=session.id,
                status=session.status,
                hls_url=session.hls_url,
                started_at=session.started_at,
                stopped_at=session.stopped_at,
            )
            for session in sessions
        ]