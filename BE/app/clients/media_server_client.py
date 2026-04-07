import httpx

from app.core.config import settings
from app.core.exceptions import AppException
from app.schemas.stream import MediaStartRequest, MediaStopRequest


class MediaServerClient:
    def __init__(self) -> None:
        self.base_url = settings.media_server_base_url.rstrip("/")
        self.timeout = settings.media_server_timeout_seconds

    async def start_stream(self, payload: MediaStartRequest) -> dict:
        url = f"{self.base_url}/streams/{payload.camera_id}/start"
        body = {
            "stream_key": payload.stream_key
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=body)
        except httpx.RequestError as e:
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_START_FAILED",
                message=f"media server start request failed: {str(e)}",
            )

        if response.status_code >= 400:
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_START_FAILED",
                message=f"media server returned {response.status_code}: {response.text}",
            )

        body = response.json()
        if not body.get("success"):
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_START_FAILED",
                message="media server start failed",
            )

        return body["data"]

    async def stop_stream(self, payload: MediaStopRequest) -> dict:
        url = f"{self.base_url}/streams/{payload.camera_id}/stop"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url)
        except httpx.RequestError as e:
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_STOP_FAILED",
                message=f"media server stop request failed: {str(e)}",
            )

        if response.status_code >= 400:
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_STOP_FAILED",
                message=f"media server returned {response.status_code}: {response.text}",
            )

        body = response.json()
        if not body.get("success"):
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_STOP_FAILED",
                message="media server stop failed",
            )

        return body["data"]