import httpx

from app.core.config import settings
from app.core.exceptions import AppException
from app.schemas.stream import (
    MediaStartRequest,
    MediaStartResponse,
    MediaStatusResponse,
    MediaStopRequest,
    MediaStopResponse,
)


class MediaServerClient:
    def __init__(self) -> None:
        self.base_url = settings.media_server_base_url.rstrip("/")
        self.timeout = settings.media_server_timeout_seconds

    async def start_stream(self, payload: MediaStartRequest) -> MediaStartResponse:
        url = f"{self.base_url}/streams/{payload.camera_id}/start"
        body = {
            "stream_key": payload.stream_key,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=body)
                response.raise_for_status()
                parsed = MediaStartResponse.model_validate(response.json())
        except httpx.RequestError as e:
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_START_FAILED",
                message=f"media server start request failed: {str(e)}",
            )
        except httpx.HTTPStatusError as e:
            detail = e.response.text
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_START_FAILED",
                message=f"media server returned {e.response.status_code}: {detail}",
            )
        except Exception as e:
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_START_FAILED",
                message=f"invalid media server start response: {str(e)}",
            )

        if not parsed.success:
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_START_FAILED",
                message="media server returned success=false on start",
            )

        return parsed

    async def stop_stream(self, payload: MediaStopRequest) -> MediaStopResponse:
        url = f"{self.base_url}/streams/{payload.camera_id}/stop"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url)
                response.raise_for_status()
                parsed = MediaStopResponse.model_validate(response.json())
        except httpx.RequestError as e:
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_STOP_FAILED",
                message=f"media server stop request failed: {str(e)}",
            )
        except httpx.HTTPStatusError as e:
            detail = e.response.text
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_STOP_FAILED",
                message=f"media server returned {e.response.status_code}: {detail}",
            )
        except Exception as e:
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_STOP_FAILED",
                message=f"invalid media server stop response: {str(e)}",
            )

        if not parsed.success:
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_STOP_FAILED",
                message="media server returned success=false on stop",
            )

        return parsed

    async def get_stream_status(self, camera_id: str) -> MediaStatusResponse:
        url = f"{self.base_url}/streams/{camera_id}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                parsed = MediaStatusResponse.model_validate(response.json())
        except httpx.RequestError as e:
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_STATUS_FAILED",
                message=f"media server status request failed: {str(e)}",
            )
        except httpx.HTTPStatusError as e:
            detail = e.response.text
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_STATUS_FAILED",
                message=f"media server returned {e.response.status_code}: {detail}",
            )
        except Exception as e:
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_STATUS_FAILED",
                message=f"invalid media server status response: {str(e)}",
            )

        if not parsed.success:
            raise AppException(
                status_code=502,
                code="MEDIA_SERVER_STATUS_FAILED",
                message="media server returned success=false on status",
            )

        return parsed