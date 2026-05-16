import time
from threading import Lock

import httpx

from app.core.config import settings


_registry: dict[str, dict] = {}  # camera_id -> {stream_key, callback_url, hls_url, last_status}
_lock = Lock()

MONITOR_INTERVAL = 3


def _mediamtx_path(stream_key: str) -> str:
    return f"{settings.mediamtx_rtmp_app}/{stream_key}"


def _check_active(stream_key: str) -> bool:
    path = _mediamtx_path(stream_key)
    try:
        resp = httpx.get(
            f"{settings.mediamtx_api_url}/v3/paths/get/{path}",
            timeout=2.0,
        )
        if resp.status_code == 200:
            return resp.json().get("ready", False)
    except Exception:
        pass
    return False


def send_callback(callback_url: str, camera_id: str, status: str, hls_url: str) -> None:
    if not callback_url:
        return
    try:
        headers = {}
        if settings.callback_secret:
            headers["X-Callback-Secret"] = settings.callback_secret
        httpx.post(
            callback_url,
            json={"camera_id": camera_id, "status": status, "hls_url": hls_url},
            headers=headers,
            timeout=5.0,
        )
    except Exception:
        pass


def start_hls_stream(camera_id: str, input_url: str, callback_url: str = "") -> dict:
    stream_key = input_url.rsplit("/", 1)[-1]
    hls_url = f"/{settings.mediamtx_rtmp_app}/{stream_key}/index.m3u8"

    with _lock:
        _registry[camera_id] = {
            "stream_key": stream_key,
            "callback_url": callback_url,
            "hls_url": hls_url,
        }

    is_active = _check_active(stream_key)
    status = "RUNNING" if is_active else "STARTING"

    return {
        "camera_id": camera_id,
        "message": "stream running" if is_active else "stream starting",
        "status": status,
        "running": is_active,
        "hls_url": hls_url,
        "instance_id": settings.instance_id,
    }


def stop_hls_stream(camera_id: str, cleanup_files: bool = False) -> dict:  # noqa: ARG001
    with _lock:
        entry = _registry.pop(camera_id, None)

    hls_url = entry["hls_url"] if entry else f"/{settings.mediamtx_rtmp_app}/{camera_id}/index.m3u8"
    callback_url = entry["callback_url"] if entry else ""

    send_callback(callback_url, camera_id, "STOPPED", hls_url)

    return {
        "camera_id": camera_id,
        "message": "stream stopped",
        "status": "STOPPED",
        "running": False,
        "files_cleaned": False,
        "hls_url": hls_url,
        "instance_id": settings.instance_id,
    }


def get_stream_status(camera_id: str) -> dict:
    with _lock:
        entry = _registry.get(camera_id)

    stream_key = entry["stream_key"] if entry else camera_id
    hls_url = entry["hls_url"] if entry else f"/{settings.mediamtx_rtmp_app}/{stream_key}/index.m3u8"

    is_active = _check_active(stream_key)

    if is_active:
        status = "RUNNING"
    elif entry:
        status = "STARTING"
    else:
        status = "STOPPED"

    return {
        "camera_id": camera_id,
        "status": status,
        "running": is_active,
        "pid": None,
        "playlist_exists": is_active,
        "playlist_ready": is_active,
        "playlist_path": None,
        "hls_url": hls_url,
        "instance_id": settings.instance_id,
    }


def list_streams() -> dict:
    with _lock:
        items = list(_registry.items())

    streams = []
    for camera_id, entry in items:
        is_active = _check_active(entry["stream_key"])
        streams.append({
            "camera_id": camera_id,
            "status": "RUNNING" if is_active else "STOPPED",
            "running": is_active,
            "hls_url": entry["hls_url"],
        })

    return {"instance_id": settings.instance_id, "streams": streams}


def monitor_processes() -> None:
    while True:
        time.sleep(MONITOR_INTERVAL)
        with _lock:
            items = list(_registry.items())

        for camera_id, entry in items:
            is_active = _check_active(entry["stream_key"])
            new_status = "RUNNING" if is_active else "STARTING"
            last_status = entry.get("last_status")

            if new_status == last_status:
                continue

            with _lock:
                if camera_id in _registry:
                    _registry[camera_id]["last_status"] = new_status

            if new_status == "RUNNING":
                send_callback(entry["callback_url"], camera_id, "RUNNING", entry["hls_url"])
