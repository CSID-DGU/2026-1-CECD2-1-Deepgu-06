import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.registry.process_registry import process_registry
from app.utils.file_utils import ensure_dir


STARTUP_TIMEOUT_SECONDS = 8
STARTUP_POLL_INTERVAL_SECONDS = 0.5
CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_camera_id(camera_id: str) -> None:
    if not camera_id or not CAMERA_ID_PATTERN.fullmatch(camera_id):
        raise ValueError("camera_id must contain only letters, numbers, underscores, and hyphens")


def validate_input_url(input_url: str) -> None:
    if not input_url or not isinstance(input_url, str):
        raise ValueError("input_url is required")

    allowed_prefixes = ("rtmp://", "rtsp://", "http://", "https://")
    if not input_url.startswith(allowed_prefixes):
        raise ValueError("input_url must start with rtmp://, rtsp://, http://, or https://")


def get_camera_output_dir(camera_id: str) -> str:
    validate_camera_id(camera_id)
    return os.path.join(settings.hls_dir, camera_id)


def get_playlist_path(camera_id: str) -> str:
    return os.path.join(get_camera_output_dir(camera_id), "index.m3u8")


def get_log_path(camera_id: str) -> str:
    return os.path.join(get_camera_output_dir(camera_id), "ffmpeg.log")


def safe_remove_dir(path: str) -> None:
    if not os.path.exists(path):
        return

    shutil.rmtree(path)


def safe_terminate_process(process: subprocess.Popen, timeout: int = 5) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def is_playlist_ready(playlist_path: str) -> bool:
    playlist = Path(playlist_path)
    return playlist.exists() and playlist.is_file() and playlist.stat().st_size > 0


def cleanup_stale_process(camera_id: str) -> Optional[subprocess.Popen]:
    process = process_registry.get(camera_id)
    if not process:
        return None

    if process.poll() is not None:
        process_registry.remove(camera_id)
        return None

    return process


def build_ffmpeg_command(input_url: str, segment_pattern: str, playlist_path: str) -> list[str]:
    return [
        settings.ffmpeg_binary,
    "-i", input_url,
    "-an",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-tune", "zerolatency",
    "-g", "40",
    "-keyint_min", "40",
    "-sc_threshold", "0",
    "-f", "hls",
    "-hls_time", str(settings.hls_time),
    "-hls_list_size", str(settings.hls_list_size),
    "-hls_flags", "delete_segments+append_list+independent_segments",
    "-hls_segment_filename", segment_pattern,
    playlist_path,
    ]


def start_hls_stream(camera_id: str, input_url: str) -> dict:
    validate_camera_id(camera_id)
    validate_input_url(input_url)

    existing_process = cleanup_stale_process(camera_id)
    if existing_process:
        return {
            "camera_id": camera_id,
            "message": "stream already running",
            "running": True,
            "pid": existing_process.pid,
            "playlist_path": get_playlist_path(camera_id),
            "hls_url": f"/hls/{camera_id}/index.m3u8",
            "instance_id": settings.instance_id,
        }

    output_dir = get_camera_output_dir(camera_id)
    playlist_path = get_playlist_path(camera_id)
    log_path = get_log_path(camera_id)
    segment_pattern = os.path.join(output_dir, settings.segment_filename)

    try:
        safe_remove_dir(output_dir)
        ensure_dir(output_dir)
    except Exception as e:
        raise RuntimeError(f"failed to prepare output directory: {e}") from e

    command = build_ffmpeg_command(input_url, segment_pattern, playlist_path)

    try:
        log_file = open(log_path, "ab")
    except Exception as e:
        raise RuntimeError(f"failed to open ffmpeg log file: {e}") from e

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=log_file,
        )
    except FileNotFoundError as e:
        log_file.close()
        raise RuntimeError(f"ffmpeg binary not found: {settings.ffmpeg_binary}") from e
    except Exception as e:
        log_file.close()
        raise RuntimeError(f"failed to start ffmpeg process: {e}") from e

    # 프로세스 객체에 로그 파일 핸들을 붙여서 stop 시 닫을 수 있게 함
    process._ffmpeg_log_file = log_file  # type: ignore[attr-defined]

    process_registry.add(camera_id, process)

    start_time = time.time()
    while time.time() - start_time < STARTUP_TIMEOUT_SECONDS:
        if process.poll() is not None:
            process_registry.remove(camera_id)
            try:
                log_file.close()
            except Exception:
                pass

            error_message = ""
            try:
                with open(log_path, "rb") as f:
                    error_message = f.read().decode(errors="ignore")[-2000:]
            except Exception:
                error_message = "unable to read ffmpeg log"

            raise RuntimeError(f"ffmpeg exited during startup: {error_message}")

        if is_playlist_ready(playlist_path):
            return {
                "camera_id": camera_id,
                "message": "stream started",
                "running": True,
                "pid": process.pid,
                "output_dir": output_dir,
                "playlist_path": playlist_path,
                "log_path": log_path,
                "hls_url": f"/hls/{camera_id}/index.m3u8",
                "instance_id": settings.instance_id,
            }

        time.sleep(STARTUP_POLL_INTERVAL_SECONDS)

    # timeout까지 기다렸는데 playlist가 아직 안 생기면 실패로 간주
    safe_terminate_process(process)
    process_registry.remove(camera_id)

    try:
        log_file.close()
    except Exception:
        pass

    error_message = ""
    try:
        with open(log_path, "rb") as f:
            error_message = f.read().decode(errors="ignore")[-2000:]
    except Exception:
        error_message = "unable to read ffmpeg log"

    raise RuntimeError(f"playlist was not created within timeout: {error_message}")


def stop_hls_stream(camera_id: str, cleanup_files: bool = False) -> dict:
    validate_camera_id(camera_id)

    process = process_registry.get(camera_id)
    output_dir = get_camera_output_dir(camera_id)

    if not process:
        result = {
            "camera_id": camera_id,
            "message": "stream not found",
            "running": False,
            "instance_id": settings.instance_id,
        }

        if cleanup_files and os.path.exists(output_dir):
            safe_remove_dir(output_dir)
            result["files_cleaned"] = True
        else:
            result["files_cleaned"] = False

        return result

    safe_terminate_process(process)
    process_registry.remove(camera_id)

    log_file = getattr(process, "_ffmpeg_log_file", None)
    if log_file:
        try:
            log_file.close()
        except Exception:
            pass

    files_cleaned = False
    if cleanup_files and os.path.exists(output_dir):
        safe_remove_dir(output_dir)
        files_cleaned = True

    return {
        "camera_id": camera_id,
        "message": "stream stopped",
        "running": False,
        "files_cleaned": files_cleaned,
        "instance_id": settings.instance_id,
    }


def get_stream_status(camera_id: str) -> dict:
    validate_camera_id(camera_id)

    process = cleanup_stale_process(camera_id)
    playlist_path = get_playlist_path(camera_id)

    if not process:
        return {
            "camera_id": camera_id,
            "running": False,
            "pid": None,
            "playlist_exists": Path(playlist_path).exists(),
            "playlist_ready": is_playlist_ready(playlist_path),
            "playlist_path": playlist_path,
            "hls_url": f"/hls/{camera_id}/index.m3u8",
            "instance_id": settings.instance_id,
        }

    return {
        "camera_id": camera_id,
        "running": process.poll() is None,
        "pid": process.pid,
        "playlist_exists": Path(playlist_path).exists(),
        "playlist_ready": is_playlist_ready(playlist_path),
        "playlist_path": playlist_path,
        "hls_url": f"/hls/{camera_id}/index.m3u8",
        "instance_id": settings.instance_id,
    }


def list_streams() -> dict:
    streams = []
    registry_snapshot = process_registry.list_all()

    for item in registry_snapshot:
        camera_id = item.get("camera_id")
        if not camera_id:
            continue

        try:
            process = cleanup_stale_process(camera_id)
            playlist_path = get_playlist_path(camera_id)

            streams.append({
                "camera_id": camera_id,
                "running": process is not None and process.poll() is None,
                "pid": process.pid if process else None,
                "playlist_exists": Path(playlist_path).exists(),
                "playlist_ready": is_playlist_ready(playlist_path),
                "playlist_path": playlist_path,
                "hls_url": f"/hls/{camera_id}/index.m3u8",
            })
        except Exception:
            streams.append({
                "camera_id": camera_id,
                "running": False,
                "pid": None,
                "playlist_exists": False,
                "playlist_ready": False,
                "playlist_path": None,
                "hls_url": f"/hls/{camera_id}/index.m3u8",
            })

    return {
        "instance_id": settings.instance_id,
        "streams": streams,
    }
# import os
# import re
# import shutil
# import subprocess
# import time
# from pathlib import Path
# from typing import Optional

# from app.core.config import settings
# from app.registry.process_registry import process_registry
# from app.utils.file_utils import ensure_dir


# STARTUP_TIMEOUT_SECONDS = 8
# STARTUP_POLL_INTERVAL_SECONDS = 0.5
# CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


# def validate_camera_id(camera_id: str) -> None:
#     if not camera_id or not CAMERA_ID_PATTERN.fullmatch(camera_id):
#         raise ValueError("camera_id must contain only letters, numbers, underscores, and hyphens")


# def validate_input_url(input_url: str) -> None:
#     if not input_url or not isinstance(input_url, str):
#         raise ValueError("input_url is required")

#     allowed_prefixes = ("rtmp://", "rtsp://", "http://", "https://")
#     if not input_url.startswith(allowed_prefixes):
#         raise ValueError("input_url must start with rtmp://, rtsp://, http://, or https://")


# def get_camera_output_dir(camera_id: str) -> str:
#     validate_camera_id(camera_id)
#     return os.path.join(settings.hls_dir, camera_id)


# def get_playlist_path(camera_id: str) -> str:
#     return os.path.join(get_camera_output_dir(camera_id), "index.m3u8")


# def get_log_path(camera_id: str) -> str:
#     return os.path.join(get_camera_output_dir(camera_id), "ffmpeg.log")


# def safe_remove_dir(path: str) -> None:
#     if not os.path.exists(path):
#         return

#     shutil.rmtree(path)


# def safe_terminate_process(process: subprocess.Popen, timeout: int = 5) -> None:
#     if process.poll() is not None:
#         return

#     process.terminate()
#     try:
#         process.wait(timeout=timeout)
#     except subprocess.TimeoutExpired:
#         process.kill()
#         process.wait(timeout=timeout)


# def is_playlist_ready(playlist_path: str) -> bool:
#     playlist = Path(playlist_path)
#     return playlist.exists() and playlist.is_file() and playlist.stat().st_size > 0


# def cleanup_stale_process(camera_id: str) -> Optional[subprocess.Popen]:
#     process = process_registry.get(camera_id)
#     if not process:
#         return None

#     if process.poll() is not None:
#         process_registry.remove(camera_id)
#         return None

#     return process


# def build_ffmpeg_command(input_url: str, segment_pattern: str, playlist_path: str) -> list[str]:
#     return [
#         settings.ffmpeg_binary,
#     "-i", input_url,
#     "-an",
#     "-c:v", "libx264",
#     "-preset", "veryfast",
#     "-tune", "zerolatency",
#     "-g", "40",
#     "-keyint_min", "40",
#     "-sc_threshold", "0",
#     "-f", "hls",
#     "-hls_time", str(settings.hls_time),
#     "-hls_list_size", str(settings.hls_list_size),
#     "-hls_flags", "delete_segments+append_list+independent_segments",
#     "-hls_segment_filename", segment_pattern,
#     playlist_path,
#     ]


# def start_hls_stream(camera_id: str, input_url: str) -> dict:
#     validate_camera_id(camera_id)
#     validate_input_url(input_url)

#     existing_process = cleanup_stale_process(camera_id)
#     if existing_process:
#         return {
#             "camera_id": camera_id,
#             "message": "stream already running",
#             "running": True,
#             "pid": existing_process.pid,
#             "playlist_path": get_playlist_path(camera_id),
#             "hls_url": f"/hls/{camera_id}/index.m3u8",
#             "instance_id": settings.instance_id,
#         }

#     output_dir = get_camera_output_dir(camera_id)
#     playlist_path = get_playlist_path(camera_id)
#     log_path = get_log_path(camera_id)
#     segment_pattern = os.path.join(output_dir, settings.segment_filename)

#     try:
#         safe_remove_dir(output_dir)
#         ensure_dir(output_dir)
#     except Exception as e:
#         raise RuntimeError(f"failed to prepare output directory: {e}") from e

#     command = build_ffmpeg_command(input_url, segment_pattern, playlist_path)

#     try:
#         log_file = open(log_path, "ab")
#     except Exception as e:
#         raise RuntimeError(f"failed to open ffmpeg log file: {e}") from e

#     try:
#         process = subprocess.Popen(
#             command,
#             stdout=subprocess.DEVNULL,
#             stderr=log_file,
#         )
#     except FileNotFoundError as e:
#         log_file.close()
#         raise RuntimeError(f"ffmpeg binary not found: {settings.ffmpeg_binary}") from e
#     except Exception as e:
#         log_file.close()
#         raise RuntimeError(f"failed to start ffmpeg process: {e}") from e

#     # 프로세스 객체에 로그 파일 핸들을 붙여서 stop 시 닫을 수 있게 함
#     process._ffmpeg_log_file = log_file  # type: ignore[attr-defined]

#     process_registry.add(camera_id, process)

#     start_time = time.time()
#     while time.time() - start_time < STARTUP_TIMEOUT_SECONDS:
#         if process.poll() is not None:
#             process_registry.remove(camera_id)
#             try:
#                 log_file.close()
#             except Exception:
#                 pass

#             error_message = ""
#             try:
#                 with open(log_path, "rb") as f:
#                     error_message = f.read().decode(errors="ignore")[-2000:]
#             except Exception:
#                 error_message = "unable to read ffmpeg log"

#             raise RuntimeError(f"ffmpeg exited during startup: {error_message}")

#         if is_playlist_ready(playlist_path):
#             return {
#                 "camera_id": camera_id,
#                 "message": "stream started",
#                 "running": True,
#                 "pid": process.pid,
#                 "output_dir": output_dir,
#                 "playlist_path": playlist_path,
#                 "log_path": log_path,
#                 "hls_url": f"/hls/{camera_id}/index.m3u8",
#                 "instance_id": settings.instance_id,
#             }

#         time.sleep(STARTUP_POLL_INTERVAL_SECONDS)

#     # timeout까지 기다렸는데 playlist가 아직 안 생기면 실패로 간주
#     safe_terminate_process(process)
#     process_registry.remove(camera_id)

#     try:
#         log_file.close()
#     except Exception:
#         pass

#     error_message = ""
#     try:
#         with open(log_path, "rb") as f:
#             error_message = f.read().decode(errors="ignore")[-2000:]
#     except Exception:
#         error_message = "unable to read ffmpeg log"

#     raise RuntimeError(f"playlist was not created within timeout: {error_message}")


# def stop_hls_stream(camera_id: str, cleanup_files: bool = False) -> dict:
#     validate_camera_id(camera_id)

#     process = process_registry.get(camera_id)
#     output_dir = get_camera_output_dir(camera_id)

#     if not process:
#         result = {
#             "camera_id": camera_id,
#             "message": "stream not found",
#             "running": False,
#             "instance_id": settings.instance_id,
#         }

#         if cleanup_files and os.path.exists(output_dir):
#             safe_remove_dir(output_dir)
#             result["files_cleaned"] = True
#         else:
#             result["files_cleaned"] = False

#         return result

#     safe_terminate_process(process)
#     process_registry.remove(camera_id)

#     log_file = getattr(process, "_ffmpeg_log_file", None)
#     if log_file:
#         try:
#             log_file.close()
#         except Exception:
#             pass

#     files_cleaned = False
#     if cleanup_files and os.path.exists(output_dir):
#         safe_remove_dir(output_dir)
#         files_cleaned = True

#     return {
#         "camera_id": camera_id,
#         "message": "stream stopped",
#         "running": False,
#         "files_cleaned": files_cleaned,
#         "instance_id": settings.instance_id,
#     }


# def get_stream_status(camera_id: str) -> dict:
#     validate_camera_id(camera_id)

#     process = cleanup_stale_process(camera_id)
#     playlist_path = get_playlist_path(camera_id)

#     if not process:
#         return {
#             "camera_id": camera_id,
#             "running": False,
#             "pid": None,
#             "playlist_exists": Path(playlist_path).exists(),
#             "playlist_ready": is_playlist_ready(playlist_path),
#             "playlist_path": playlist_path,
#             "hls_url": f"/hls/{camera_id}/index.m3u8",
#             "instance_id": settings.instance_id,
#         }

#     return {
#         "camera_id": camera_id,
#         "running": process.poll() is None,
#         "pid": process.pid,
#         "playlist_exists": Path(playlist_path).exists(),
#         "playlist_ready": is_playlist_ready(playlist_path),
#         "playlist_path": playlist_path,
#         "hls_url": f"/hls/{camera_id}/index.m3u8",
#         "instance_id": settings.instance_id,
#     }


# def list_streams() -> dict:
#     streams = []
#     registry_snapshot = process_registry.list_all()

#     for item in registry_snapshot:
#         camera_id = item.get("camera_id")
#         if not camera_id:
#             continue

#         try:
#             process = cleanup_stale_process(camera_id)
#             playlist_path = get_playlist_path(camera_id)

#             streams.append({
#                 "camera_id": camera_id,
#                 "running": process is not None and process.poll() is None,
#                 "pid": process.pid if process else None,
#                 "playlist_exists": Path(playlist_path).exists(),
#                 "playlist_ready": is_playlist_ready(playlist_path),
#                 "playlist_path": playlist_path,
#                 "hls_url": f"/hls/{camera_id}/index.m3u8",
#             })
#         except Exception:
#             streams.append({
#                 "camera_id": camera_id,
#                 "running": False,
#                 "pid": None,
#                 "playlist_exists": False,
#                 "playlist_ready": False,
#                 "playlist_path": None,
#                 "hls_url": f"/hls/{camera_id}/index.m3u8",
#             })

#     return {
#         "instance_id": settings.instance_id,
#         "streams": streams,
#     }