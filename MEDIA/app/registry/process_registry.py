from subprocess import Popen
from threading import Lock
from typing import Optional


class ProcessEntry:
    def __init__(self, process: Popen, callback_url: str, hls_url: str):
        self.process = process
        self.callback_url = callback_url
        self.hls_url = hls_url
        self.last_status: Optional[str] = None


class ProcessRegistry:
    def __init__(self):
        self._entries: dict[str, ProcessEntry] = {}
        self._lock = Lock()

    def add(self, camera_id: str, process: Popen, callback_url: str = "", hls_url: str = "") -> None:
        with self._lock:
            self._entries[camera_id] = ProcessEntry(process, callback_url, hls_url)

    def get(self, camera_id: str) -> Optional[Popen]:
        with self._lock:
            entry = self._entries.get(camera_id)
            return entry.process if entry else None

    def get_entry(self, camera_id: str) -> Optional[ProcessEntry]:
        with self._lock:
            return self._entries.get(camera_id)

    def remove(self, camera_id: str) -> Optional[Popen]:
        with self._lock:
            entry = self._entries.pop(camera_id, None)
            return entry.process if entry else None

    def set_last_status(self, camera_id: str, status: str) -> None:
        with self._lock:
            entry = self._entries.get(camera_id)
            if entry:
                entry.last_status = status

    def snapshot(self) -> list[tuple[str, ProcessEntry]]:
        with self._lock:
            return list(self._entries.items())

    def list_all(self) -> list[dict]:
        with self._lock:
            result = []
            for camera_id, entry in self._entries.items():
                result.append(
                    {
                        "camera_id": camera_id,
                        "pid": entry.process.pid,
                        "running": entry.process.poll() is None,
                    }
                )
            return result


process_registry = ProcessRegistry()
