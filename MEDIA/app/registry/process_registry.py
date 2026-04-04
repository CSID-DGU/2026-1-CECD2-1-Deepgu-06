from subprocess import Popen
from threading import Lock
from typing import Optional


class ProcessRegistry:
    def __init__(self):
        self._processes: dict[str, Popen] = {}
        self._lock = Lock()

    def add(self, camera_id: str, process: Popen) -> None:
        with self._lock:
            self._processes[camera_id] = process

    def get(self, camera_id: str) -> Optional[Popen]:
        with self._lock:
            return self._processes.get(camera_id)

    def remove(self, camera_id: str) -> Optional[Popen]:
        with self._lock:
            return self._processes.pop(camera_id, None)

    def list_all(self) -> list[dict]:
        with self._lock:
            result = []
            for camera_id, process in self._processes.items():
                result.append(
                    {
                        "camera_id": camera_id,
                        "pid": process.pid,
                        "running": process.poll() is None,
                    }
                )
            return result


process_registry = ProcessRegistry()