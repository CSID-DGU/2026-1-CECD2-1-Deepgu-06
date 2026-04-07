from enum import Enum


class CameraStatus(str, Enum):
    INACTIVE = "INACTIVE"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


class StreamSessionStatus(str, Enum):
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"