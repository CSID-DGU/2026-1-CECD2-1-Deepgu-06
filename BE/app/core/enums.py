from enum import Enum


class CameraStatus(str, Enum):
    INACTIVE = "INACTIVE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


class StreamSessionStatus(str, Enum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"