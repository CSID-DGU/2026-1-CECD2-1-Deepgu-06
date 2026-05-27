import os
import requests

BE_BASE_URL = "http://43.201.17.169:8000"
CALLBACK_SECRET = "deepgu"


def post_event(camera_id, detected_at, anomaly_type, confidence,
               description="", video_path=None):
    headers = {}
    if CALLBACK_SECRET:
        headers["X-Callback-Secret"] = CALLBACK_SECRET

    data = {
        "camera_id": camera_id,
        "detected_at": detected_at.isoformat() if hasattr(detected_at, "isoformat") else str(detected_at),
        "anomaly_type": anomaly_type,
        "confidence": str(confidence),
        "description": description,
    }

    files = {}
    f = None
    if video_path and os.path.exists(video_path):
        f = open(video_path, "rb")
        files["video"] = ("clip.mp4", f, "video/mp4")

    try:
        resp = requests.post(
            f"{BE_BASE_URL}/internal/events",
            data=data,
            files=files if files else None,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    finally:
        if f:
            f.close()
