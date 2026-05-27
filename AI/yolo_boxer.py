"""
YOLO bounding box overlay for saved event clip videos.

Usage (as module):
    from yolo_boxer import annotate_clip, annotate_events

    annotate_events(result["event_payload"])   # in-place, overwrites clip files
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import cv2


# [수정] mp4v → H.264 변환 헬퍼: 브라우저 호환 H.264 mp4 생성
def _transcode_to_h264(src: str, dst: str) -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-preset", "fast",
             "-crf", "23", "-movflags", "+faststart", "-an", dst],
            capture_output=True, timeout=120,
        )
        return r.returncode == 0
    except Exception:
        return False

DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent / "yolov8n.pt"
)

_model_cache: Dict[str, object] = {}


def _load_model(model_path: str):
    if model_path not in _model_cache:
        from ultralytics import YOLO
        _model_cache[model_path] = YOLO(model_path)
    return _model_cache[model_path]


def _draw_boxes(frame, detections, conf_threshold: float = 0.3):
    for box in detections.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        label = f"{detections.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame, label, (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            cv2.LINE_AA,
        )
    return frame


def annotate_clip(
    clip_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    conf_threshold: float = 0.3,
    output_path: Optional[str] = None,
):
    """
    Load a saved clip, draw YOLO boxes on every frame, and save.
    Overwrites the original file unless output_path is given.
    """
    clip_path = str(clip_path)
    if not os.path.exists(clip_path):
        print(f"[yolo_boxer] clip not found: {clip_path}")
        return

    model = _load_model(model_path)

    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_path = clip_path + ".tmp.mp4"
    writer = cv2.VideoWriter(
        tmp_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model(frame, verbose=False)
        if results:
            frame = _draw_boxes(frame, results[0], conf_threshold)
        writer.write(frame)

    cap.release()
    writer.release()

    dest = output_path or clip_path
    # [수정] mp4v → H.264 변환 후 최종 파일로 교체 (브라우저 호환)
    h264_tmp = dest + ".h264.mp4"
    if _transcode_to_h264(tmp_path, h264_tmp):
        os.remove(tmp_path)
        os.replace(h264_tmp, dest)
    else:
        os.replace(tmp_path, dest)
    print(f"[yolo_boxer] annotated: {dest}")


def annotate_events(
    payload: Dict,
    model_path: str = DEFAULT_MODEL_PATH,
    conf_threshold: float = 0.3,
):
    """
    Annotate all event clips in a payload dict in-place.
    Each event's clip_video_path is overwritten with the boxed version.
    """
    for event in payload.get("events", []):
        clip_path = event.get("clip_video_path")
        if clip_path:
            annotate_clip(clip_path, model_path=model_path, conf_threshold=conf_threshold)
