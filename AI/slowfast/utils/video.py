import math
import os
import subprocess
from pathlib import Path

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


def normalize_video_fps(fps, default=30.0):
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        return float(default)

    if not math.isfinite(fps) or fps <= 0:
        return float(default)

    return fps


def load_video_frames(video_path, fallback_fps=30.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"failed to open video: {video_path}")

    fps = normalize_video_fps(cap.get(cv2.CAP_PROP_FPS), default=fallback_fps)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def save_video(frames, output_path, fps):
    if not frames:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]

    # [수정] mp4v로 먼저 쓰고 ffmpeg으로 H.264 변환 (브라우저 호환)
    tmp_path = str(output_path) + ".tmp_mp4v.mp4"
    writer = cv2.VideoWriter(
        tmp_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)
    writer.release()

    if _transcode_to_h264(tmp_path, str(output_path)):
        os.remove(tmp_path)
    else:
        import shutil
        shutil.move(tmp_path, str(output_path))


def clip_frame_span(clip_id, clip_length, stride):
    start_frame = clip_id * stride
    end_frame = start_frame + clip_length - 1
    return start_frame, end_frame


def frames_to_seconds(frame_index, fps):
    if fps <= 0:
        return 0.0
    return frame_index / fps
