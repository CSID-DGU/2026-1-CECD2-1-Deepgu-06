import math
from pathlib import Path

import cv2


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
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)
    writer.release()


def clip_frame_span(clip_id, clip_length, stride):
    start_frame = clip_id * stride
    end_frame = start_frame + clip_length - 1
    return start_frame, end_frame


def frames_to_seconds(frame_index, fps):
    if fps <= 0:
        return 0.0
    return frame_index / fps
