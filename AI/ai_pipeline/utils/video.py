import math

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
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f" 영상 열기 실패: {video_path}")

    fps = normalize_video_fps(
        cap.get(cv2.CAP_PROP_FPS),
        default=fallback_fps,
    )
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    return frames, fps


def clip_frame_span(clip_id, clip_length, stride):
    start_frame = clip_id * stride
    end_frame = start_frame + clip_length - 1
    return start_frame, end_frame


def frames_to_seconds(frame_index, fps):
    if fps <= 0:
        return 0.0
    return frame_index / fps
