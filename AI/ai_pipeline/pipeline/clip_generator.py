import os
import cv2
from tqdm import tqdm

from utils.video import load_video_frames, normalize_video_fps


def generate_clips(
    video_path,
    output_dir,
    clip_len=16,
    stride=8,
    save=True,
    save_fps=None,
    return_metadata=False,
    verbose=True,
):
    """
    video → clip 리스트 생성

    Args:
        video_path (str): 입력 영상 경로
        output_dir (str): clip 저장 경로
        clip_len (int): 한 clip 당 frame 수
        stride (int): 슬라이딩 간격
        save (bool): clip 영상 파일로 저장 여부

    Returns:
        clips (list): 각 clip의 frame 리스트
    """

    if save:
        os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(" 영상 프레임 로딩 중...")

    frames, source_fps = load_video_frames(video_path)
    save_fps = normalize_video_fps(save_fps, default=source_fps)

    if verbose:
        print(f"총 프레임 수: {len(frames)}")

    clips = []

    if verbose:
        print(" clip 생성 중...")

    clip_starts = range(0, len(frames) - clip_len + 1, stride)
    iterator = tqdm(clip_starts) if verbose else clip_starts

    for start in iterator:
        clip_frames = frames[start:start + clip_len]
        clips.append(clip_frames)

        if save:
            clip_name = f"clip_{start}_{start+clip_len}.mp4"
            clip_path = os.path.join(output_dir, clip_name)

            save_clip(clip_frames, clip_path, fps=save_fps)

    if verbose:
        print(f" 총 clip 개수: {len(clips)}")

    if return_metadata:
        return clips, {
            "fps": float(source_fps),
            "num_frames": len(frames),
            "save_fps": float(save_fps),
        }

    return clips


def save_clip(frames, output_path, fps=30):
    """
    frame 리스트를 영상(mp4)으로 저장
    """

    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
