import numpy as np


def sample_frames(clip_frames, num_samples=4):
    """
    clip에서 대표 frame 추출

    Args:
        clip_frames (list): frame 리스트
        num_samples (int): 추출할 frame 수

    Returns:
        sampled_frames (list)
    """

    total_frames = len(clip_frames)

    if total_frames < num_samples:
        # frame 부족하면 그대로 반환
        return clip_frames

    indices = np.linspace(0, total_frames - 1, num_samples).astype(int)

    sampled_frames = [clip_frames[i] for i in indices]

    return sampled_frames

def sample_from_candidates(candidates, num_samples=4):
    """
    여러 candidate clip에 대해 sampling 적용
    """

    for c in candidates:
        clip = c["clip"]
        sampled = sample_frames(clip, num_samples)

        c["sampled_frames"] = sampled

    return candidates