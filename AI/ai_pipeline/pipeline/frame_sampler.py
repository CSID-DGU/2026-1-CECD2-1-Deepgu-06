import numpy as np


def sample_uniform_plus_center(clip_frames, num_samples=6):
    total_frames = len(clip_frames)

    if total_frames <= num_samples:
        return clip_frames

    base_count = max(2, num_samples - 2)
    uniform_indices = np.linspace(0, total_frames - 1, base_count).astype(int).tolist()

    center = total_frames // 2
    extra_indices = [max(0, center - 1), min(total_frames - 1, center + 1)]
    indices = sorted(set(uniform_indices + extra_indices))

    if len(indices) > num_samples:
        indices = indices[:num_samples]

    return [clip_frames[i] for i in indices]


def sample_from_candidates(candidates, num_samples=6, strategy="uniform_plus_center"):
    for candidate in candidates:
        clip = candidate["clip"]

        if strategy == "uniform_plus_center":
            sampled = sample_uniform_plus_center(clip, num_samples=num_samples)
        else:
            sampled = sample_uniform_plus_center(clip, num_samples=num_samples)

        candidate["sampled_frames"] = sampled

    return candidates
