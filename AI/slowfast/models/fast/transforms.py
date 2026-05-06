from typing import List

import cv2
import numpy as np
import torch


def resize_frames(frames: List[np.ndarray], width: int, height: int) -> List[np.ndarray]:
    resized = []
    for frame in frames:
        resized.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR))
    return resized


def temporal_uniform_subsample(frames: List[np.ndarray], num_samples: int) -> List[np.ndarray]:
    if not frames:
        raise ValueError("cannot sample from an empty frame list")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    indices = np.linspace(0, len(frames) - 1, num=num_samples)
    return [frames[int(round(index))] for index in indices]


def _indices_from_bins(bin_sizes: List[int], num_samples: int, total_frames: int) -> List[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if sum(bin_sizes) != num_samples:
        raise ValueError("bin_sizes must sum to num_samples")

    # Split the full clip into contiguous temporal zones and sample within each zone.
    edges = np.linspace(0, total_frames, num=len(bin_sizes) + 1)
    indices: List[int] = []
    for zone_index, count in enumerate(bin_sizes):
        start = int(np.floor(edges[zone_index]))
        end = int(np.floor(edges[zone_index + 1])) - 1
        if zone_index == len(bin_sizes) - 1:
            end = total_frames - 1
        end = max(start, min(end, total_frames - 1))
        zone_indices = np.linspace(start, end, num=count)
        indices.extend(int(round(index)) for index in zone_indices)
    return [max(0, min(index, total_frames - 1)) for index in indices]


def temporal_anchor_balanced_subsample(frames: List[np.ndarray], num_samples: int) -> List[np.ndarray]:
    if not frames:
        raise ValueError("cannot sample from an empty frame list")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_samples != 13:
        return temporal_uniform_subsample(frames, num_samples=num_samples)
    indices = _indices_from_bins([3, 2, 3, 2, 3], num_samples=num_samples, total_frames=len(frames))
    return [frames[index] for index in indices]


def temporal_center_biased_subsample(frames: List[np.ndarray], num_samples: int) -> List[np.ndarray]:
    if not frames:
        raise ValueError("cannot sample from an empty frame list")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_samples != 13:
        return temporal_uniform_subsample(frames, num_samples=num_samples)
    indices = _indices_from_bins([2, 3, 3, 3, 2], num_samples=num_samples, total_frames=len(frames))
    return [frames[index] for index in indices]


def temporal_subsample(
    frames: List[np.ndarray],
    num_samples: int,
    sampling: str = "uniform",
) -> List[np.ndarray]:
    mode = str(sampling).lower()
    if mode == "uniform":
        return temporal_uniform_subsample(frames, num_samples=num_samples)
    if mode == "anchor_balanced":
        return temporal_anchor_balanced_subsample(frames, num_samples=num_samples)
    if mode == "center_biased":
        return temporal_center_biased_subsample(frames, num_samples=num_samples)
    raise ValueError(f"unsupported clip sampling mode: {sampling}")


def frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    array = np.stack(rgb_frames).astype(np.float32) / 255.0
    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.225, 0.225, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    array = (array - mean) / std
    array = np.transpose(array, (3, 0, 1, 2))
    return torch.from_numpy(array)


def preprocess_clip_frames(
    frames: List[np.ndarray],
    num_samples: int,
    resize_width: int,
    resize_height: int,
    sampling: str = "uniform",
) -> torch.Tensor:
    sampled = temporal_subsample(frames, num_samples=num_samples, sampling=sampling)
    resized = resize_frames(sampled, resize_width, resize_height)
    return frames_to_tensor(resized)
