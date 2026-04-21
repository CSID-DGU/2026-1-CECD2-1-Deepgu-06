import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_default_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class FightClipDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        split="train",
        include_datasets=None,
        train_subsets=None,
        val_subsets=None,
        transform=None,
        require_existing_video=True,
        num_segments=8,
        max_samples=None,
        seed=42,
    ):
        self.metadata_path = Path(metadata_path)
        self.split = split
        self.include_datasets = set(include_datasets or [])
        self.train_subsets = set(train_subsets or ["training"])
        self.val_subsets = set(val_subsets or ["validation", "eval"])
        self.transform = transform or build_default_transform(train=(split == "train"))
        self.require_existing_video = require_existing_video
        self.num_segments = num_segments
        self.max_samples = max_samples
        self.seed = seed

        self.records = self._load_records()

    def _matches_split(self, record):
        subset = record.get("subset")
        if self.split == "train":
            return subset in self.train_subsets
        if self.split == "val":
            return subset in self.val_subsets
        return True

    def _load_records(self):
        records = load_jsonl(self.metadata_path)
        filtered = []

        for record in records:
            if self.include_datasets and record.get("dataset") not in self.include_datasets:
                continue
            if not self._matches_split(record):
                continue

            video_path = Path(record["video_path"])
            if self.require_existing_video and not video_path.exists():
                continue

            filtered.append(record)

        if self.max_samples is not None and len(filtered) > self.max_samples:
            rng = random.Random(self.seed)
            filtered = rng.sample(filtered, self.max_samples)

        return filtered

    def _read_clip(self, video_path, start_frame, end_frame):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")

        actual_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if actual_frame_count <= 0:
            cap.release()
            raise RuntimeError(f"failed to get frame count: {video_path}")

        clip_len = max(1, end_frame - start_frame + 1)
        max_end_frame = actual_frame_count - 1

        # Some dataset metadata overestimates frame count by a few frames.
        end_frame = min(end_frame, max_end_frame)
        start_frame = min(start_frame, max_end_frame)

        if start_frame > end_frame:
            start_frame = max(0, end_frame - clip_len + 1)

        if end_frame - start_frame + 1 < clip_len:
            start_frame = max(0, end_frame - clip_len + 1)

        frames = []
        frame_indices = list(range(start_frame, end_frame + 1))

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if not frames:
            raise RuntimeError(
                f"no frames read from {video_path} [{start_frame}, {end_frame}]"
            )

        while len(frames) < len(frame_indices):
            frames.append(frames[-1].copy())

        return frames

    def _sample_frames(self, frames):
        if len(frames) == self.num_segments:
            return frames

        if len(frames) < self.num_segments:
            sampled = list(frames)
            while len(sampled) < self.num_segments:
                sampled.append(sampled[-1].copy())
            return sampled

        indices = np.linspace(0, len(frames) - 1, num=self.num_segments, dtype=int)
        return [frames[idx] for idx in indices.tolist()]

    def _apply_transform(self, frames):
        tensors = []
        for frame in frames:
            image = Image.fromarray(frame)
            tensors.append(self.transform(image))

        clip_tensor = torch.stack(tensors)
        t, c, h, w = clip_tensor.shape
        return clip_tensor.view(t * c, h, w)

    def __getitem__(self, index):
        record = self.records[index]
        frames = self._read_clip(
            record["video_path"],
            int(record["clip_start_frame"]),
            int(record["clip_end_frame"]),
        )
        frames = self._sample_frames(frames)
        clip_tensor = self._apply_transform(frames)
        label = torch.tensor(float(record["label"]), dtype=torch.float32)

        return {
            "frames": clip_tensor,
            "label": label,
            "video_id": record["video_id"],
            "dataset": record["dataset"],
            "source": record["source"],
        }

    def __len__(self):
        return len(self.records)


def make_balanced_sampler(dataset):
    labels = [int(record["label"]) for record in dataset.records]
    if not labels:
        raise ValueError("dataset is empty")

    class_counts = np.bincount(labels, minlength=2)
    class_weights = np.zeros_like(class_counts, dtype=np.float64)

    for class_idx, count in enumerate(class_counts):
        if count > 0:
            class_weights[class_idx] = 1.0 / count

    sample_weights = [class_weights[label] for label in labels]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_pos_weight(dataset):
    labels = [int(record["label"]) for record in dataset.records]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(negatives / positives, dtype=torch.float32)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
