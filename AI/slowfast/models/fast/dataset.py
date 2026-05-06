from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from models.fast.transforms import preprocess_clip_frames


class FightClipDataset(Dataset):
    def __init__(
        self,
        csv_path,
        resize_width=160,
        resize_height=160,
        num_samples=13,
        sampling="uniform",
    ):
        self.table = pd.read_csv(csv_path)
        self.resize_width = int(resize_width)
        self.resize_height = int(resize_height)
        self.num_samples = int(num_samples)
        self.sampling = str(sampling)

    def __len__(self):
        return len(self.table)

    def _load_clip_frames(self, video_path, start_frame, end_frame):
        cap = cv2.VideoCapture(str(Path(video_path)))
        if not cap.isOpened():
            raise ValueError(f"failed to open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        frames = []
        target_count = int(end_frame) - int(start_frame) + 1
        for _ in range(max(target_count, 0)):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(
                f"failed to decode frames for clip: video={video_path} start={start_frame} end={end_frame}"
            )
        return frames

    def __getitem__(self, index):
        row = self.table.iloc[index]
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        clip_frames = self._load_clip_frames(row["video_path"], start_frame, end_frame)
        tensor = preprocess_clip_frames(
            clip_frames,
            num_samples=self.num_samples,
            resize_width=self.resize_width,
            resize_height=self.resize_height,
            sampling=self.sampling,
        )
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return {
            "clip_id": row["clip_id"],
            "video_id": row["video_id"],
            "inputs": tensor,
            "label": label,
        }
