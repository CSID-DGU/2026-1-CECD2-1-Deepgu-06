from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


class ResNetFrameFeatureExtractor:
    """ResNet-50 per-frame feature extractor.

    BGR 프레임 리스트 → (T, 2048) numpy array.
    fc 레이어를 제거하고 avgpool 출력을 feature로 사용. frozen 추론 전용.
    """

    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(self, device: str = "cpu", batch_size: int = 16):
        backbone = tv_models.resnet50(pretrained=True)
        self._model = nn.Sequential(*list(backbone.children())[:-1])
        self._model.eval()
        self._model.to(device)
        self.device = device
        self.batch_size = batch_size
        self._transform = T.Compose([
            T.Resize(256, interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=self._MEAN, std=self._STD),
        ])

    @torch.no_grad()
    def extract(self, frames_bgr: list) -> np.ndarray:
        """
        Args:
            frames_bgr: List[np.ndarray]  BGR (H, W, 3)
        Returns:
            np.ndarray (T, 2048)
        """
        tensors = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(self._transform(Image.fromarray(rgb)))

        all_feats = []
        for i in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[i : i + self.batch_size]).to(self.device)
            feats = self._model(batch).squeeze(-1).squeeze(-1)  # (B, 2048)
            all_feats.append(feats.cpu().numpy())

        return np.concatenate(all_feats, axis=0)  # (T, 2048)
