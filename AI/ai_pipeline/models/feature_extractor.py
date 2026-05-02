"""
ResNet-50 frame feature 추출기.

candidate filtering 팀이 사용하는 것과 동일한 ResNet-50을 재현합니다.
학습 데이터 준비(prepare_data.py)에서 독립적으로 feature를 추출할 때 사용합니다.
실제 파이프라인에서는 candidate filtering 결과의 features 필드를 그대로 사용합니다.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class ResNet50Extractor:
    """
    ResNet-50의 avgpool 출력(2048-dim)을 frame feature로 사용합니다.
    fc layer는 제거합니다.
    """

    def __init__(self, device="cuda", batch_size=32):
        self.device = device
        self.batch_size = batch_size

        model = models.resnet50(pretrained=True)
        # fc 제거 → avgpool 출력: (B, 2048, 1, 1)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.backbone.eval().to(device)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def extract_from_frames(self, frames):
        """
        frames: list of np.ndarray (BGR, HWC)
        returns: np.ndarray (T, 2048)
        """
        if not frames:
            return np.zeros((0, 2048), dtype=np.float32)

        all_features = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            tensors = []
            for f in batch:
                img = Image.fromarray(f[..., ::-1])  # BGR → RGB
                tensors.append(self.transform(img))

            x = torch.stack(tensors).to(self.device)
            feat = self.backbone(x)                  # (B, 2048, 1, 1)
            feat = feat.squeeze(-1).squeeze(-1)      # (B, 2048)
            all_features.append(feat.cpu().numpy())

        return np.concatenate(all_features, axis=0).astype(np.float32)
