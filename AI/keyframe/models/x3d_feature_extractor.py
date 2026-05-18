"""
X3D-S spatiotemporal feature 추출기.

clip을 입력받아 head 이전 stage4 output을 공간 평균내어
frame-level feature (T', FEAT_DIM) 형태로 반환합니다.

ResNet50Extractor의 (T, 2048) 대신 (T', 192)를 출력하며,
BiGRU FrameScorer(input_dim=192)와 함께 사용합니다.

X3D-S는 3D conv로 시간축 정보를 직접 모델링하므로
ResNet-50(프레임 독립 처리)보다 더 풍부한 temporal feature를 제공합니다.

로드:
  torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
  첫 실행 시 인터넷에서 가중치를 다운로드합니다 (~12MB).
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class X3DFeatureExtractor:
    # X3D-S stage4 출력 채널 (head projection 이전)
    FEAT_DIM = 192
    # X3D-S 표준 입력 프레임 수 / 해상도
    N_FRAMES = 13
    INPUT_SIZE = 160

    def __init__(self, device="cuda"):
        self.device = device

        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "x3d_s",
            pretrained=True,
        )
        # blocks[-1]이 X3DHead (global pool + projection + fc) → 제거
        # blocks[0..4]: stem + 4 ResStages → (B, 192, T', H', W') 출력
        self.backbone = nn.Sequential(*list(model.blocks[:-1]))
        self.backbone.eval().to(device)

        # pytorchvideo X3D 전처리 기준
        self.transform = transforms.Compose([
            transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225],
            ),
        ])

    def _sample_frames(self, frames):
        """가변 길이 clip → N_FRAMES 균등 샘플링."""
        T = len(frames)
        if T == self.N_FRAMES:
            return frames
        idxs = np.linspace(0, T - 1, self.N_FRAMES).round().astype(int)
        return [frames[i] for i in idxs]

    def _to_tensor(self, frames):
        """
        frames: list of np.ndarray (BGR, HWC)
        returns: (1, C, T, H, W)
        """
        tensors = []
        for f in frames:
            img = Image.fromarray(f[..., ::-1])   # BGR → RGB
            tensors.append(self.transform(img))    # (C, H, W)
        # stack along temporal dim: (C, T, H, W) → add batch dim
        x = torch.stack(tensors, dim=1).unsqueeze(0)  # (1, C, T, H, W)
        return x

    @torch.no_grad()
    def extract_from_frames(self, frames):
        """
        frames: list of np.ndarray (BGR, HWC)
        returns: np.ndarray (T', FEAT_DIM)
          T' = N_FRAMES (X3D-S는 시간 축 다운샘플링 없음)
        """
        if not frames:
            return np.zeros((0, self.FEAT_DIM), dtype=np.float32)

        sampled = self._sample_frames(frames)
        x = self._to_tensor(sampled).to(self.device)   # (1, C, T, H, W)

        feat = self.backbone(x)                         # (1, C', T', H', W')
        feat = feat.mean(dim=[-2, -1])                  # spatial avg → (1, C', T')
        feat = feat.squeeze(0).permute(1, 0)            # (T', C')

        return feat.cpu().numpy().astype(np.float32)
