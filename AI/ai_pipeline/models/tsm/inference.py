import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class DummyTSM:
    """
     임시 TSM (진짜 모델 연결 전 테스트용)
    """

    def __init__(self):
        pass

    def predict(self, clip_frames):
        """
        clip_frames: list of numpy array (BGR)
        """

        # 랜덤 확률 (테스트용)
        probs = np.random.rand(5)
        probs = probs / probs.sum()

        class_names = ["walking", "running", "fighting", "sitting", "standing"]

        return dict(zip(class_names, probs))