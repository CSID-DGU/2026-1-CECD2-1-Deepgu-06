import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from models.tsm.temporal_shift import make_temporal_shift
from utils.device import choose_torch_device


class TSMModel(nn.Module):
    def __init__(self, num_segments=16, num_classes=1):
        super().__init__()
        self.num_segments = num_segments

        self.base_model = models.resnet50(pretrained=False)
        self.base_model.fc = nn.Linear(
            self.base_model.fc.in_features, num_classes
        )

        make_temporal_shift(
            self.base_model,
            n_segment=num_segments,
            n_div=8,
            place='blockres'
        )

    def forward(self, x):
        N, TC, H, W = x.shape
        T = self.num_segments
        C = TC // T

        x = x.view(N * T, C, H, W)
        out = self.base_model(x)
        out = out.view(N, T, -1)
        out = out.mean(dim=1)

        return out


class TSMInference:
    def __init__(
        self,
        weight_path,
        device=None,
        preferred_gpu_indices=None,
        num_segments=16,
        num_classes=1,
        binary_mode=True,
    ):

        if device is None:
            self.device = choose_torch_device(
                preferred_gpu_indices=preferred_gpu_indices,
                allow_cpu_fallback=True
            )
        else:
            self.device = device

        self.num_segments = num_segments
        self.num_classes = num_classes
        self.binary_mode = binary_mode

        self.model = TSMModel(
            num_segments=self.num_segments,
            num_classes=self.num_classes
        )

        checkpoint = torch.load(weight_path, map_location=self.device)

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("module."):
                k = k[7:]
            state_dict[k] = v

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _sample_frames(self, clip):
        """
        clip 길이를 num_segments로 맞추는 함수
        """
        T = len(clip)
        idxs = torch.linspace(0, T - 1, steps=self.num_segments).long()
        return [clip[i] for i in idxs]

    def _prepare_clip_tensor(self, clip):
        clip = self._sample_frames(clip)

        clip = [
            Image.fromarray(f) if not isinstance(f, Image.Image) else f
            for f in clip
        ]

        frames = [self.transform(f) for f in clip]
        x = torch.stack(frames)

        T, C, H, W = x.shape
        return x.view(T * C, H, W)

    def _predict_tensor_batch(self, x):
        with torch.inference_mode():
            logits = self.model(x)

            if self.binary_mode and self.num_classes == 1:
                probs = torch.sigmoid(logits).squeeze(-1)
            else:
                probs = F.softmax(logits, dim=1)

        return probs.cpu()

    def predict(self, clip):
        x = self._prepare_clip_tensor(clip).unsqueeze(0).to(self.device)
        probs = self._predict_tensor_batch(x)
        return probs[0]

    def predict_batch(self, clips, batch_size=8):
        outputs = []
        prepared_batch = []

        for clip in clips:
            prepared_batch.append(self._prepare_clip_tensor(clip))
            if len(prepared_batch) < batch_size:
                continue

            x = torch.stack(prepared_batch).to(self.device)
            outputs.extend(self._predict_tensor_batch(x))
            prepared_batch = []

        if prepared_batch:
            x = torch.stack(prepared_batch).to(self.device)
            outputs.extend(self._predict_tensor_batch(x))

        return outputs
