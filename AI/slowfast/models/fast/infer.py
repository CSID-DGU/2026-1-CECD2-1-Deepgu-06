from pathlib import Path

import numpy as np
from utils.io import load_json

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency for heuristic mode
    torch = None


class FastViolenceScorer:
    def __init__(self, config):
        self.config = config
        self.device = self._resolve_device(config.get("device", "auto"))
        self.batch_size = int(config.get("batch_size", 8))
        self.fallback_mode = config.get("fallback_mode", "heuristic")
        self.temperature = 1.0
        self.model = None
        self._load_model_if_available()

    def _resolve_device(self, value):
        if torch is None:
            return "cpu"
        if value == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return value

    def _load_model_if_available(self):
        checkpoint_path = self.config.get("checkpoint_path")
        if not checkpoint_path:
            return
        if torch is None:
            raise RuntimeError("torch is required when using a trained fast-model checkpoint")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        from models.fast.x3d_model import build_fast_model

        model = build_fast_model(
            architecture=self.config.get("architecture", "x3d_s"),
            num_classes=int(self.config.get("num_classes", 1)),
            pretrained=bool(self.config.get("use_pretrained_backbone", False)),
            input_clip_length=int(self.config.get("input_clip_length", 13)),
            input_crop_size=int(self.config.get("input_crop_size", 160)),
        )
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(self.device)
        self.model = model
        self.temperature = self._resolve_temperature(state)

        # blocks[-2]: head 직전 마지막 backbone stage.
        # forward 시 (B, 192, T', H', W') 출력 → 공간 평균 → (B, T', 192) 저장.
        self._backbone_features = None
        self.model.blocks[-2].register_forward_hook(self._on_backbone_output)

    def _on_backbone_output(self, module, input, output):
        # output: (B, C=192, T', H', W')
        feat = output.mean(dim=[-2, -1])                       # (B, 192, T')
        self._backbone_features = feat.permute(0, 2, 1).detach()  # (B, T', 192)

    def _resolve_temperature(self, checkpoint_state):
        calibration_path = self.config.get("calibration_path")
        if calibration_path:
            payload = load_json(calibration_path)
            return max(float(payload.get("temperature", 1.0)), 1e-3)
        if isinstance(checkpoint_state, dict):
            calibration = checkpoint_state.get("calibration")
            if isinstance(calibration, dict) and "temperature" in calibration:
                return max(float(calibration["temperature"]), 1e-3)
        return max(float(self.config.get("temperature", 1.0)), 1e-3)

    def score_clips(self, clips, clip_config, return_features=False):
        if not clips:
            return ([], []) if return_features else []
        if self.model is None:
            scores = [self._heuristic_score(item["frames"]) for item in clips]
            if return_features:
                return scores, [None] * len(scores)
            return scores
        return self._model_score(clips, clip_config, return_features=return_features)

    def _heuristic_score(self, frames):
        if len(frames) < 2:
            return 0.0
        diffs = []
        for prev, cur in zip(frames[:-1], frames[1:]):
            diffs.append(float(np.mean(np.abs(cur.astype(np.float32) - prev.astype(np.float32)))))
        motion = float(np.mean(diffs)) / 255.0
        score = max(0.0, min(1.0, motion * 3.0))
        return score

    def _model_score(self, clips, clip_config, return_features=False):
        from models.fast.transforms import preprocess_clip_frames

        resize_width = int(clip_config["resize"]["width"])
        resize_height = int(clip_config["resize"]["height"])
        num_samples = int(clip_config["sampled_frames"])
        sampling = str(clip_config.get("sampling", "uniform"))

        scores = []
        features = [] if return_features else None
        batch = []

        for clip in clips:
            batch.append(
                preprocess_clip_frames(
                    clip["frames"],
                    num_samples=num_samples,
                    resize_width=resize_width,
                    resize_height=resize_height,
                    sampling=sampling,
                )
            )
            if len(batch) == self.batch_size:
                if return_features:
                    s, f = self._run_batch_with_features(batch)
                    scores.extend(s)
                    features.extend(f)
                else:
                    scores.extend(self._run_batch(batch))
                batch = []

        if batch:
            if return_features:
                s, f = self._run_batch_with_features(batch)
                scores.extend(s)
                features.extend(f)
            else:
                scores.extend(self._run_batch(batch))

        return (scores, features) if return_features else scores

    def _run_batch(self, batch):
        inputs = torch.stack(batch).to(self.device)
        with torch.no_grad():
            logits = self.model(inputs).flatten()
            logits = logits / self.temperature
            probs = torch.sigmoid(logits).detach().cpu().tolist()
        return [float(item) for item in probs]

    def _run_batch_with_features(self, batch):
        self._backbone_features = None
        inputs = torch.stack(batch).to(self.device)
        with torch.no_grad():
            logits = self.model(inputs).flatten()
            logits = logits / self.temperature
            probs = torch.sigmoid(logits).detach().cpu().tolist()
        # hook이 채워준 (B, T', 192) → clip별 list로 분리
        if self._backbone_features is not None:
            feats = self._backbone_features.cpu().numpy()
            return [float(p) for p in probs], [feats[i] for i in range(len(batch))]
        return [float(p) for p in probs], [None] * len(batch)
