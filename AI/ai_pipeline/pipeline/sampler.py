import os
import sys
import numpy as np
import torch
from sklearn.cluster import KMeans


class KeyframeSampler:
    """
    Phase 1 (model_path 없음): Temporal Clustering + Motion
      - K-means로 clip을 N구간으로 나눠 시간적 커버리지 보장
      - 각 구간에서 motion이 가장 큰 frame 선택
      - pseudo-label 생성 및 초기 동작에 사용

    Phase 2 (model_path 있음): 학습된 FrameScorer
      - Phase 1 + InternVL2로 생성한 pseudo-label로 학습된 모델 사용
      - InternVL2가 잘 이해하는 frame을 직접 선택하도록 학습됨

    입력 candidates 필수 필드:
        clip          : list of np.ndarray (BGR frames)
        features      : np.ndarray (T, 2048)  — ResNet-50 feature
    """

    def __init__(self, n_frames=8, model_path=None, device="cpu"):
        self.n_frames = n_frames
        self.device = device
        self.scorer = None

        if model_path and os.path.isfile(model_path):
            from models.frame_selector import FrameScorer
            self.scorer = FrameScorer(input_dim=2048).to(device)
            self.scorer.load_state_dict(
                torch.load(model_path, map_location=device)
            )
            self.scorer.eval()
            print(f"[KeyframeSampler] Phase 2 모델 로드: {model_path}")
        else:
            print("[KeyframeSampler] Phase 1 동작 (Temporal Clustering + Motion)")

    # ------------------------------------------------------------------
    # Phase 1: Temporal Clustering + Motion
    # ------------------------------------------------------------------

    @staticmethod
    def _motion_scores(features, alpha=0.5):
        """
        속도(velocity)와 가속도(acceleration)를 결합한 motion proxy (T,).

        velocity[t]     = ||f[t] - f[t-1]||  — 인접 frame 간 변화량 크기
        acceleration[t] = |vel[t] - vel[t-1]| — 변화량의 급격한 전환 강도

        alpha=1.0 이면 순수 velocity, alpha=0.0 이면 순수 acceleration.
        기본값 0.5: "많이 움직이면서 갑자기 바뀌는 frame" 선호.
        """
        vel = np.linalg.norm(features[1:] - features[:-1], axis=1)
        vel = np.concatenate([[0.0], vel])           # (T,)

        accel = np.abs(vel[1:] - vel[:-1])
        accel = np.concatenate([[0.0], accel])       # (T,)

        return alpha * vel + (1 - alpha) * accel

    def _phase1_select(self, frames, features):
        """
        K-means로 features를 N개 cluster로 나누고,
        각 cluster에서 motion이 가장 큰 frame index를 선택합니다.
        """
        T = len(frames)
        n = min(self.n_frames, T)

        if T <= n:
            return list(range(T))

        motion = self._motion_scores(features)

        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        selected = []
        for k in range(n):
            cluster_idx = np.where(labels == k)[0]
            if len(cluster_idx) == 0:
                continue
            best = cluster_idx[np.argmax(motion[cluster_idx])]
            selected.append(int(best))

        return sorted(selected)

    # ------------------------------------------------------------------
    # Phase 2: 학습된 FrameScorer
    # ------------------------------------------------------------------

    def _phase2_select(self, frames, features, win=16, stride=8):
        """
        슬라이딩 윈도우로 BiGRU를 적용해 전체 영상 frame 점수를 계산합니다.

        T <= win 이면 그냥 통으로 넣음 (clip-level 동작과 동일).
        T > win 이면 win 크기 윈도우를 stride 간격으로 슬라이드하면서
        각 윈도우의 frame별 점수를 누적(max)한 뒤 전체 top-n을 선택.
        """
        T = len(frames)
        n = min(self.n_frames, T)

        if T <= win:
            x = torch.FloatTensor(features).to(self.device)
            with torch.no_grad():
                scores = self.scorer(x).cpu().numpy()
        else:
            scores = np.full(T, -np.inf, dtype=np.float32)
            starts = list(range(0, T - win + 1, stride))
            if starts[-1] + win < T:          # 마지막 윈도우가 끝까지 못 닿을 때
                starts.append(T - win)
            for s in starts:
                e = s + win
                x = torch.FloatTensor(features[s:e]).to(self.device)
                with torch.no_grad():
                    win_scores = self.scorer(x).cpu().numpy()  # (win,)
                scores[s:e] = np.maximum(scores[s:e], win_scores)

        top_idx = np.argsort(scores)[::-1][:n]
        return sorted(top_idx.tolist())

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------

    def sample(self, candidate):
        frames = candidate["clip"]
        features = np.asarray(candidate["features"], dtype=np.float32)

        if self.scorer is not None:
            selected = self._phase2_select(frames, features)
        else:
            selected = self._phase1_select(frames, features)

        candidate["sampled_frames"] = [frames[i] for i in selected]
        candidate["selected_indices"] = selected
        return candidate

    def sample_from_candidates(self, candidates):
        for c in candidates:
            self.sample(c)
        return candidates


class PGLSumSampler:
    """
    PGL-SUM + Anomaly Proxy 기반 keyframe 선택기.

    keyframe_plan.md 에 기술된 방식:
      1. PGL-SUM 모델로 frame importance score (T,) 계산  ← "구조적 중요도"
      2. 인접 frame feature L2 거리로 anomaly proxy (T,) 계산  ← "변화량"
      3. 두 점수를 각각 min-max 정규화 후 element-wise 곱셈
      4. minimum temporal gap 조건 하에 top-k 선택

    KEYFRAME_DESIGN.md 에서 이 방식 대신 Phase 1/2를 채택한 이유:
      - PGL-SUM이 "대표 장면" 선택에 최적화 → "이상행동 장면"과 목적 불일치
      - anomaly proxy가 진짜 anomaly signal이 아닌 feature 변화량 대리 신호
    → Phase 1/2와의 정량 비교(evaluate_selector.py)로 채택 여부를 결정.

    PGL-SUM 모델 설정:
      - 기본값: input_size=2048 (ResNet-50 features, AI pipeline 기준)
      - 사전학습된 PGL-SUM은 GoogleNet features (1024-dim) 기반이므로
        ResNet-50 features로 학습된 별도 checkpoint가 필요합니다.
      - PGL-SUM 학습: PGL-SUM/ 디렉토리에서 --video_type 옵션으로 학습 가능.

    입력 candidate 필수 필드:
        clip     : list of np.ndarray (BGR frames)
        features : np.ndarray (T, input_size)  — ResNet-50 feature (T, 2048)
    """

    # PGL-SUM 기본 하이퍼파라미터 (논문/inference.py 기준)
    _PGLSUM_DEFAULTS = dict(
        num_segments=4,
        heads=8,
        fusion="add",
        pos_enc="absolute",
    )

    # CLIP 텍스트 프롬프트
    _ANOMALY_PROMPTS = [
        "a person fighting",
        "a person assaulting another person",
        "a violent physical altercation",
        "people involved in a fight",
        "an act of abuse or violence",
    ]
    _NORMAL_PROMPTS = [
        "people walking normally on a street",
        "a quiet everyday scene",
        "people going about their daily activities",
        "a normal public area with no violence",
        "ordinary crowd behavior",
    ]

    def __init__(
        self,
        model_path,
        n_frames=8,
        min_gap=4,
        input_size=2048,
        num_segments=4,
        heads=8,
        fusion="add",
        pos_enc="absolute",
        device="cpu",
        use_clip=False,
        clip_weight=0.5,
    ):
        """
        Args:
            model_path  : PGL-SUM checkpoint (.pth) 경로.
            n_frames    : 선택할 frame 수.
            min_gap     : 선택된 frame 간 최소 index 간격.
            input_size  : PGL-SUM 입력 feature 차원.
            use_clip    : True이면 CLIP anomaly score를 추가 신호로 사용.
                          candidate["clip"]에 실제 BGR 프레임이 있어야 함.
            clip_weight : 최종 스코어에서 CLIP 신호 비중.
                          final = pgl^(1-w) * clip^w
            device      : "cuda" or "cpu".
        """
        self.n_frames = n_frames
        self.min_gap = min_gap
        self.device = device
        self.use_clip = use_clip
        self.clip_weight = clip_weight
        self._clip_model = None
        self._clip_preprocess = None
        self._anomaly_feats = None

        # PGL-SUM 모델 로드
        # summarizer.py 는 `from layers.attention import` 를 사용하므로
        # model/ 과 model/layers/ 둘 다 sys.path 에 필요
        _pglsum_model  = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../PGL-SUM/model")
        )
        _pglsum_layers = os.path.join(_pglsum_model, "layers")
        for _p in [_pglsum_layers, _pglsum_model]:
            if _p not in sys.path:
                sys.path.insert(0, _p)

        from summarizer import PGL_SUM  # noqa: PGL-SUM/model/layers/summarizer.py

        if use_clip:
            self._load_clip()

        self.model = PGL_SUM(
            input_size=input_size,
            output_size=input_size,
            num_segments=num_segments,
            heads=heads,
            fusion=fusion,
            pos_enc=pos_enc,
        ).to(device)

        state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[PGLSumSampler] PGL-SUM 모델 로드: {model_path}  (input_size={input_size})")

    # ------------------------------------------------------------------
    # CLIP 로드
    # ------------------------------------------------------------------

    def _load_clip(self):
        import clip
        from PIL import Image as _PIL_Image
        self._PIL_Image = _PIL_Image
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        model.eval()
        self._clip_model = model
        self._clip_preprocess = preprocess
        with torch.no_grad():
            a_tokens = clip.tokenize(self._ANOMALY_PROMPTS).to(self.device)
            a_feats  = model.encode_text(a_tokens)
            self._anomaly_feats = a_feats / a_feats.norm(dim=-1, keepdim=True)

            n_tokens = clip.tokenize(self._NORMAL_PROMPTS).to(self.device)
            n_feats  = model.encode_text(n_tokens)
            self._normal_feats = n_feats / n_feats.norm(dim=-1, keepdim=True)
        print("[PGLSumSampler] CLIP ViT-B/32 로드 완료 (contrastive anomaly scoring 활성화)")

    def _clip_scores(self, frames_bgr):
        """
        실제 BGR 프레임 리스트 → 프레임별 CLIP anomaly similarity (T,).
        None 항목은 0.0으로 처리.
        """
        import cv2
        T = len(frames_bgr)
        scores = np.zeros(T, dtype=np.float32)
        images, valid_idx = [], []
        for i, f in enumerate(frames_bgr):
            if f is None:
                continue
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil = self._PIL_Image.fromarray(rgb)
            images.append(self._clip_preprocess(pil))
            valid_idx.append(i)
        if not images:
            return scores
        img_tensor = torch.stack(images).to(self.device)
        with torch.no_grad():
            img_feats = self._clip_model.encode_image(img_tensor)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        sims = (img_feats @ self._anomaly_feats.T).mean(dim=1).cpu().numpy()
        for i, vi in enumerate(valid_idx):
            scores[vi] = float(sims[i])
        return scores

    # ------------------------------------------------------------------
    # 내부 계산
    # ------------------------------------------------------------------

    @staticmethod
    def _minmax(arr, eps=1e-8):
        """배열을 [0, 1] 범위로 min-max 정규화."""
        arr = np.asarray(arr, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + eps)

    def _pgl_scores(self, features):
        """
        PGL-SUM으로 frame importance score (T,) 계산.
        PGL_SUM.forward() 반환값: (1, T), 이미 sigmoid 통과 → [0, 1] 범위.
        """
        x = torch.FloatTensor(features).to(self.device)  # (T, D)
        with torch.no_grad():
            scores, _ = self.model(x)                    # (1, T)
        return scores.squeeze(0).cpu().numpy()            # (T,)

    @staticmethod
    def _anomaly_proxy(features, alpha=0.5):
        """
        속도(velocity)와 가속도(acceleration)를 결합한 anomaly proxy (T,).

        velocity[t]     = ||f[t] - f[t-1]||  — 인접 frame 간 변화량 크기
        acceleration[t] = |vel[t] - vel[t-1]| — 변화량의 급격한 전환 강도

        alpha=1.0 이면 순수 velocity, alpha=0.0 이면 순수 acceleration.
        기본값 0.5: "많이 움직이면서 갑자기 바뀌는 frame" 선호.
        """
        vel = np.linalg.norm(features[1:] - features[:-1], axis=1)
        vel = np.concatenate([[0.0], vel])           # (T,)

        accel = np.abs(vel[1:] - vel[:-1])
        accel = np.concatenate([[0.0], accel])       # (T,)

        return alpha * vel + (1 - alpha) * accel

    def _select_topk(self, scores, k):
        """
        min_gap 조건을 만족하면서 score 상위 k개 frame index 선택.
        gap 조건으로 k개 못 채울 경우 gap 없이 나머지 순위대로 채움 (fallback).
        """
        order = np.argsort(scores)[::-1]
        selected = []
        for idx in map(int, order):
            if all(abs(idx - s) >= self.min_gap for s in selected):
                selected.append(idx)
                if len(selected) == k:
                    break

        # gap 조건 충족 불가 시 fallback
        if len(selected) < k:
            for idx in map(int, order):
                if idx not in selected:
                    selected.append(idx)
                    if len(selected) == k:
                        break

        return sorted(selected)

    # ------------------------------------------------------------------
    # 공개 인터페이스 (KeyframeSampler와 동일)
    # ------------------------------------------------------------------

    def sample(self, candidate):
        """
        candidate dict 에 "sampled_frames" 와 "selected_indices" 를 채워 반환.

        처리 순서:
          1. PGL-SUM importance score (T,)
          2. anomaly proxy (T,)  ← 인접 frame feature L2 거리
          3. 둘 다 min-max 정규화 후 element-wise 곱 → final_scores
          4. min_gap 조건 하에 top-k 선택
        """
        frames = candidate["clip"]
        features = np.asarray(candidate["features"], dtype=np.float32)
        T = len(frames)
        n = min(self.n_frames, T)

        if T <= n:
            candidate["sampled_frames"] = list(frames)
            candidate["selected_indices"] = list(range(T))
            return candidate

        pgl = self._minmax(self._pgl_scores(features))       # (T,)

        has_real_frames = any(f is not None for f in frames)
        if self._clip_model is not None and has_real_frames:
            # CLIP anomaly score × PGL-SUM score (가중 결합)
            clip_sc = self._minmax(self._clip_scores(frames))  # (T,)
            w = self.clip_weight
            final_scores = (pgl ** (1 - w)) * (clip_sc ** w)
        else:
            proxy = self._minmax(self._anomaly_proxy(features))
            final_scores = pgl * proxy

        selected = self._select_topk(final_scores, n)
        candidate["sampled_frames"] = [frames[i] for i in selected]
        candidate["selected_indices"] = selected
        return candidate

    def sample_from_candidates(self, candidates):
        for c in candidates:
            self.sample(c)
        return candidates
