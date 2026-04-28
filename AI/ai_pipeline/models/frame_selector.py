import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameScorer(nn.Module):
    """
    각 frame의 '중요도 점수'를 출력하는 경량 모델.

    BiGRU로 시간적 맥락을 파악한 뒤 MLP로 frame별 scalar 점수를 냅니다.
    점수가 높은 frame일수록 InternVL2가 anomaly를 판단하는 데 유용한 frame입니다.
    """

    def __init__(self, input_dim=2048, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            batch_first=True, bidirectional=True
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        x: (T, input_dim)
        returns: scores (T,) — 값이 클수록 InternVL2에 유용한 frame
        """
        h, _ = self.gru(x.unsqueeze(0))          # (1, T, hidden*2)
        scores = self.scorer(h.squeeze(0))        # (T, 1)
        return scores.squeeze(-1)                 # (T,)


class DifferentiableFrameSelector(nn.Module):
    """
    학습용 모듈.

    훈련 시:
      FrameScorer → soft attention weights → features의 가중합
      → classifier → cross-entropy (pseudo-label 기준)

    추론 시:
      FrameScorer → top-k frame index 반환
      (KeyframeSampler Phase 2에서 FrameScorer만 따로 로드해서 사용)

    학습 아이디어:
      Phase 1 + InternVL2로 만든 pseudo-label을 정답으로 사용.
      "이 frame들을 선택했을 때 InternVL2가 anomaly를 맞혔는가"를
      classifier가 모사하도록 학습 → scorer는 그 classifier가 맞히게끔
      frame을 선택하는 법을 배움.
    """

    def __init__(self, input_dim=2048, hidden_dim=256,
                 n_frames=8, n_classes=2, dropout=0.3):
        super().__init__()
        self.n_frames = n_frames
        self.scorer = FrameScorer(input_dim, hidden_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, features, temperature=0.5):
        """
        features: (T, input_dim)
        temperature: softmax 온도. 낮을수록 hard selection에 가까워짐.
                     학습 초기엔 높게(1.0), 후기엔 낮게(0.1) 설정 권장.

        returns:
            logits: (n_classes,) — classifier 출력
            scores: (T,)         — frame별 중요도 점수 (로깅/디버깅용)
        """
        scores = self.scorer(features)                        # (T,)

        # soft top-k: 상위 n_frames frame에 집중되는 soft weight
        weights = self._soft_topk_weights(scores, temperature)  # (T,)

        # 가중합으로 clip 표현 생성
        clip_repr = (features * weights.unsqueeze(1)).sum(0)    # (input_dim,)

        logits = self.classifier(clip_repr)                    # (n_classes,)
        return logits, scores

    def _soft_topk_weights(self, scores, temperature):
        """
        상위 n_frames개에 집중되는 soft weight를 계산합니다.
        top-k 이후 해당 위치만 남기고 softmax → 미분 가능한 선택.
        """
        n = min(self.n_frames, scores.shape[0])

        # top-k 위치 마스크 (hard, gradient 안 흐름)
        topk_vals, topk_idx = torch.topk(scores, n)
        mask = torch.zeros_like(scores)
        mask[topk_idx] = 1.0

        # 마스크 영역만 softmax (gradient는 scores를 통해 흐름)
        masked_scores = scores * mask + (1 - mask) * (-1e9)
        weights = F.softmax(masked_scores / temperature, dim=0)
        return weights
