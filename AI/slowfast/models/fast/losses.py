import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.detach().clone().to(torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        targets = targets.to(torch.float32)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
        focal_factor = (1.0 - pt).pow(self.gamma)
        loss = bce * focal_factor

        if self.alpha is not None:
            alpha_factor = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
            loss = loss * alpha_factor

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
