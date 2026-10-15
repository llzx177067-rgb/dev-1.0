from typing import NamedTuple

import torch
import torch.nn.functional as F


class DecisionLoss(NamedTuple):
    total: torch.Tensor
    cross_entropy: torch.Tensor
    brier: torch.Tensor


def decision_loss(logits: torch.Tensor, target: torch.Tensor, brier_weight: float = 0.0) -> DecisionLoss:
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    cross_entropy = -(target * log_probabilities).sum(-1).mean()
    brier = (log_probabilities.exp() - target).square().sum(-1).mean()
    total = cross_entropy if brier_weight == 0 else cross_entropy + brier_weight * brier
    return DecisionLoss(total, cross_entropy, brier)
