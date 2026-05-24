import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    """Penalizes large changes in segmentation between consecutive frames."""
    def forward(self, pred_t: torch.Tensor, pred_t1: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_t, pred_t1)


class BoundarySupervisionLoss(nn.Module):
    """Weighted BCE that puts extra emphasis on boundary pixels."""
    def __init__(self, boundary_weight: float = 5.0):
        super().__init__()
        self.boundary_weight = boundary_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                boundary_mask: torch.Tensor = None) -> torch.Tensor:
        base_loss = F.binary_cross_entropy(pred, target.float())
        if boundary_mask is None:
            return base_loss
        weighted = base_loss + self.boundary_weight * F.binary_cross_entropy(
            pred * boundary_mask, target.float() * boundary_mask
        )
        return weighted


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft Dice loss for binary segmentation."""
    pred   = pred.flatten(1)
    target = target.float().flatten(1)
    num    = (2 * pred * target).sum(1)
    denom  = pred.sum(1) + target.sum(1) + 1e-8
    return (1 - num / denom).mean()
