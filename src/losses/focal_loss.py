from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class and binary Focal Loss with optional per-class weighting."""

    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor, Sequence[float]]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if alpha is not None:
            if isinstance(alpha, (int, float)):
                alpha = torch.tensor(float(alpha), dtype=torch.float32)
            elif not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Binary Focal Loss (single output neuron)
        if inputs.ndim == 1 or inputs.shape[-1] == 1:
            inputs = inputs.view(-1)
            targets = targets.view(-1).float()

            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )
            p = torch.sigmoid(inputs)
            pt = p * targets + (1.0 - p) * (1.0 - targets)
            focal_weight = (1.0 - pt) ** self.gamma

            if self.alpha is not None:
                alpha = self.alpha.to(inputs.device)
                if alpha.ndim == 0:  # scalar alpha for class 1
                    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
                elif alpha.ndim == 1 and len(alpha) == 2:  # [alpha_0, alpha_1]
                    alpha_t = alpha[1] * targets + alpha[0] * (1.0 - targets)
                else:
                    alpha_t = 1.0
                focal_weight = alpha_t * focal_weight

            focal_loss = focal_weight * bce_loss
        else:
            # Multiclass Focal Loss
            weight = self.alpha.to(inputs.device) if self.alpha is not None else None
            ce_loss = F.cross_entropy(
                inputs, targets.long(), reduction="none", weight=weight
            )
            pt = torch.exp(-F.cross_entropy(inputs, targets.long(), reduction="none"))
            focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
