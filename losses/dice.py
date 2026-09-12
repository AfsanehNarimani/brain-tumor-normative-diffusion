"""Differentiable region Dice loss."""
import torch
from torch import nn


class DiceLoss(nn.Module):
    """Average soft Dice loss over subjects and tissue channels."""

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape or logits.ndim != 5 or any(size == 0 for size in logits.shape):
            raise ValueError("Expected matching [B,C,D,H,W] logits and targets.")
        probabilities = logits.sigmoid()
        intersection = (probabilities * target).sum(dim=(2, 3, 4))
        denominator = (probabilities + target).sum(dim=(2, 3, 4))
        return (1 - (2 * intersection + self.smooth) / (denominator + self.smooth)).mean()
