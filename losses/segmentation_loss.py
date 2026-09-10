"""Region-based Dice plus numerically stable binary cross entropy."""
import torch
from torch import nn
from losses.dice import DiceLoss


class SegmentationLoss(nn.Module):
    """Weight Dice and BCE, averaging each over the three regions."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0) -> None:
        super().__init__()
        if min(dice_weight, bce_weight) < 0 or dice_weight + bce_weight == 0:
            raise ValueError("Loss weights must be nonnegative and not both zero.")
        self.dice_weight, self.bce_weight = dice_weight, bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.bce_weight * self.bce(logits, target)
