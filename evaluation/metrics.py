"""Binary region Dice metrics in WT/TC/ET order."""
import torch


def region_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Return [B,3] Dice; both-empty regions score 1, one-empty scores 0."""
    if logits.shape != target.shape or logits.ndim != 5 or logits.shape[1] != 3:
        raise ValueError("Expected matching [B,3,D,H,W] tensors.")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one.")
    prediction, truth = logits.sigmoid() >= threshold, target.bool()
    intersection = (prediction & truth).sum((2, 3, 4)).float()
    denominator = prediction.sum((2, 3, 4)) + truth.sum((2, 3, 4))
    return torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), 1.0)
