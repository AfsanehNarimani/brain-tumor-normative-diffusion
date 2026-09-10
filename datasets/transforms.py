"""Deterministic modality-wise preprocessing."""
import torch


def normalize_nonzero(image: torch.Tensor) -> torch.Tensor:
    """Z-score nonzero voxels per modality; preserve zero background."""
    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError("Expected [4,D,H,W].")
    result = image.clone().float()
    for channel in result:
        foreground = channel != 0
        if foreground.any():
            values = channel[foreground]
            channel[foreground] = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)
    return result
