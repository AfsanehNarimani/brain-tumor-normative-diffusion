"""Independent full-width bottleneck projections; no disentanglement objective."""
import torch
from torch import nn


class BottleneckProjection(nn.Module):
    """Two learnable 1x1x1 convolutions with independent weights and biases."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
            raise ValueError('channels must be a positive integer.')
        self.anatomy_projector = nn.Conv3d(channels, channels, 1)
        self.disease_projector = nn.Conv3d(channels, channels, 1)

    def forward(self, bottleneck: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.anatomy_projector(bottleneck), self.disease_projector(bottleneck)
