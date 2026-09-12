"""Four-resolution convolutional encoder, reusable independently of its decoder."""
import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two padded convolutions with small-batch GroupNorm and ReLU.

    Keep at least two channels per group, including at a 1-voxel bottleneck.
    """
    groups = next((g for g in (8, 4, 2, 1) if out_channels % g == 0 and out_channels // g >= 2), 1)
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Return three skip features and a bottleneck, at widths b,2b,4b,8b."""

    def __init__(self, in_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        widths = [base_channels * factor for factor in (1, 2, 4, 8)]
        self.blocks = nn.ModuleList(conv_block(source, dest) for source, dest in zip([in_channels, *widths[:-1]], widths))
        self.pool = nn.MaxPool3d(2)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = []
        for index, block in enumerate(self.blocks):
            image = block(self.pool(image) if index else image)
            features.append(image)
        return tuple(features)
