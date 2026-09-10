"""Small convolutional encoder for the baseline U-Net."""
import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two padded 3D convolutions with ReLU activations."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Encode two skip resolutions and a bottleneck."""

    def __init__(self, in_channels: int = 4, base_channels: int = 16) -> None:
        super().__init__()
        self.level1 = conv_block(in_channels, base_channels)
        self.level2 = conv_block(base_channels, base_channels * 2)
        self.bottleneck = conv_block(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool3d(2)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = self.level1(image)
        second = self.level2(self.pool(first))
        return first, second, self.bottleneck(self.pool(second))
