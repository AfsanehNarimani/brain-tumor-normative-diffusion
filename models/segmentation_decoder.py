"""Skip-connected decoder producing raw per-tissue logits."""
import torch
from torch import nn
from torch.nn import functional as F
from models.encoder import conv_block


class SegmentationDecoder(nn.Module):
    """Interpolate explicitly to skip shapes, including odd spatial sizes."""

    def __init__(self, base_channels: int = 32, out_channels: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(conv_block(base_channels * factor * 3, base_channels * factor)
                                    for factor in (4, 2, 1))
        self.head = nn.Conv3d(base_channels, out_channels, 1)

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError("Decoder requires three skip features and a bottleneck.")
        image = features[-1]
        for block, skip in zip(self.blocks, reversed(features[:-1])):
            image = F.interpolate(image, size=skip.shape[2:], mode="trilinear", align_corners=False)
            image = block(torch.cat((image, skip), dim=1))
        return self.head(image)
