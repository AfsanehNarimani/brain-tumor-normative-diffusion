"""Skip-connected decoder producing independent region logits."""
import torch
from torch import nn
from torch.nn import functional as F
from models.encoder import conv_block


class SegmentationDecoder(nn.Module):
    """Resize to skip shapes to support odd spatial dimensions."""

    def __init__(self, base_channels: int = 16, out_channels: int = 3) -> None:
        super().__init__()
        self.level2 = conv_block(base_channels * 6, base_channels * 2)
        self.level1 = conv_block(base_channels * 3, base_channels)
        self.head = nn.Conv3d(base_channels, out_channels, 1)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        first, second, latent = features
        up = F.interpolate(latent, size=second.shape[2:], mode="trilinear", align_corners=False)
        up = self.level2(torch.cat((up, second), dim=1))
        up = F.interpolate(up, size=first.shape[2:], mode="trilinear", align_corners=False)
        return self.head(self.level1(torch.cat((up, first), dim=1)))
