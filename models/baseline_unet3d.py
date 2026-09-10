"""Pure PyTorch multimodal region-based 3D U-Net."""
import torch
from torch import nn
from models.encoder import Encoder
from models.segmentation_decoder import SegmentationDecoder


class BaselineUNet3D(nn.Module):
    """Map [B,4,D,H,W] to [B,3,D,H,W] WT/TC/ET logits."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        if in_channels != 4 or out_channels != 3 or base_channels < 1:
            raise ValueError("Baseline requires 4 inputs, 3 outputs, and positive base_channels.")
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = SegmentationDecoder(base_channels, out_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 4 or min(image.shape[2:]) < 4:
            raise ValueError("Expected [B,4,D,H,W] with each spatial dimension >= 4.")
        return self.decoder(self.encoder(image))
