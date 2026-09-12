"""Modest pure-PyTorch 3D U-Net with configurable input/output channels."""
import torch
from torch import nn
from models.encoder import Encoder
from models.segmentation_decoder import SegmentationDecoder


class BaselineUNet3D(nn.Module):
    """Return raw [B,out_channels,D,H,W] logits, without sigmoid.

    Defaults: four modalities to NETC/SNFH/ET/RC, widths 32/64/128/256.
    Three pooling steps require spatial dimensions >= 8.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in (in_channels, out_channels, base_channels)):
            raise ValueError("Channel counts must be positive integers.")
        self.in_channels = in_channels
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = SegmentationDecoder(base_channels, out_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != self.in_channels or min(image.shape[2:]) < 8:
            raise ValueError(f"Expected [B,{self.in_channels},D,H,W] with each spatial dimension >= 8.")
        return self.decoder(self.encoder(image))
