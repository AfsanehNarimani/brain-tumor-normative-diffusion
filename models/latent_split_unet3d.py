"""E1-A latent-split control, not evidence of healthy/disease disentanglement."""
from dataclasses import dataclass
import torch
from torch import nn
from models.encoder import Encoder
from models.segmentation_decoder import SegmentationDecoder
from models.latent_projection import BottleneckProjection


@dataclass(frozen=True)
class LatentSplitOutput:
    logits: torch.Tensor
    anatomy_latent: torch.Tensor
    disease_latent: torch.Tensor


class LatentSplitUNet3D(nn.Module):
    """Decode disease projection AND unchanged encoder skips into raw logits.

    Anatomy projection has no direct task supervision or segmentation gradients.
    Both latents retain all 8*base_channels channels. Encoder runs once per call.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 32,
                 latent_projection: str = 'conv1x1x1', latent_channels: int | None = None) -> None:
        super().__init__()
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (in_channels, out_channels, base_channels)):
            raise ValueError('Channel counts must be positive integers.')
        if latent_projection != 'conv1x1x1':
            raise ValueError('E1-A requires conv1x1x1 projections.')
        self.latent_channels = 8 * base_channels
        if latent_channels is not None and (type(latent_channels) is not int or latent_channels != self.latent_channels):
            raise ValueError('latent_channels must equal 8 * base_channels.')
        self.in_channels = in_channels
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = SegmentationDecoder(base_channels, out_channels)
        self.projection = BottleneckProjection(self.latent_channels)

    def forward_with_latents(self, image: torch.Tensor) -> LatentSplitOutput:
        if image.ndim != 5 or image.shape[1] != self.in_channels or min(image.shape[2:]) < 8:
            raise ValueError(f'Expected [B,{self.in_channels},D,H,W] with each spatial dimension >= 8.')
        features = self.encoder(image)
        anatomy, disease = self.projection(features[-1])
        logits = self.decoder((*features[:-1], disease))
        return LatentSplitOutput(logits, anatomy, disease)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_with_latents(image).logits
