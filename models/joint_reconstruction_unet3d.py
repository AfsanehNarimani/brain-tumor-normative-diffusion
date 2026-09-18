"""E1-B joint MRI reconstruction alongside unchanged E1-A segmentation."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.encoder import conv_block
from models.latent_split_unet3d import LatentSplitUNet3D


@dataclass(frozen=True)
class JointReconstructionOutput:
    segmentation_logits: torch.Tensor
    reconstruction: torch.Tensor
    z_a: torch.Tensor
    z_d: torch.Tensor


class ReconstructionDecoder3D(nn.Module):
    """Decode concatenated latents without encoder skips or output activation."""

    def __init__(self, latent_channels: int, base_channels: int, out_channels: int) -> None:
        super().__init__()
        widths = (4 * base_channels, 2 * base_channels, base_channels)
        sources = (2 * latent_channels, *widths[:-1])
        self.blocks = nn.ModuleList(conv_block(source, width)
                                    for source, width in zip(sources, widths))
        self.head = nn.Conv3d(widths[-1], out_channels, 1)

    def forward(self, latents: torch.Tensor, output_size: tuple[int, int, int]) -> torch.Tensor:
        image = latents
        for block, divisor in zip(self.blocks, (4, 2, 1)):
            size = tuple(max(1, (dimension + divisor - 1) // divisor) for dimension in output_size)
            image = F.interpolate(image, size=size, mode='trilinear', align_corners=False)
            image = block(image)
        return self.head(image)


class JointReconstructionUNet3D(LatentSplitUNet3D):
    """Reuse E1-A encoder, projections, skips, and segmentation decoder exactly."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 32,
                 latent_projection: str = 'conv1x1x1', latent_channels: int | None = None) -> None:
        super().__init__(in_channels, out_channels, base_channels, latent_projection, latent_channels)
        self.reconstruction_decoder = ReconstructionDecoder3D(self.latent_channels, base_channels, in_channels)

    def forward(self, image: torch.Tensor) -> JointReconstructionOutput:
        segmented = super().forward_with_latents(image)
        latents = torch.cat((segmented.anatomy_latent, segmented.disease_latent), dim=1)
        reconstruction = self.reconstruction_decoder(latents, tuple(image.shape[2:]))
        return JointReconstructionOutput(segmented.logits, reconstruction,
                                         segmented.anatomy_latent, segmented.disease_latent)

    def segmentation_only(self, image: torch.Tensor) -> torch.Tensor:
        """Use the E1-A path for unchanged tiled validation, without reconstruction."""
        return super().forward(image)
