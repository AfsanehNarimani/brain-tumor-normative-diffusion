"""E1-C: E1-B plus Za-only non-lesional MRI reconstruction."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.encoder import conv_block
from models.joint_reconstruction_unet3d import JointReconstructionUNet3D


@dataclass(frozen=True)
class AnatomyConstrainedOutput:
    segmentation_logits: torch.Tensor
    reconstruction: torch.Tensor
    anatomy_reconstruction: torch.Tensor
    z_a: torch.Tensor
    z_d: torch.Tensor


class AnatomyReconstructionDecoder3D(nn.Module):
    """Decode the anatomy latent without encoder skips or output activation."""

    def __init__(self, latent_channels: int, base_channels: int, out_channels: int) -> None:
        super().__init__()
        widths = (4 * base_channels, 2 * base_channels, base_channels)
        sources = (latent_channels, *widths[:-1])
        self.blocks = nn.ModuleList(conv_block(source, width)
                                    for source, width in zip(sources, widths))
        self.head = nn.Conv3d(widths[-1], out_channels, 1)

    def forward(self, latent: torch.Tensor, output_size: tuple[int, int, int]) -> torch.Tensor:
        image = latent
        for block, divisor in zip(self.blocks, (4, 2, 1)):
            size = tuple(max(1, (dimension + divisor - 1) // divisor)
                         for dimension in output_size)
            image = block(F.interpolate(image, size=size, mode='trilinear', align_corners=False))
        return self.head(image)


class AnatomyConstrainedJointReconstructionUNet3D(JointReconstructionUNet3D):
    """Preserve E1-B and add anatomy/background reconstruction from Za only."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 32,
                 latent_projection: str = 'conv1x1x1', latent_channels: int | None = None) -> None:
        super().__init__(in_channels, out_channels, base_channels,
                         latent_projection, latent_channels)
        self.anatomy_decoder = AnatomyReconstructionDecoder3D(
            self.latent_channels, base_channels, in_channels)

    def forward(self, image: torch.Tensor) -> AnatomyConstrainedOutput:
        joint = super().forward(image)
        anatomy = self.anatomy_decoder(joint.z_a, tuple(image.shape[2:]))
        return AnatomyConstrainedOutput(joint.segmentation_logits, joint.reconstruction,
                                        anatomy, joint.z_a, joint.z_d)
