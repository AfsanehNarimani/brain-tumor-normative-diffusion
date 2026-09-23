"""Compact latent diffusion network and support-masked diffusion utilities."""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dimension: int = 64) -> torch.Tensor:
    timesteps = torch.as_tensor(timesteps)
    if timesteps.ndim != 1 or dimension % 2:
        raise ValueError("timesteps must be [B] and embedding dimension must be even")
    half = dimension // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1))
    values = timesteps.float()[:, None] * frequencies[None, :]
    return torch.cat((values.sin(), values.cos()), dim=1)


class TimestepEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, 256))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.projection(sinusoidal_timestep_embedding(timesteps, 64))


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(256, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.residual = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_projection(timestep_embedding)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.residual(x)


class LatentDiffusionUNet3D(nn.Module):
    """The locked 256-channel, 12^3 latent diffusion U-Net."""

    def __init__(self) -> None:
        super().__init__()
        self.time_embedding = TimestepEmbedding()
        self.input_conv = nn.Conv3d(256, 64, 3, padding=1)
        self.level0 = nn.ModuleList([ResidualBlock3D(64, 64), ResidualBlock3D(64, 64)])
        self.down1 = nn.Conv3d(64, 128, 3, stride=2, padding=1)
        self.level1 = nn.ModuleList([ResidualBlock3D(128, 128), ResidualBlock3D(128, 128)])
        self.down2 = nn.Conv3d(128, 256, 3, stride=2, padding=1)
        self.level2 = nn.ModuleList([ResidualBlock3D(256, 256), ResidualBlock3D(256, 256)])
        self.middle = nn.ModuleList([ResidualBlock3D(256, 256), ResidualBlock3D(256, 256)])
        self.up1 = nn.Conv3d(256, 128, 3, padding=1)
        self.decoder1 = nn.ModuleList([ResidualBlock3D(256, 128), ResidualBlock3D(128, 128)])
        self.up2 = nn.Conv3d(128, 64, 3, padding=1)
        self.decoder0 = nn.ModuleList([ResidualBlock3D(128, 64), ResidualBlock3D(64, 64)])
        self.output_norm = nn.GroupNorm(32, 64)
        self.output_conv = nn.Conv3d(64, 256, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, z_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if z_t.ndim != 5 or tuple(z_t.shape[1:]) != (256, 12, 12, 12):
            raise ValueError("Expected latent input [B,256,12,12,12]")
        if timesteps.ndim != 1 or timesteps.shape[0] != z_t.shape[0]:
            raise ValueError("Expected one timestep per batch element")
        te = self.time_embedding(timesteps)
        x0 = self.input_conv(z_t)
        for block in self.level0:
            x0 = block(x0, te)
        x1 = self.down1(x0)
        for block in self.level1:
            x1 = block(x1, te)
        x2 = self.down2(x1)
        for block in self.level2:
            x2 = block(x2, te)
        for block in self.middle:
            x2 = block(x2, te)
        x = self.up1(F.interpolate(x2, scale_factor=2, mode="nearest"))
        x = torch.cat((x, x1), dim=1)
        for block in self.decoder1:
            x = block(x, te)
        x = self.up2(F.interpolate(x, scale_factor=2, mode="nearest"))
        x = torch.cat((x, x0), dim=1)
        for block in self.decoder0:
            x = block(x, te)
        return self.output_conv(F.silu(self.output_norm(x)))


def cosine_schedule(timesteps: int = 1000, offset: float = 0.008, max_beta: float = 0.999) -> dict[str, torch.Tensor]:
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    grid = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos(((grid / timesteps) + offset) / (1.0 + offset) * math.pi / 2.0) ** 2
    alpha_bar_full = f / f[0]
    beta = (1.0 - alpha_bar_full[1:] / alpha_bar_full[:-1]).clamp(max=max_beta)
    beta = beta.float()
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return {"beta": beta, "alpha": alpha, "alpha_bar": alpha_bar}


make_cosine_schedule = cosine_schedule


def q_sample(z0: torch.Tensor, timesteps: torch.Tensor, support_mask: torch.Tensor,
             schedule: dict[str, torch.Tensor], noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    if z0.ndim != 5 or support_mask.ndim != 5 or support_mask.shape[1] != 1 or z0.shape[0] != support_mask.shape[0] or z0.shape[2:] != support_mask.shape[2:]:
        raise ValueError("z0 and support_mask shapes are incompatible")
    if noise is None:
        noise = torch.randn_like(z0)
    mask = support_mask.to(dtype=z0.dtype)
    epsilon = noise * mask
    ab = schedule["alpha_bar"].to(device=z0.device, dtype=z0.dtype)[timesteps.long()]
    ab = ab[:, None, None, None, None]
    return (torch.sqrt(ab) * z0 + torch.sqrt(1.0 - ab) * epsilon) * mask


def masked_epsilon_mse(prediction: torch.Tensor, target: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 5 or support_mask.shape != (prediction.shape[0], 1, *prediction.shape[2:]):
        raise ValueError("prediction, target, and support_mask shapes are incompatible")
    mask = support_mask.to(dtype=prediction.dtype)
    denominator = mask.sum() * prediction.shape[1]
    if denominator.item() <= 0:
        raise ValueError("Cannot compute masked loss with zero valid support")
    return (((prediction - target) ** 2) * mask).sum() / denominator
