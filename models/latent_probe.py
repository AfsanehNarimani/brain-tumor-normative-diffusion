"""Frozen bottleneck access probes; neither latent has a health interpretation."""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models.encoder import conv_block
from models.latent_split_unet3d import LatentSplitUNet3D
from models.joint_reconstruction_unet3d import JointReconstructionUNet3D


SOURCE_SHA256 = "f8f3fea793f96a2e05eacfc16c1ba09ab2fabd5588c9e93d02cc3eb28b6c61b8"
SOURCE_MODEL = dict(name="latent_split_unet3d", in_channels=4, out_channels=4,
                    base_channels=32, latent_projection="conv1x1x1", latent_channels=256)
E1B_SOURCE_SHA256 = "142166a90ff3fb652fbc776ad98e1b33fe56988e8796ef87334ab91d86228606"
E1B_SOURCE_MODEL = dict(name="joint_reconstruction_unet3d", in_channels=4, out_channels=4,
                        base_channels=32, latent_projection="conv1x1x1", latent_channels=256)
SOURCE_SPECS = {
    "e1a": dict(label="E1-A", sha256=SOURCE_SHA256, epoch=15,
                model=SOURCE_MODEL, constructor=LatentSplitUNet3D),
    "e1b": dict(label="E1-B", sha256=E1B_SOURCE_SHA256, epoch=13,
                model=E1B_SOURCE_MODEL, constructor=JointReconstructionUNet3D),
}
PROBE_ARCHITECTURE = dict(name="latent_probe_decoder3d", in_channels=256,
                          widths=[128, 64, 32], out_channels=4,
                          interpolation="trilinear", convolution="two Conv3d+GroupNorm+ReLU per block")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(path: str | Path, *, source_experiment: str = "e1a",
                expected_sha256: str | None = None,
                expected_epoch: int | None = None) -> tuple[LatentSplitUNet3D, dict]:
    """Check identity and architecture before using a supported frozen source."""
    if source_experiment not in SOURCE_SPECS:
        raise ValueError(f"Unsupported frozen source experiment: {source_experiment}")
    spec = SOURCE_SPECS[source_experiment]
    expected_sha256 = spec["sha256"] if expected_sha256 is None else expected_sha256
    expected_epoch = spec["epoch"] if expected_epoch is None else expected_epoch
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen {spec['label']} source checkpoint missing: {path}")
    sha256 = file_sha256(path)
    if sha256 != expected_sha256:
        raise ValueError(f"Frozen {spec['label']} source SHA256 mismatch: {sha256} != {expected_sha256}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("epoch") != expected_epoch:
        raise ValueError(f"Frozen {spec['label']} source epoch must be {expected_epoch}; got {state.get('epoch')}")
    config = state.get("config")
    if not isinstance(config, dict) or config.get("model") != spec["model"]:
        raise ValueError(f"Frozen {spec['label']} source model architecture/config mismatch")
    model = spec["constructor"](**{k: v for k, v in spec["model"].items() if k != "name"})
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model, dict(path=str(path), sha256=sha256, epoch=expected_epoch,
                       experiment=source_experiment, model=spec["model"]["name"],
                       config=config, parameter_count=sum(p.numel() for p in model.parameters()))


class LatentProbeDecoder3D(nn.Module):
    """Three interpolation and convolution stages without skips."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([conv_block(256, 128), conv_block(128, 64),
                                     conv_block(64, 32)])
        self.head = nn.Conv3d(32, 4, 1)

    def forward(self, latent: torch.Tensor, output_size: tuple[int, int, int]) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != 256 or len(output_size) != 3:
            raise ValueError("Expected [B,256,d,h,w] latent and 3D output size")
        # The intermediate sizes are derived from the requested image shape.
        for block, divisor in zip(self.blocks, (4, 2, 1)):
            target = tuple(max(1, (size + divisor - 1) // divisor) for size in output_size)
            latent = block(F.interpolate(latent, size=target, mode="trilinear", align_corners=False))
        return self.head(latent)


class FrozenLatentProbe(nn.Module):
    def __init__(self, source: LatentSplitUNet3D, latent_source: str,
                 probe: LatentProbeDecoder3D | None = None) -> None:
        super().__init__()
        if latent_source not in ("anatomy", "disease"):
            raise ValueError("latent_source must be anatomy or disease")
        self.source = source.requires_grad_(False)
        self.source.eval()
        self.latent_source = latent_source
        self.probe = probe if probe is not None else LatentProbeDecoder3D()

    def train(self, mode: bool = True):
        super().train(mode)
        self.source.eval()
        return self

    def extract_latent(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != self.source.in_channels or min(image.shape[2:]) < 8:
            raise ValueError("Expected [B,4,D,H,W] with each spatial dimension >= 8")
        # Do not call forward_with_latents: that also runs the original decoder.
        with torch.no_grad():
            bottleneck = self.source.encoder(image)[-1]
            projector = getattr(self.source.projection, f"{self.latent_source}_projector")
            latent = projector(bottleneck)
        return latent.detach()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.probe(self.extract_latent(image), tuple(image.shape[2:]))
