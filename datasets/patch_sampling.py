"""Aligned spatial crops, with optional foreground-centered sampling."""
from collections.abc import Sequence
from numbers import Integral
import torch

DEFAULT_PATCH_SIZE = (96, 96, 96)


class PatchSampler:
    """Sample one crop per call using a CPU generator or PyTorch's global RNG.

    Foreground voxels are sampled uniformly from a binary spatial union.
    Center jitter is at most one quarter patch width per axis, bounded so the
    selected voxel always remains inside the crop. Empty foreground falls back
    to uniform random valid crop starts. No padding or resizing is performed.
    """

    def __init__(self, patch_size: Sequence[int] = DEFAULT_PATCH_SIZE,
                 foreground_sampling_probability: float = 0.7,
                 generator: torch.Generator | None = None) -> None:
        if len(patch_size) != 3 or any(
            not isinstance(size, Integral) or isinstance(size, bool) or size <= 0 for size in patch_size
        ):
            raise ValueError("patch_size must contain three positive integers.")
        if not 0 <= foreground_sampling_probability <= 1:
            raise ValueError("foreground_sampling_probability must be between 0 and 1.")
        self.patch_size = tuple(int(size) for size in patch_size)
        self.probability = foreground_sampling_probability
        self.generator = generator

    def _randint(self, low: int, high: int) -> int:
        return int(torch.randint(low, high, (), generator=self.generator))

    def sample_slices(self, foreground: torch.Tensor) -> tuple[slice, slice, slice]:
        """Return shared D/H/W slices for a three-dimensional foreground mask."""
        if foreground.ndim != 3:
            raise ValueError("Foreground must have shape [D,H,W].")
        if any(patch > volume for patch, volume in zip(self.patch_size, foreground.shape)):
            raise ValueError(f"Patch size {self.patch_size} exceeds volume shape {tuple(foreground.shape)}.")
        voxel = None
        if float(torch.rand((), generator=self.generator)) < self.probability:
            coordinates = foreground.nonzero()
            if len(coordinates):
                voxel = coordinates[self._randint(0, len(coordinates))].tolist()
        starts = []
        for axis, (patch, volume) in enumerate(zip(self.patch_size, foreground.shape)):
            if voxel is None:
                start = self._randint(0, volume - patch + 1)
            else:
                radius = min(patch // 4, (patch - 1) // 2)
                jitter = self._randint(-radius, radius + 1)
                start = max(0, min(voxel[axis] - patch // 2 + jitter, volume - patch))
            starts.append(start)
        return tuple(slice(start, start + size) for start, size in zip(starts, self.patch_size))

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Crop aligned [4,D,H,W] image and NETC/SNFH/ET/RC target tensors."""
        if image.ndim != 4 or image.shape != target.shape or image.shape[0] != 4:
            raise ValueError("Expected matching image and target shapes [4,D,H,W].")
        crop = (slice(None), *self.sample_slices(target.bool().any(0)))
        return image[crop].contiguous(), target[crop].contiguous()
