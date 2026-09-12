"""Conservative training augmentation: flip, scale, shift, then noise."""
import math
import torch


class TrainingAugmentation:
    """CPU augmentation using a supplied generator or the seeded global RNG.

    Intensity operations use a fixed mask of nonzero normalized input voxels.
    This preserves zeros, including foreground voxels normalized exactly to zero.
    Each modality receives independent scale, shift, and noise standard deviation.
    """

    def __init__(self, config, generator=None):
        self.config = config
        self.generator = generator
        self.enabled = config.get('enabled', False)
        self.flip = config.get('flip', {})
        self.probability = self.flip.get('probability', 0.5)
        if not 0 <= self.probability <= 1:
            raise ValueError('Flip probability must be between 0 and 1.')
        self.ranges = {}
        for name, keys, defaults in (
            ('intensity_scale', ('min', 'max'), (0.9, 1.1)),
            ('intensity_shift', ('min', 'max'), (-0.1, 0.1)),
            ('gaussian_noise', ('std_min', 'std_max'), (0.0, 0.05)),
        ):
            section = config.get(name, {})
            low, high = (section.get(key, default) for key, default in zip(keys, defaults))
            if not all(math.isfinite(v) for v in (low, high)) or low > high:
                raise ValueError(f'{name}: require finite ordered bounds.')
            if name != 'intensity_shift' and low < 0:
                raise ValueError(f'{name}: bounds must be nonnegative.')
            self.ranges[name] = (low, high)

    def _sample(self, name):
        low, high = self.ranges[name]
        return low + (high - low) * torch.rand((), generator=self.generator).item()

    def __call__(self, image, target, raw_mask=None):
        """Return image/target, plus aligned raw_mask when supplied."""
        if image.shape != target.shape or image.ndim != 4 or image.shape[0] != 4:
            raise ValueError('Expected matching [4,D,H,W] image and target.')
        if raw_mask is not None and raw_mask.shape != image.shape[1:]:
            raise ValueError('Raw mask must share spatial dimensions.')
        if not self.enabled:
            return (image, target, raw_mask) if raw_mask is not None else (image, target)
        image, target = image.clone().float(), target.clone().float()
        axes = []
        if self.flip.get('enabled', True):
            axes = [axis for axis in range(3) if torch.rand((), generator=self.generator).item() < self.probability]
        if axes:
            image = image.flip([axis + 1 for axis in axes])
            target = target.flip([axis + 1 for axis in axes])
            if raw_mask is not None:
                raw_mask = raw_mask.flip(axes)
        for channel in image:
            foreground = channel != 0
            values = channel[foreground]
            if self.config.get('intensity_scale', {}).get('enabled', True):
                values = values * self._sample('intensity_scale')
            if self.config.get('intensity_shift', {}).get('enabled', True):
                values = values + self._sample('intensity_shift')
            if self.config.get('gaussian_noise', {}).get('enabled', True):
                std = self._sample('gaussian_noise')
                values = values + torch.randn(values.shape, generator=self.generator, dtype=values.dtype) * std
            channel[foreground] = values
        return (image, target, raw_mask) if raw_mask is not None else (image, target)
