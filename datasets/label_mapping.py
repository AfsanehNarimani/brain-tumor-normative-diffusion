"""Release-dependent raw-label conversion; no BraTS mapping is assumed."""
from collections.abc import Mapping, Sequence
import torch

REGIONS = ("WT", "TC", "ET")

class LabelMapping:
    """Require explicit valid labels and nested region memberships."""

    def __init__(self, region_labels: Mapping[str, Sequence[int]], known_labels: Sequence[int]) -> None:
        if not known_labels or set(region_labels) != set(REGIONS):
            raise ValueError("Configure known_labels and WT/TC/ET for the exact dataset release.")
        if any(not region_labels[r] for r in REGIONS):
            raise ValueError("Each region requires explicit raw labels.")
        values = list(known_labels) + [v for r in REGIONS for v in region_labels[r]]
        if any(not isinstance(v, int) or isinstance(v, bool) for v in values):
            raise ValueError("Raw label IDs must be integers.")
        self.known_labels = set(known_labels)
        self.region_labels = {r: set(region_labels[r]) for r in REGIONS}
        if not self.region_labels["ET"] <= self.region_labels["TC"] <= self.region_labels["WT"] <= self.known_labels:
            raise ValueError("Require ET labels <= TC labels <= WT labels <= known_labels.")

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert [D,H,W] raw labels to float [3,D,H,W] region targets."""
        if mask.ndim != 3 or not torch.isfinite(mask).all() or not torch.equal(mask, mask.round()):
            raise ValueError("Mask must be finite, integer-valued, and three-dimensional.")
        unknown = set(mask.unique().tolist()) - self.known_labels
        if unknown:
            raise ValueError(f"Unknown segmentation labels: {sorted(unknown)}")
        return torch.stack([
            torch.stack([mask == label for label in sorted(self.region_labels[r])]).any(0)
            for r in REGIONS
        ]).float()
