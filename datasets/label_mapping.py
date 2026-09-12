"""BraTS 2024 post-treatment tissue labels and explicit composite regions."""
from collections.abc import Mapping, Sequence
import torch

BACKGROUND = 0
NETC = 1
SNFH = 2
ET = 3
RC = 4

TISSUE_LABELS = (NETC, SNFH, ET, RC)
TISSUE_NAMES = ("NETC", "SNFH", "ET", "RC")
KNOWN_LABELS = (BACKGROUND, *TISSUE_LABELS)

# Challenge manuscript, section 3: https://arxiv.org/html/2405.18368v1#S3
# RC is evaluated separately and excluded from both composites.
COMPOSITE_TISSUES = {"WT": (NETC, SNFH, ET), "TC": (NETC, ET)}


def segmentation_to_tissue_masks(mask: torch.Tensor) -> torch.Tensor:
    """Convert [D,H,W] to binary float32 [4,D,H,W]: NETC,SNFH,ET,RC.

    Accept integer or integer-valued floating tensors (e.g. NIfTI data).
    Preserve the input and its device/voxel order. Absent tissues yield zeros.
    """
    if mask.ndim != 3 or not torch.isfinite(mask).all() or not torch.equal(mask, mask.round()):
        raise ValueError("Mask must be finite, integer-valued, and three-dimensional.")
    unknown = set(mask.unique().tolist()) - set(KNOWN_LABELS)
    if unknown:
        raise ValueError(f"Unknown segmentation labels: {sorted(unknown)}; expected {KNOWN_LABELS}.")
    return torch.stack([mask == label for label in TISSUE_LABELS]).to(torch.float32)


def _validate_tissue_masks(tissues: torch.Tensor) -> None:
    if tissues.ndim != 4 or tissues.shape[0] != len(TISSUE_LABELS):
        raise ValueError("Tissue masks must have shape [4,D,H,W] in NETC,SNFH,ET,RC order.")
    if not ((tissues == 0) | (tissues == 1)).all():
        raise ValueError("Tissue masks must contain only binary values (0 or 1).")


def tissue_masks_to_regions(tissues: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return binary float32 [D,H,W] WT and TC from [4,D,H,W] tissues.

    WT = NETC | SNFH | ET; TC = NETC | ET. Neither includes RC.
    COMPOSITE_TISSUES holds explicit memberships for future release changes.
    Input must be binary masks, not unthresholded probabilities or logits.
    """
    _validate_tissue_masks(tissues)
    return {
        name: tissues[[TISSUE_LABELS.index(label) for label in labels]].bool().any(0).to(torch.float32)
        for name, labels in COMPOSITE_TISSUES.items()
    }


def derive_wt(tissues: torch.Tensor) -> torch.Tensor:
    """Return WT = NETC | SNFH | ET, excluding resection cavity."""
    return tissue_masks_to_regions(tissues)["WT"]


def derive_tc(tissues: torch.Tensor) -> torch.Tensor:
    """Return TC = NETC | ET, excluding resection cavity."""
    return tissue_masks_to_regions(tissues)["TC"]


def derive_et(tissues: torch.Tensor) -> torch.Tensor:
    """Return exactly the ET channel as binary float32 [D,H,W]."""
    _validate_tissue_masks(tissues)
    return tissues[TISSUE_LABELS.index(ET)].to(torch.float32)


def derive_rc(tissues: torch.Tensor) -> torch.Tensor:
    """Return exactly the separate RC channel as binary float32 [D,H,W]."""
    _validate_tissue_masks(tissues)
    return tissues[TISSUE_LABELS.index(RC)].to(torch.float32)


# Backward-compatible names for existing dataset consumers.
whole_tumor = derive_wt
tumor_core = derive_tc


REGIONS = ("WT", "TC", "ET")

class LabelMapping:
    """Legacy three-region adapter for unchanged baseline scripts.

    New dataset consumers default to segmentation_to_tissue_masks instead.
    Require explicit valid labels and nested region memberships.
    """

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
