"""Standard volumetric Dice; no lesion-wise challenge metrics."""
import torch
from datasets.label_mapping import TISSUE_NAMES, derive_wt, derive_tc

EVALUATION_REGIONS = (*TISSUE_NAMES, "WT", "TC")


def _add_composites(tissues: torch.Tensor) -> torch.Tensor:
    """Append WT/TC unions to a batch of binary tissue masks."""
    wt = torch.stack([derive_wt(case) for case in tissues])
    tc = torch.stack([derive_tc(case) for case in tissues])
    return torch.cat([tissues, wt.unsqueeze(1), tc.unsqueeze(1)], dim=1).bool()


def region_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Compute per-subject volumetric Dice from logits and binary targets.

    Four-channel input order is NETC,SNFH,ET,RC; returns [B,6] ordered
    NETC,SNFH,ET,RC,WT,TC. Threshold tissues before deriving WT and TC.
    RC remains separate and is excluded from both unions.
    Legacy three-channel WT,TC,ET input returns [B,3] in that same order.
    Both-empty regions score 1; one-empty regions score 0.
    """
    if logits.shape != target.shape or logits.ndim != 5 or logits.shape[1] not in (3, len(TISSUE_NAMES)):
        raise ValueError("Expected matching [B,4,D,H,W] tissue or legacy [B,3,D,H,W] tensors.")
    if any(size == 0 for size in logits.shape):
        raise ValueError("Dice inputs must have nonempty batch and spatial dimensions.")
    if not torch.isfinite(logits).all():
        raise ValueError("Logits must be finite.")
    if not ((target == 0) | (target == 1)).all():
        raise ValueError("Targets must contain only binary values (0 or 1).")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one.")
    prediction, truth = logits.sigmoid() >= threshold, target.bool()
    if logits.shape[1] == len(TISSUE_NAMES):
        prediction, truth = _add_composites(prediction), _add_composites(truth)
    intersection = (prediction & truth).sum((2, 3, 4)).float()
    denominator = prediction.sum((2, 3, 4)) + truth.sum((2, 3, 4))
    return torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), 1.0)


VALIDATION_METRIC_FIELDS = tuple(
    f"{region}_{suffix}"
    for suffix in ("case_dice", "dataset_dice", "positive_cases")
    for region in EVALUATION_REGIONS
)


class ValidationDice:
    """Presence-aware case means and pooled voxel Dice for four tissue logits.

    Target-empty cases never enter case means, but their false positives DO
    enter pooled prediction counts. Undefined case means (no positive targets)
    and pooled Dice (no positive target cases) are None, logged blank.
    """

    def __init__(self):
        self.positive = torch.zeros(len(EVALUATION_REGIONS), dtype=torch.int64)
        self.case_sum = torch.zeros(len(EVALUATION_REGIONS), dtype=torch.float64)
        self.intersection = torch.zeros(len(EVALUATION_REGIONS), dtype=torch.int64)
        self.predicted = torch.zeros_like(self.intersection)
        self.target = torch.zeros_like(self.intersection)

    @torch.no_grad()
    def update(self, logits, target, threshold=0.5):
        if logits.ndim != 5 or logits.shape[1] != len(TISSUE_NAMES):
            raise ValueError('Validation requires four tissue channels.')
        # Reuse low-level input validation and documented per-case Dice.
        dice = region_dice(logits, target, threshold).double().cpu()
        prediction = _add_composites(logits.sigmoid() >= threshold)
        truth = _add_composites(target.bool())
        intersection = (prediction & truth).sum((2, 3, 4)).cpu()
        predicted = prediction.sum((2, 3, 4)).cpu()
        targets = truth.sum((2, 3, 4)).cpu()
        positive = targets > 0
        self.positive += positive.sum(0)
        self.case_sum += (dice * positive).sum(0)
        self.intersection += intersection.sum(0)
        self.predicted += predicted.sum(0)
        self.target += targets.sum(0)

    def compute(self):
        metrics = {}
        for index, region in enumerate(EVALUATION_REGIONS):
            count = int(self.positive[index])
            denominator = int(self.predicted[index] + self.target[index])
            metrics[f'{region}_positive_cases'] = count
            metrics[f'{region}_case_dice'] = float(self.case_sum[index]) / count if count else None
            metrics[f'{region}_dataset_dice'] = 2 * int(self.intersection[index]) / denominator if count else None
        return metrics


def best_validation_metric(metrics):
    """Equal region mean of target-positive case means; None if none eligible."""
    eligible = [metrics[f'{region}_case_dice'] for region in EVALUATION_REGIONS
                if metrics[f'{region}_positive_cases'] > 0]
    return sum(eligible) / len(eligible) if eligible else None
