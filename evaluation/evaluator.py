"""Whole-volume baseline evaluation."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.label_mapping import REGIONS
from evaluation.metrics import EVALUATION_REGIONS, region_dice


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str | torch.device = "cpu",
             threshold: float = 0.5) -> dict[str, float]:
    """Compute subject-averaged Dice for each region, restoring model mode."""
    was_training = model.training
    model.eval()
    scores = []
    try:
        for batch in loader:
            logits = model(batch["image"].to(device))
            scores.append(region_dice(logits, batch["target"].to(device), threshold).cpu())
    finally:
        model.train(was_training)
    if not scores:
        raise ValueError("Evaluation loader is empty.")
    means = torch.cat(scores).mean(0)
    names = EVALUATION_REGIONS if len(means) == len(EVALUATION_REGIONS) else REGIONS
    return {region: means[i].item() for i, region in enumerate(names)}
