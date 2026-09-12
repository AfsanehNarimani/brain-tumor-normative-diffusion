"""Minimal baseline training loop."""
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: str | torch.device = "cpu") -> float:
    """Run one epoch and return subject-weighted mean loss."""
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        image, target = batch["image"].to(device), batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(image), target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        total += loss.item() * image.shape[0]
        count += image.shape[0]
    if not count:
        raise ValueError("Training loader is empty.")
    return total / count
