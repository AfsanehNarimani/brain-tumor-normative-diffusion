"""Save and load baseline state dictionaries and run configuration."""
from pathlib import Path
from typing import Any
import torch
from torch import nn


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, config: dict[str, Any], best_metric: float | None = None,
                    seed: int | None = None, best_epoch: int | None = None) -> None:
    """Save model, optimizer, completed epoch, and configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "config": config, "best_metric": best_metric, "seed": seed,
                "best_epoch": best_epoch}, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    """Restore state on CPU; caller controls the model device."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state
