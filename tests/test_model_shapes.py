"""Synthetic shape, optimization, metric, and checkpoint smoke tests."""
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader
from models.baseline_unet3d import BaselineUNet3D
from losses.segmentation_loss import SegmentationLoss
from evaluation.metrics import region_dice
from evaluation.evaluator import evaluate
from training.trainer import train_epoch
from training.checkpointing import save_checkpoint, load_checkpoint


@pytest.mark.parametrize("shape", [(1, 4, 8, 8, 8), (2, 4, 9, 11, 13)])
def test_output_shape_and_backward(shape: tuple[int, ...]) -> None:
    model = BaselineUNet3D(base_channels=2)
    logits = model(torch.randn(shape))
    assert logits.shape == (shape[0], 3, *shape[2:])
    target = torch.randint(0, 2, logits.shape).float()
    loss = SegmentationLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_invalid_input() -> None:
    with pytest.raises(ValueError):
        BaselineUNet3D(base_channels=2)(torch.randn(1, 3, 8, 8, 8))


def test_region_dice_empty_and_perfect() -> None:
    target = torch.zeros(2, 3, 4, 4, 4)
    target[0, :, :2] = 1
    logits = target * 40 - 20
    torch.testing.assert_close(region_dice(logits, target), torch.ones(2, 3))
    torch.testing.assert_close(region_dice(-logits, target), torch.zeros(2, 3))


def test_training_evaluation_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = BaselineUNet3D(base_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader([{"image": torch.randn(4, 8, 8, 8),
                          "target": torch.ones(3, 8, 8, 8)}])
    before = next(model.parameters()).detach().clone()
    loss = train_epoch(model, loader, optimizer, SegmentationLoss())
    assert loss > 0 and not torch.equal(before, next(model.parameters()))
    scores = evaluate(model, loader)
    assert set(scores) == {"WT", "TC", "ET"}
    assert all(0 <= score <= 1 for score in scores.values())
    assert model.training
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 1, {"model": {"base_channels": 2}})
    restored = BaselineUNet3D(base_channels=2)
    state = load_checkpoint(path, restored)
    assert state["epoch"] == 1
    image = torch.randn(1, 4, 8, 8, 8)
    torch.testing.assert_close(model(image), restored(image))
