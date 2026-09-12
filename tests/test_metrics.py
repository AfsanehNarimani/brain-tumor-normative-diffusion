"""Volumetric tissue/composite Dice, including separate cavity evaluation."""
import pytest
import torch
from datasets.label_mapping import (
    BACKGROUND, NETC, SNFH, ET, RC, segmentation_to_tissue_masks,
)
from evaluation.metrics import EVALUATION_REGIONS, region_dice
from evaluation.evaluator import evaluate


def masks(labels: list[int]) -> torch.Tensor:
    return segmentation_to_tissue_masks(torch.tensor(labels).reshape(1, 1, -1)).unsqueeze(0)


@pytest.mark.parametrize("labels", [
    [BACKGROUND, NETC, SNFH, ET, RC], [BACKGROUND, SNFH, RC], [BACKGROUND],
])
def test_perfect_and_empty_dice(labels: list[int]) -> None:
    target = masks(labels)
    assert EVALUATION_REGIONS == ("NETC", "SNFH", "ET", "RC", "WT", "TC")
    torch.testing.assert_close(region_dice(target * 40 - 20, target), torch.ones(1, 6))


def test_cavity_does_not_contribute_to_composites() -> None:
    target = masks([NETC, SNFH, ET, RC])
    prediction = masks([NETC, SNFH, ET, BACKGROUND])
    torch.testing.assert_close(
        region_dice(prediction * 40 - 20, target),
        torch.tensor([[1., 1., 1., 0., 1., 1.]]),
    )


def test_partial_overlap_and_subject_averaging() -> None:
    target = masks([NETC, SNFH, ET, RC])
    prediction = masks([NETC, BACKGROUND, NETC, RC])
    logits = prediction * 40 - 20
    expected = torch.tensor([[2 / 3, 0., 0., 1., 4 / 5, 1.]])
    torch.testing.assert_close(region_dice(logits, target), expected)
    batch_logits = torch.cat([logits, target * 40 - 20])
    batch_target = target.repeat(2, 1, 1, 1, 1)
    model = torch.nn.Identity()
    scores = evaluate(model, [{"image": batch_logits, "target": batch_target}])
    assert tuple(scores) == EVALUATION_REGIONS
    torch.testing.assert_close(torch.tensor(list(scores.values())), (expected[0] + 1) / 2)
    assert model.training


@pytest.mark.parametrize("threshold", [0., 1.])
def test_invalid_threshold(threshold: float) -> None:
    target = masks([BACKGROUND])
    with pytest.raises(ValueError, match="Threshold"):
        region_dice(target, target, threshold)


def test_nonbinary_target_rejected() -> None:
    target = masks([BACKGROUND])
    with pytest.raises(ValueError, match="binary"):
        region_dice(target, target + 0.5)
