"""Synthetic label IDs here are deliberately not a BraTS release mapping."""
import pytest
import torch
from datasets.label_mapping import LabelMapping


def test_region_memberships() -> None:
    mapping = LabelMapping({"WT": [10, 20, 30], "TC": [20, 30], "ET": [30]}, [0, 10, 20, 30])
    result = mapping(torch.tensor([0, 10, 20, 30]).reshape(1, 1, 4))
    expected = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]]).float()
    torch.testing.assert_close(result[:, 0, 0], expected)


@pytest.mark.parametrize("mask", [torch.tensor([99.0]), torch.tensor([10.5]), torch.tensor([float("nan")])])
def test_invalid_labels(mask: torch.Tensor) -> None:
    mapping = LabelMapping({"WT": [10], "TC": [10], "ET": [10]}, [0, 10])
    with pytest.raises(ValueError):
        mapping(mask.reshape(1, 1, 1))


def test_mapping_must_be_explicit() -> None:
    with pytest.raises(ValueError):
        LabelMapping({"WT": None, "TC": None, "ET": None}, [0, 10])


def test_region_hierarchy() -> None:
    with pytest.raises(ValueError):
        LabelMapping({"WT": [10], "TC": [20], "ET": [20]}, [0, 10, 20])
