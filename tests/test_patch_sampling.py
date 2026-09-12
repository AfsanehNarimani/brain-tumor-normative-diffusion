"""Patch geometry, foreground selection, and reproducible sampling."""
import pytest
import torch
from datasets.patch_sampling import PatchSampler
from datasets.label_mapping import TISSUE_LABELS, NETC, SNFH, ET, RC


def generator(seed: int = 42) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("patch", [(3, 4, 5), (1, 1, 1), (8, 9, 10)])
@pytest.mark.parametrize("probability", [0., 1.])
def test_shape_and_alignment(patch: tuple[int, ...], probability: float) -> None:
    target = torch.zeros(4, 8, 9, 10)
    target[:, 2:6, 3:7, 4:8] = 1
    image = target + 10
    result, truth = PatchSampler(patch, probability, generator())(image, target)
    assert result.shape == truth.shape == (4, *patch)
    torch.testing.assert_close(result - 10, truth)


@pytest.mark.parametrize("label", [NETC, SNFH, ET, RC])
@pytest.mark.parametrize("point", [(0, 0, 0), (7, 8, 9), (4, 4, 5)])
def test_single_tissue_and_boundaries(label: int, point: tuple[int, ...]) -> None:
    target = torch.zeros(4, 8, 9, 10)
    target[(TISSUE_LABELS.index(label), *point)] = 1
    sampler = PatchSampler((3, 4, 5), 1., generator())
    for _ in range(10):
        image, patch = sampler(target, target)
        assert patch.shape == (4, 3, 4, 5)
        assert patch.sum() == 1
        torch.testing.assert_close(image, patch)


@pytest.mark.parametrize("probability", [0., 0.7, 1.])
def test_deterministic_sequence(probability: float) -> None:
    foreground = torch.ones(8, 9, 10, dtype=torch.bool)
    first = PatchSampler((3, 4, 5), probability, generator())
    second = PatchSampler((3, 4, 5), probability, generator())
    starts = []
    for _ in range(8):
        crop = first.sample_slices(foreground)
        assert crop == second.sample_slices(foreground)
        starts.append(tuple(s.start for s in crop))
    assert len(set(starts)) > 1


def test_random_ignores_foreground_and_empty_fallback() -> None:
    empty = torch.zeros(8, 9, 10, dtype=torch.bool)
    full = ~empty
    assert PatchSampler((3, 4, 5), 0., generator()).sample_slices(empty) == PatchSampler((3, 4, 5), 0., generator()).sample_slices(full)
    crop = PatchSampler((3, 4, 5), 1., generator()).sample_slices(empty)
    assert empty[crop].shape == (3, 4, 5)


@pytest.mark.parametrize("size", [(0, 2, 3), (2, 3), (2.5, 3, 4)])
def test_invalid_size(size: tuple) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        PatchSampler(size)


def test_oversized_patch() -> None:
    with pytest.raises(ValueError, match="exceeds volume"):
        PatchSampler().sample_slices(torch.zeros(8, 9, 10))
