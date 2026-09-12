"""BraTS 2024 tissue mapping and legacy region-adapter regression tests."""
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


from datasets.label_mapping import (
    BACKGROUND, NETC, SNFH, ET, RC, KNOWN_LABELS, TISSUE_LABELS, TISSUE_NAMES,
    segmentation_to_tissue_masks, tissue_masks_to_regions, whole_tumor, tumor_core,
)


def test_brats_constants_and_channel_order() -> None:
    assert (BACKGROUND, NETC, SNFH, ET, RC) == (0, 1, 2, 3, 4)
    assert TISSUE_NAMES == ("NETC", "SNFH", "ET", "RC")
    raw = torch.tensor(KNOWN_LABELS).reshape(1, 1, -1)
    original = raw.clone()
    masks = segmentation_to_tissue_masks(raw)
    assert masks.shape == (4, 1, 1, 5)
    assert masks.dtype == torch.float32
    torch.testing.assert_close(masks[:, 0, 0], torch.cat([torch.zeros(4, 1), torch.eye(4)], dim=1))
    torch.testing.assert_close(raw, original)


@pytest.mark.parametrize("absent", TISSUE_LABELS)
def test_absent_tissue(absent: int) -> None:
    raw = torch.tensor([label for label in KNOWN_LABELS if label != absent]).reshape(1, 2, 2)
    masks = segmentation_to_tissue_masks(raw)
    assert masks.shape == (4, 1, 2, 2)
    assert not masks[TISSUE_LABELS.index(absent)].any()
    for channel, label in enumerate(TISSUE_LABELS):
        torch.testing.assert_close(masks[channel], (raw == label).float())


def test_background_only() -> None:
    assert not segmentation_to_tissue_masks(torch.full((2, 3, 4), BACKGROUND)).any()


@pytest.mark.parametrize("value,message", [
    (-1, "Unknown segmentation labels"), (5, "Unknown segmentation labels"),
    (1.5, "integer-valued"), (float("nan"), "finite"), (float("inf"), "finite"),
])
def test_invalid_brats_labels(value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        segmentation_to_tissue_masks(torch.tensor([value]).reshape(1, 1, 1))


def test_invalid_raw_shape() -> None:
    with pytest.raises(ValueError, match="three-dimensional"):
        segmentation_to_tissue_masks(torch.zeros(2, 3))


def test_composites_exclude_cavity_and_background() -> None:
    tissues = segmentation_to_tissue_masks(torch.tensor(KNOWN_LABELS).float().reshape(1, 1, -1))
    regions = tissue_masks_to_regions(tissues)
    torch.testing.assert_close(regions["WT"], torch.tensor([0, 1, 1, 1, 0]).float().reshape(1, 1, -1))
    torch.testing.assert_close(regions["TC"], torch.tensor([0, 1, 0, 1, 0]).float().reshape(1, 1, -1))
    torch.testing.assert_close(whole_tumor(tissues), regions["WT"])
    torch.testing.assert_close(tumor_core(tissues), regions["TC"])


@pytest.mark.parametrize("tissues", [torch.zeros(3, 1, 1, 1), torch.full((4, 1, 1, 1), 0.5)])
def test_invalid_tissue_masks(tissues: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="Tissue masks"):
        tissue_masks_to_regions(tissues)


@pytest.mark.parametrize("absent", [None, NETC, ET])
def test_explicit_derived_regions(absent: int | None) -> None:
    from datasets.label_mapping import derive_wt, derive_tc, derive_et, derive_rc

    labels = [label for label in KNOWN_LABELS if label != absent]
    raw = torch.tensor(labels).reshape(1, 1, -1)
    tissues = segmentation_to_tissue_masks(raw)
    netc, snfh, et, rc = tissues.bool()
    expected = (netc | snfh | et, netc | et, et, rc)
    for helper, truth in zip((derive_wt, derive_tc, derive_et, derive_rc), expected):
        result = helper(tissues)
        torch.testing.assert_close(result, truth.float())
        assert ((result == 0) | (result == 1)).all()
    assert not derive_wt(tissues)[rc].any()
    assert not derive_tc(tissues)[rc].any()
