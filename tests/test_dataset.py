"""Temporary synthetic NIfTI files exercise the actual dataset loader."""
import csv
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
import torch
from datasets.brats_dataset import BraTSDataset, MODALITIES
from datasets.label_mapping import LabelMapping
from datasets.transforms import normalize_nonzero


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> BraTSDataset:
    shape = (8, 9, 10)
    row = {"subject_id": "synthetic"}
    for i, key in enumerate((*MODALITIES, "mask")):
        array = np.full(shape, 10 if key == "mask" else i + 1, dtype=np.float32)
        path = tmp_path / f"{key}.nii.gz"
        nib.save(nib.Nifti1Image(array, np.eye(4)), path)
        row[key] = path.name
    manifest = tmp_path / "subjects.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return BraTSDataset(manifest, tmp_path,
                        LabelMapping({"WT": [10], "TC": [10], "ET": [10]}, [0, 10]), normalize=False)


def test_loading_order_and_regions(synthetic_dataset: BraTSDataset) -> None:
    sample = synthetic_dataset[0]
    assert sample["subject_id"] == "synthetic"
    assert sample["image"].shape == (4, 8, 9, 10)
    assert sample["target"].shape == (3, 8, 9, 10)
    torch.testing.assert_close(sample["image"][:, 0, 0, 0], torch.arange(1, 5).float())
    assert sample["target"].eq(1).all()


@pytest.mark.parametrize("mismatch", ["shape", "affine"])
def test_grid_mismatch(synthetic_dataset: BraTSDataset, mismatch: str) -> None:
    path = synthetic_dataset.records[0]["paths"]["T2f"]
    affine = np.eye(4)
    shape = (7, 9, 10) if mismatch == "shape" else (8, 9, 10)
    if mismatch == "affine":
        affine[0, 3] = 5
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), path)
    with pytest.raises(ValueError):
        synthetic_dataset[0]


def test_normalization() -> None:
    image = torch.zeros(4, 4, 4, 4)
    image[:, 1:] = torch.arange(48).reshape(3, 4, 4).float() + 1
    result = normalize_nonzero(image)
    assert result[:, 0].eq(0).all()
    torch.testing.assert_close(result[0, 1:].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(result[0, 1:].std(unbiased=False), torch.tensor(1.0))
    assert torch.isfinite(normalize_nonzero(torch.ones_like(image))).all()
    assert normalize_nonzero(torch.zeros_like(image)).eq(0).all()


def test_default_tissue_targets_and_raw_labels(synthetic_dataset: BraTSDataset) -> None:
    from datasets.label_mapping import KNOWN_LABELS, segmentation_to_tissue_masks

    record = synthetic_dataset.records[0]
    raw = np.resize(np.asarray(KNOWN_LABELS, dtype=np.int16), (8, 9, 10))
    nib.save(nib.Nifti1Image(raw, np.eye(4)), record["paths"]["mask"])
    synthetic_dataset = BraTSDataset(
        record["paths"]["mask"].parent / "subjects.csv",
        record["paths"]["mask"].parent, normalize=False,
    )
    sample = synthetic_dataset[0]
    assert sample["target"].shape == (4, 8, 9, 10)
    assert sample["target"].dtype == torch.float32
    assert sample["raw_mask"].dtype == torch.int64
    torch.testing.assert_close(sample["raw_mask"], torch.from_numpy(raw).long())
    torch.testing.assert_close(sample["target"], segmentation_to_tissue_masks(sample["raw_mask"]))


@pytest.fixture
def case_root(tmp_path: Path) -> Path:
    from datasets.label_mapping import KNOWN_LABELS

    for case_id in ("case-b", "case-a"):
        directory = tmp_path / case_id
        directory.mkdir()
        base = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        for index, modality in enumerate(MODALITIES):
            array = base * (index + 1)
            nib.save(nib.Nifti1Image(array, np.eye(4)), directory / f"{case_id}-{modality.lower()}.nii.gz")
        seg = np.resize(np.asarray(KNOWN_LABELS, dtype=np.int16), base.shape)
        nib.save(nib.Nifti1Image(seg, np.eye(4)), directory / f"{case_id}-seg.nii.gz")
    return tmp_path


def test_discovery_order_conversion_and_normalization(case_root: Path) -> None:
    from datasets.label_mapping import segmentation_to_tissue_masks

    dataset = BraTSDataset(root=case_root, normalize=False)
    assert len(dataset) == 2
    assert [record["subject_id"] for record in dataset.records] == ["case-a", "case-b"]
    sample = dataset[0]
    assert sample["case_id"] == "case-a"
    assert sample["spacing"] == (1., 1., 1.)
    assert sample["image"].shape == sample["target"].shape == (4, 2, 3, 4)
    assert sample["image"].dtype == sample["target"].dtype == torch.float32
    torch.testing.assert_close(sample["image"][:, 0, 0, 1], torch.arange(1, 5).float())
    torch.testing.assert_close(sample["target"], segmentation_to_tissue_masks(sample["raw_mask"]))
    normalized = BraTSDataset(case_root)[0]
    torch.testing.assert_close(normalized["target"], sample["target"])
    torch.testing.assert_close(normalized["raw_mask"], sample["raw_mask"])
    for raw, channel in zip(sample["image"], normalized["image"]):
        assert channel[raw == 0].eq(0).all()
        torch.testing.assert_close(channel[raw != 0].mean(), torch.tensor(0.), atol=1e-6, rtol=0)
        torch.testing.assert_close(channel[raw != 0].std(unbiased=False), torch.tensor(1.))


@pytest.mark.parametrize("suffix", ["t1n", "t1c", "t2w", "t2f", "seg"])
def test_missing_case_file(case_root: Path, suffix: str) -> None:
    (case_root / "case-a" / f"case-a-{suffix}.nii.gz").unlink()
    with pytest.raises(FileNotFoundError, match=f"Case case-a: missing required .*case-a-{suffix}"):
        BraTSDataset(root=case_root)


@pytest.mark.parametrize("suffix", ["t1c", "seg"])
@pytest.mark.parametrize("mismatch", ["shape", "spacing", "affine/grid"])
def test_discovered_geometry(case_root: Path, suffix: str, mismatch: str) -> None:
    affine = np.eye(4)
    shape = (2, 3, 5) if mismatch == "shape" else (2, 3, 4)
    if mismatch == "spacing":
        affine[0, 0] = 2
    if mismatch == "affine/grid":
        affine[0, 3] = 2
    nib.save(nib.Nifti1Image(np.zeros(shape), affine), case_root / "case-a" / f"case-a-{suffix}.nii.gz")
    with pytest.raises(ValueError, match=f"Case case-a.*{mismatch} mismatch"):
        BraTSDataset(root=case_root)[0]


def test_constant_foreground_and_empty_modality() -> None:
    image = torch.zeros(4, 2, 3, 4)
    image[:3, :, :, 1:] = 7
    result = normalize_nonzero(image)
    assert torch.isfinite(result).all()
    assert result.eq(0).all()
    assert image[:3, :, :, 1:].eq(7).all()


def test_empty_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No case directories"):
        BraTSDataset(root=tmp_path)


def test_training_patch_and_full_validation(case_root: Path) -> None:
    full = BraTSDataset(root=case_root)[0]
    dataset = BraTSDataset(root=case_root, training=True, patch_size=(1, 2, 3),
                          generator=torch.Generator().manual_seed(42))
    patch = dataset[0]
    crop = tuple(slice(start, start + size) for start, size in zip(patch["crop_start"], (1, 2, 3)))
    for key in ("image", "target"):
        assert patch[key].shape == (4, 1, 2, 3)
        torch.testing.assert_close(patch[key], full[key][(slice(None), *crop)])
    torch.testing.assert_close(patch["raw_mask"], full["raw_mask"][crop])
    validation = BraTSDataset(root=case_root, training=False, patch_size=(1, 2, 3))[0]
    assert validation["image"].shape == (4, 2, 3, 4)
    assert "crop_start" not in validation


def test_augmentation_training_only(case_root):
    config = {'enabled': True, 'flip': {'probability': 1},
              'intensity_scale': {'enabled': False}, 'intensity_shift': {'enabled': False},
              'gaussian_noise': {'enabled': False}}
    baseline = BraTSDataset(root=case_root)[0]
    validation = BraTSDataset(root=case_root, augmentation=config)[0]
    torch.testing.assert_close(validation['image'], baseline['image'])
    torch.testing.assert_close(validation['target'], baseline['target'])
    train = BraTSDataset(root=case_root, training=True, patch_size=(2, 3, 4), augmentation=config)[0]
    torch.testing.assert_close(train['image'], baseline['image'].flip((1, 2, 3)))
    torch.testing.assert_close(train['target'], baseline['target'].flip((1, 2, 3)))
    torch.testing.assert_close(train['raw_mask'], baseline['raw_mask'].flip((0, 1, 2)))
