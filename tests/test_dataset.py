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
