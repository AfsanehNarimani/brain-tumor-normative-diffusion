"""Manifest-driven loading of aligned MRI volumes and raw segmentation masks."""
import csv
from pathlib import Path
from typing import Any
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.label_mapping import LabelMapping
from datasets.transforms import normalize_nonzero

MODALITIES = ("T1n", "T1c", "T2w", "T2f")


def read_manifest(manifest: str | Path, root: str | Path) -> list[dict[str, Any]]:
    """Read CSV columns subject_id,T1n,T1c,T2w,T2f,mask."""
    records = []
    seen = set()
    with Path(manifest).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"subject_id", *MODALITIES, "mask"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {sorted(required)}")
        for row in reader:
            if any(not row.get(key) for key in required):
                raise ValueError("Manifest contains an empty required field.")
            if row["subject_id"] in seen:
                raise ValueError(f"Duplicate subject ID: {row['subject_id']}")
            seen.add(row["subject_id"])
            records.append({"subject_id": row["subject_id"], "paths": {
                key: Path(root) / row[key] for key in (*MODALITIES, "mask")
            }})
    if not records:
        raise ValueError("Manifest contains no subjects.")
    return records


def load_subject(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Load finite 3D volumes on the same voxel grid, without resampling."""
    volumes = [nib.load(record["paths"][key]) for key in (*MODALITIES, "mask")]
    reference = volumes[0]
    arrays = []
    for volume in volumes:
        if len(volume.shape) != 3 or volume.shape != reference.shape:
            raise ValueError("All modalities and the mask must share a 3D shape.")
        if not np.allclose(volume.affine, reference.affine, atol=1e-5, rtol=0):
            raise ValueError("All modalities and the mask must share an affine/grid.")
        array = volume.get_fdata(dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Volumes must contain only finite values.")
        arrays.append(torch.from_numpy(array))
    return torch.stack(arrays[:4]), arrays[4]


class BraTSDataset(Dataset):
    """Return subject_id, image [4,D,H,W], and target [3,D,H,W]."""

    def __init__(self, manifest: str | Path, root: str | Path,
                 label_mapping: LabelMapping, normalize: bool = True) -> None:
        self.records = read_manifest(manifest, root)
        self.label_mapping = label_mapping
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image, mask = load_subject(record)
        return {"subject_id": record["subject_id"],
                "image": normalize_nonzero(image) if self.normalize else image,
                "target": self.label_mapping(mask)}
