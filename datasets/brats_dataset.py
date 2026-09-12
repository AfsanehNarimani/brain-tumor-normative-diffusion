"""Directory- or manifest-driven loading of aligned MRI volumes and raw segmentation masks."""
import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.label_mapping import segmentation_to_tissue_masks
from datasets.transforms import normalize_nonzero
from datasets.augmentations import TrainingAugmentation
from datasets.patch_sampling import DEFAULT_PATCH_SIZE, PatchSampler
from datasets.label_mapping import TISSUE_LABELS

MODALITIES = ("T1n", "T1c", "T2w", "T2f")


def _verify_files(record: dict[str, Any]) -> None:
    for key, path in record["paths"].items():
        if not path.is_file():
            raise FileNotFoundError(f"Case {record['subject_id']}: missing required {key} file: {path}")


def discover_cases(root: str | Path) -> list[dict[str, Any]]:
    """Discover sorted immediate case directories; reject incomplete cases.

    Hidden directories are ignored. Every other directory is treated as a case,
    so incomplete cases cannot be silently omitted. No fixed case count is used.
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")
    records = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        case_id = directory.name
        paths = {key: directory / f"{case_id}-{key.lower()}.nii.gz" for key in MODALITIES}
        paths["mask"] = directory / f"{case_id}-seg.nii.gz"
        record = {"subject_id": case_id, "paths": paths}
        _verify_files(record)
        records.append(record)
    if not records:
        raise ValueError(f"No case directories found in dataset root: {root}")
    return records


def read_manifest(manifest: str | Path, root: str | Path) -> list[dict[str, Any]]:
    """Read CSV columns subject_id,T1n,T1c,T2w,T2f,mask."""
    records = []
    seen = set()
    with Path(manifest).expanduser().open(newline="") as stream:
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
                key: Path(root).expanduser() / Path(row[key]).expanduser() for key in (*MODALITIES, "mask")
            }})
    if not records:
        raise ValueError("Manifest contains no subjects.")
    for record in records:
        _verify_files(record)
    return records


def load_subject(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Load finite 3D volumes on the same voxel grid, without resampling."""
    _verify_files(record)
    volumes = [nib.load(record["paths"][key]) for key in (*MODALITIES, "mask")]
    reference = volumes[0]
    arrays = []
    for key, volume in zip((*MODALITIES, "mask"), volumes):
        context = f"Case {record['subject_id']} ({key})"
        if len(volume.shape) != 3 or volume.shape != reference.shape:
            raise ValueError(f"{context}: shape mismatch; expected 3D {reference.shape}, got {volume.shape}.")
        spacing = np.asarray(volume.header.get_zooms()[:3])
        if not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError(f"{context}: spacing must be finite and positive.")
        if not np.allclose(spacing, reference.header.get_zooms()[:3], atol=1e-5, rtol=0):
            raise ValueError(f"{context}: spacing mismatch with {MODALITIES[0]}.")
        if not np.allclose(volume.affine, reference.affine, atol=1e-5, rtol=0):
            raise ValueError(f"{context}: affine/grid mismatch with {MODALITIES[0]}.")
        array = volume.get_fdata(dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"{context}: volume must contain only finite values.")
        arrays.append(torch.from_numpy(array))
    return torch.stack(arrays[:4]), arrays[4]


class BraTSDataset(Dataset):
    """Return image, default tissue target [4,D,H,W], and raw_mask [D,H,W].

    Set training=True for one random patch per access; default is full volume.
    Normalization uses the full volume before cropping.
    raw_mask preserves label IDs as int64. An explicit label_mapping can
    override target conversion for the legacy three-region baseline.
    """

    def __init__(self, manifest: str | Path | None = None, root: str | Path | None = None,
                 label_mapping: Callable[[torch.Tensor], torch.Tensor] = segmentation_to_tissue_masks,
                 normalize: bool = True, *, training: bool = False,
                 patch_size: tuple[int, int, int] = DEFAULT_PATCH_SIZE,
                 foreground_sampling_probability: float = 0.7,
                 generator: torch.Generator | None = None,
                 augmentation: dict | None = None) -> None:
        # A sole positional path denotes a data root; two paths retain the
        # existing (manifest, root) interface for legacy baseline scripts.
        if root is None:
            root, manifest = manifest, None
        if root is None:
            raise ValueError("Provide a dataset root.")
        self.records = discover_cases(root) if manifest is None else read_manifest(manifest, root)
        self.label_mapping = label_mapping
        self.normalize = normalize
        self.augmentation = TrainingAugmentation(augmentation or {}, generator) if training else None
        self.patch_sampler = PatchSampler(patch_size, foreground_sampling_probability, generator) if training else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image, mask = load_subject(record)
        image = normalize_nonzero(image) if self.normalize else image
        target = self.label_mapping(mask)
        crop_start = None
        if self.patch_sampler is not None:
            # Use raw tissue IDs so RC contributes even with a legacy adapter.
            foreground = torch.zeros_like(mask, dtype=torch.bool)
            for label in TISSUE_LABELS:
                foreground |= mask == label
            slices = self.patch_sampler.sample_slices(foreground)
            crop_start = tuple(part.start for part in slices)
            image = image[(slice(None), *slices)].contiguous()
            target = target[(slice(None), *slices)].contiguous()
            mask = mask[slices].contiguous()
        if self.augmentation is not None:
            image, target, mask = self.augmentation(image, target, mask)
        sample = {"case_id": record["subject_id"],
                "subject_id": record["subject_id"],
                "spacing": tuple(float(v) for v in nib.load(record["paths"][MODALITIES[0]]).header.get_zooms()[:3]),
                "image": image,
                "target": target,
                "raw_mask": mask.to(torch.int64)}

        if crop_start is not None:
            sample["crop_start"] = crop_start
        return sample
