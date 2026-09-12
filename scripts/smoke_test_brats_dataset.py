"""Inspect the first three real BraTS cases without training."""
import argparse
import torch
from datasets.brats_dataset import BraTSDataset, MODALITIES
from datasets.label_mapping import TISSUE_NAMES, derive_wt, derive_tc
from datasets.transforms import normalize_nonzero


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    dataset = BraTSDataset(root=args.data_root, normalize=False)
    print(f"Discovered {len(dataset)} cases", flush=True)
    for index in range(min(3, len(dataset))):
        sample = dataset[index]
        raw_image = sample["image"]
        image = normalize_nonzero(raw_image)
        target = sample["target"]
        print(f"\ncase_id: {sample['case_id']}")
        print(f"image shape: {tuple(image.shape)}; target shape: {tuple(target.shape)}")
        print(f"image dtype: {image.dtype}; target dtype: {target.dtype}")
        print(f"spacing: {sample['spacing']}")
        # Original foreground is essential: normalization can turn tissue
        # voxels equal to the mean into zeros (all voxels for constant MRI).
        for name, raw, normalized in zip(MODALITIES, raw_image, image):
            values = normalized[raw != 0]
            mean = values.mean().item() if values.numel() else 0.0
            std = values.std(unbiased=False).item() if values.numel() else 0.0
            print(f"{name.lower()} normalized mean/std over original nonzero voxels: {mean:.6f}/{std:.6f}")
        for name, mask in zip(TISSUE_NAMES, target):
            print(f"{name} voxel count: {int(mask.sum())}")
        print(f"WT voxel count: {int(derive_wt(target).sum())}")
        print(f"TC voxel count: {int(derive_tc(target).sum())}", flush=True)
        assert image.dtype == target.dtype == torch.float32


if __name__ == "__main__":
    main()
