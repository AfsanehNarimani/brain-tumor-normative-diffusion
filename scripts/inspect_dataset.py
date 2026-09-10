"""Inspect raw subject volumes without requiring a label mapping."""
import argparse
import nibabel as nib
import numpy as np
from datasets.brats_dataset import MODALITIES, read_manifest


def main() -> None:
    """Print paths, shapes, labels, and full-volume intensity statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    for record in read_manifest(args.manifest, args.root)[:args.limit]:
        print(f"Subject ID: {record['subject_id']}")
        for key in (*MODALITIES, "mask"):
            path = record["paths"][key]
            array = nib.load(path).get_fdata(dtype=np.float32)
            print(f"  {key}: {path} shape={array.shape}")
            if key == "mask":
                print(f"    unique segmentation labels: {np.unique(array).tolist()}")
            else:
                print(f"    intensity (all voxels): min={array.min():.5g} max={array.max():.5g} "
                      f"mean={array.mean():.5g} std={array.std():.5g} "
                      f"finite={np.isfinite(array).all()}")


if __name__ == "__main__":
    main()
