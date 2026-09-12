# Local data

No dataset is included or downloaded. Obtain the chosen BraTS Adult Glioma release through its authorized distribution and follow its terms.

The user-provided full-dataset inspection of **BraTS 2024 Adult Glioma Post-Treatment** found:

- Root: `~/datasets/BraTS2024/train/training_data1_v2`.
- 1,350 cases; no missing segmentation files.
- Per-case files: `<case>-t1n.nii.gz`, `<case>-t1c.nii.gz`, `<case>-t2w.nii.gz`, `<case>-t2f.nii.gz`, `<case>-seg.nii.gz`.
- All volumes: `(182, 218, 182)` voxels at `1 × 1 × 1 mm` spacing.
- Observed raw labels across the dataset: 0, 1, 2, 3, 4. Individual cases may omit tissues.

| Constant | Raw label | Meaning |
| --- | --- | --- |
| BACKGROUND | 0 | Background |
| NETC | 1 | Non-Enhancing Tumor Core |
| SNFH | 2 | Surrounding Non-Enhancing FLAIR Hyperintensity |
| ET | 3 | Enhancing Tissue |
| RC | 4 | Resection Cavity |

This is a post-treatment dataset with a separate cavity label, unlike the older BraTS setup used in the MSc project. The [challenge manuscript, section 3](https://arxiv.org/html/2405.18368v1#S3) defines TC as NETC + ET and WT as NETC + SNFH + ET; neither includes RC. These unions are explicit in `COMPOSITE_TISSUES`, not inferred from nonzero labels.

`BraTSDataset` defaults to float32 binary tissue targets in `[NETC, SNFH, ET, RC]` order and returns original IDs as `raw_mask` (int64). It preserves voxel-axis order. Shape/spacing observations above describe the inspected data, not hard-coded loader constraints. The loader expands `~` in root and manifest paths automatically. Use `BraTSDataset(root=data_root)` for directory discovery; manifests are optional for this interface.

`configs/data.yaml` records the verified release and external root. Subject-disjoint train/validation manifests must be configured before real training. The baseline uses four-tissue targets directly.

Supply separate training and validation CSV manifests with columns:

```csv
subject_id,T1n,T1c,T2w,T2f,mask
example,example/a.nii.gz,example/b.nii.gz,example/c.nii.gz,example/d.nii.gz,example/mask.nii.gz
```

These are illustrative filenames, not claims about any BraTS release. Modality and mask paths may be absolute or relative to the configured `root`. Configuration paths and manifest paths are relative to the process working directory unless absolute. Subject IDs must be unique in each manifest and disjoint across training and validation. Each subject requires four aligned 3D NIfTI volumes and one integer-valued segmentation mask on the same grid.
