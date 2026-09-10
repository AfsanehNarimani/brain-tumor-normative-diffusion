# Local data

No dataset is included or downloaded. Obtain the chosen BraTS Adult Glioma release through its authorized distribution and follow its terms.

TODO: confirm the exact release, raw label definitions, modality filenames, and preprocessing before training. Set paths and the explicit raw-label mapping in `configs/data.yaml`.

Supply separate training and validation CSV manifests with columns:

```csv
subject_id,T1n,T1c,T2w,T2f,mask
example,example/a.nii.gz,example/b.nii.gz,example/c.nii.gz,example/d.nii.gz,example/mask.nii.gz
```

These are illustrative filenames, not claims about any BraTS release. Modality and mask paths may be absolute or relative to the configured `root`. Configuration paths and manifest paths are relative to the process working directory unless absolute. Subject IDs must be unique in each manifest and disjoint across training and validation. Each subject requires four aligned 3D NIfTI volumes and one integer-valued segmentation mask on the same grid.
