# D0-B0 Support-Mask Audit — Frozen Artifact Manifest

## Purpose

D0-B0 is a train-only, no-training audit performed before D0-B latent diffusion.

Its purpose is to characterize how raw multimodal MRI support within randomly sampled 96×96×96 input patches maps to the 12×12×12 latent grid, and to define the spatial-sampling and latent-support policy for D0-B.

This audit does not train or optimize a model.

## Data Scope

- Dataset: BraTS 2024 Adult Glioma Post-Treatment v2
- Split: fixed training split only
- Cases: 1,080
- Candidate patches per case: 20
- Total patches: 21,600
- Patch size: 96×96×96
- Validation data used: no
- Augmentation used: no
- Tumor/foreground enrichment: no
- Support-based rejection sampling: no
- Segmentation used for sampling: no
- Segmentation used for support definition: no

Training split SHA256:

`581dc6326a22cb0b9582fa249b8e13fd948597a60c5f454cb5ade0493fa7c18e`

## Code Provenance

Experiment implementation commit:

`78d498a2ecb1782cd24019df03b5262520ae5bfb`

Experiment base commit:

`cd2fdb16a159af6626c6cabb2c10739b1a505afd`

The scientific run was executed from the repository root with the repository explicitly placed on the Python import path:

`PYTHONPATH=. python scripts/audit_d0b_support_mask.py --data-config configs/data.yaml --output-dir outputs/d0b_support_mask_audit`

An earlier launch without `PYTHONPATH=.` failed during Python import with `ModuleNotFoundError` before scientific processing began and produced no scientific output.

The successful run recorded a dirty Server worktree. The known dirty state included the machine-specific dataset path configuration and the pre-existing Git-LFS pointer/worktree representation of the frozen E1-C checkpoint. The E1-C checkpoint content had previously been independently verified against its expected SHA256. These Server-specific worktree conditions were not used to alter the D0-B0 scientific protocol.

## Support Definition

Input support is defined before normalization:

A voxel is supported if at least one of the four MRI modalities is nonzero.

Segmentation labels are not used.

Input support is mapped from 96×96×96 to 12×12×12 using exact non-overlapping 8×8×8 blocks.

For each latent location, occupancy is the fraction of the corresponding 512 input voxels belonging to MRI support.

## Audit Result

Input support fraction across the 21,600 random patches:

- mean: 0.570405
- p01: 0.203001
- p05: 0.297499
- p25: 0.451878
- p50: 0.569541
- p75: 0.688553
- p95: 0.849578
- p99: 0.920364
- min: 0.069795
- max: 0.986681

The train-only audit did not show random spatial sampling to be dominated by near-empty background patches.

Therefore D0-B will not introduce support-based patch rejection.

## Locked D0-B Sampling Decision

- case-balanced sampling
- random 96×96×96 spatial patches
- no tumor-centered oversampling
- no support-based patch rejection
- no validation-derived sampling rule

## Locked Primary Latent-Support Rule

Primary valid latent support:

`occupancy >= 0.50`

This is a geometric majority-support rule: at least half of the corresponding 8×8×8 input block must contain MRI support.

The threshold was not selected by optimizing validation performance and is not claimed to be empirically optimal.

Planned support-threshold sensitivity values:

- 0.25
- 0.50 (primary)
- 0.75

At the primary 0.50 threshold, the mean fraction of latent locations classified as supported was approximately 0.570700.

## Scope Limitation

D0-B0 defines spatial sampling and MRI-support handling only.

It does not determine:

- abnormal-region handling during diffusion training
- diffusion loss formulation
- diffusion architecture
- noise schedule
- inference operating point
- DDIM reconstruction settings
- deviation-map evaluation

Those decisions belong to the subsequent D0-B scientific design.

## Frozen Source Artifacts

- `analysis_provenance.json`
  - SHA256: `e5f7527e25759d0497fc56130a46782c43298dc0ffc73f6fbd2b30d71db98a15`

- `case_support_statistics.csv`
  - SHA256: `d3df3d324264e1e5bd319f1fa249ebf200a432fdc30af17d749b56d64b0c3aa0`

- `patch_manifest.csv`
  - SHA256: `9c4ba67fc195537a5500229722ce55e4772ea5b7d135b351b3a2fc4b857f44b9`

- `patch_support_statistics.csv`
  - SHA256: `ccc6292a7bba4ca05dc5402df4ec053e76cd36d213db87dc328c4aa70ef01a58`

- `support_mask_decision.json`
  - SHA256: `9317fe5f7be1bd55dd69b1c1f3e12c1081f6cf5ecb486a0748ac3d15b6794b77`

- `support_mask_summary.json`
  - SHA256: `2dc58b67885dc4c3a3e5bb5dd428873b68ba5bf5f37b9b101eac01e7e1fc2933`

These six source artifacts were transferred from the Server to the Mac repository and verified byte-for-byte using SHA256 before this manifest was created.
