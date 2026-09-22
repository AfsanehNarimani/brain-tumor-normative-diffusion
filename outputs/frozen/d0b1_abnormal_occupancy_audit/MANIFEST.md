# D0-B1 — Abnormal Latent Occupancy Audit

## Purpose

Train-only, no-training audit measuring how annotated abnormal tissue maps from
the fixed 96x96x96 MRI patches used by D0-B0 to the 12x12x12 latent grid.

This audit informs a future abnormal-region exclusion policy for latent
diffusion training. It does not select the final exclusion threshold.

## Experimental provenance

- Implementation commit: `8bcebc9afb170b444c523d6a6beb51144ce7358a`
- Experiment base commit: `12738979614d4268a8f2e42774808758317db4dc`
- Train split SHA256: `581dc6326a22cb0b9582fa249b8e13fd948597a60c5f454cb5ade0493fa7c18e`
- D0-B0 patch manifest SHA256: `9c4ba67fc195537a5500229722ce55e4772ea5b7d135b351b3a2fc4b857f44b9`
- Cases: 1,080
- Patches: 21,600
- Patch size: 96x96x96
- Latent grid: 12x12x12
- Mapping block: 8x8x8
- Raw abnormal labels: 1, 2, 3, 4
- Locked MRI-support occupancy threshold: 0.50
- Validation data used: no
- Model training performed: no
- New random sampling performed: no
- Rejection sampling used: no
- Augmentation used: no
- Final abnormal threshold selected: no

Ground-truth segmentation is used only for this training-policy audit and is
not an inference input.

## Integrity

- `analysis_provenance.json`
  SHA256: `51f68a30b4b89e47e1499a063e0c5991b16c0782d67b64466a72287ae2974851`

- `patch_abnormal_statistics.csv`
  SHA256: `bea8431c6f97cc97b89fd6f218d1c23deabaca9193338f7c63ceebbad856ccd1`
  Data rows: 21,600

- `case_abnormal_statistics.csv`
  SHA256: `e493d4cfe783fee0645dd0be802c6b4e11c9b9a7c1f2dc371a44477a2e7aa6c9`
  Data rows: 1,080

- `abnormal_occupancy_summary.json`
  SHA256: `d0345c9c6c9e06241c9906082bd91e02ee57e036e4d66996e2f85934d9425da7`

The four scientific artifacts were transferred Server-to-Mac and verified
byte-exact against their source SHA256 values before freezing.

## Main findings

- Abnormal-positive patches: 18,951 / 21,600 = 87.74%.
- Abnormal-negative patches: 2,649 / 21,600 = 12.26%.

Mean fraction of valid MRI-supported latent cells excluded:

| Abnormal occupancy criterion | All patches | Abnormal-positive patches |
| --- | ---: | ---: |
| > 0 | 0.121039 | 0.137958 |
| >= 0.01 | 0.112438 | 0.128155 |
| >= 0.05 | 0.098604 | 0.112387 |
| >= 0.10 | 0.088719 | 0.101120 |
| >= 0.25 | 0.070865 | 0.080770 |
| >= 0.50 | 0.050798 | 0.057899 |

Thus, direct exclusion of every valid-support latent cell whose corresponding
8x8x8 block contains any annotated abnormal voxel would exclude approximately
12.10% of valid-support latent cells on average across all sampled patches.

## Interpretation

D0-B1 does not establish a final abnormal-occupancy threshold.

The frozen encoder has an effective bottleneck stride of 8 voxels, while the
theoretical receptive field of a bottleneck location is substantially larger
than its directly corresponding 8x8x8 block. Therefore, zero abnormal
occupancy in the directly corresponding block does not establish absence of
pathology-related context in the latent representation.

A receptive-field-aware audit is required before locking the D0-B abnormal
exclusion/corruption strategy.

No claim is made that Za is a healthy or normative representation.
