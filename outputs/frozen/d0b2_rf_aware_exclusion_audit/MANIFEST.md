# D0-B2 — Receptive-Field-Aware Abnormal Exclusion Audit

## Status

Scientific run completed and frozen.

This is a train-only, no-training feasibility audit. It does not select a final diffusion corruption strategy or abnormal-region policy.

## Source

Implementation branch:

`experiment/d0b2-rf-aware-exclusion-audit`

Experiment base commit:

`7c0fcd74f04cfee82a0c0f24e7d4d350d24e6ce9`

Implementation commit:

`25cbc929c88f6b8c781eb1341c2e004e7ad33798`

## Frozen inputs

Training split SHA256:

`581dc6326a22cb0b9582fa249b8e13fd948597a60c5f454cb5ade0493fa7c18e`

D0-B0 patch manifest SHA256:

`9c4ba67fc195537a5500229722ce55e4772ea5b7d135b351b3a2fc4b857f44b9`

D0-B1 abnormal occupancy summary SHA256:

`d0345c9c6c9e06241c9906082bd91e02ee57e036e4d66996e2f85934d9425da7`

## Audit scope

- Train cases: 1,080
- Frozen random patches: 21,600
- Patch shape: 96 x 96 x 96
- Latent grid: 12 x 12 x 12
- Direct correspondence block: 8 x 8 x 8
- Raw abnormal labels: {1, 2, 3, 4}
- No validation data
- No new random sampling
- No augmentation
- No rejection sampling
- No model training
- No diffusion model

MRI support is defined as any raw MRI modality being nonzero before normalization.

Valid latent support uses exact 8 x 8 x 8 block occupancy >= 0.50.

## Receptive-field geometry

The receptive-field geometry is derived from the actual frozen encoder architecture.

- Effective bottleneck stride: 8 voxels
- Theoretical receptive-field size: 68 x 68 x 68 voxels
- Boundary receptive fields are intersected with the real 96 x 96 x 96 input patch.

A latent location is RF-contaminated when any annotated abnormal voxel intersects its clipped real-input theoretical receptive field.

Primary eligibility:

`valid_support AND NOT rf_contaminated`

## Integrity

Patch statistics:

- 21,600 data rows

Case statistics:

- 1,080 data rows

D0-B1 direct-exclusion consistency:

- Frozen expected mean: 0.12103873524831282
- Recomputed mean: 0.12103873492763219
- Absolute difference: 3.2068063482437736e-10
- Tolerance: 1e-6
- Status: PASS

## Main finding

At the case level, strict receptive-field-aware exclusion removed approximately 62.85% of valid MRI-supported latent locations on average.

Approximately 37.15% of valid MRI-supported latent locations remained eligible on average.

Eligibility was highly heterogeneous across cases. The case-level eligible fraction of valid support had:

- p01: 0.01348
- p05: 0.05475
- p25: 0.20854
- median: 0.37660
- p75: 0.51942
- p95: 0.71014
- p99: 0.82315

Therefore, direct 8 x 8 x 8 block exclusion substantially understates potential pathology exposure under the encoder's larger theoretical receptive field, while strict post-encoding RF-aware exclusion is highly restrictive and heterogeneous.

This audit does not establish a final abnormal-handling policy and does not establish that Za is healthy or normative.

## Frozen artifact SHA256

`analysis_provenance.json`

`753647401864e2e19607f6ba1b6aef51149c1f2c1f01e4dae41d5969b0e6b394`

`case_rf_statistics.csv`

`fef6dc16fcfc9b680bae602f7e7d00e3294615d141ec0c3a9c17fddc1d556fff`

`patch_rf_statistics.csv`

`ba30f4b5e36b029415af4e60d887cd949cc61374250ed8e4f3bcbb11d6241027`

`rf_exclusion_summary.json`

`b1b0039802c5099e5406323c6c1048db5f15d57c14db6ff022350e6aa6cf79c8`

`rf_geometry.json`

`23bb5ff94c83b634e46ae4fe96449a8d5de2ea078764234d767c7964e5c83922`
