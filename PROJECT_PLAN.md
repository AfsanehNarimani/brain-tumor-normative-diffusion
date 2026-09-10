# Project plan

## Milestone 1: Strong 3D multimodal BraTS segmentation baseline

Current milestone. Build and validate a four-modality (T1n, T1c, T2w, T2f) 3D segmentation baseline with WT/TC/ET sigmoid regions, Dice + BCE loss, and per-region Dice evaluation.

Initial scaffold includes data loading, explicit release-dependent mapping, a small U-Net, training/evaluation scripts, checkpointing, inspection, and synthetic tests. It does not establish baseline quality on real data.

TODO before real experiments:
- Confirm the exact modern BraTS Adult Glioma release and its raw label definitions.
- Verify modality identities, alignment, preprocessing, and file layout.
- Establish subject-disjoint splits and document the evaluation protocol, including empty-region handling.
- Run and assess the baseline; document settings, results, and limitations before advancing.

## Milestone 2: Healthy/pathological latent disentanglement

Future work, not implemented. Investigate separating healthy and pathological latent representations. Define objectives and validation criteria after assessing Milestone 1.

## Milestone 3: Latent diffusion modeling of the healthy representation

Future work, not implemented. Investigate latent diffusion of the healthy representation established in Milestone 2. No diffusion modules, objectives, samplers, or schedules are included now.

## Milestone 4: Fusion for final WT/TC/ET segmentation

Future work, not implemented. Fuse disease features, normative diffusion features, and latent deviation features for final WT/TC/ET segmentation. Determine fusion design and comparisons after earlier milestones are validated.
