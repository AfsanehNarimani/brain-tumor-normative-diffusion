# Project plan

## Milestone 1: Strong 3D multimodal BraTS segmentation baseline

Current milestone. Build and validate a four-modality (T1n, T1c, T2w, T2f) 3D segmentation baseline. Primary tissue targets remain **[NETC, SNFH, ET, RC]**; WT and TC are derived evaluation regions, not direct model targets.

The verified BraTS 2024 Adult Glioma Post-Treatment raw labels are Background = 0, NETC = 1, SNFH = 2, ET = 3, RC = 4. Exact composite definitions are **WT = NETC ∪ SNFH ∪ ET = {1, 2, 3}** and **TC = NETC ∪ ET = {1, 3}**. RC is evaluated independently and excluded from both WT and TC. Unlike the older BraTS setup used in the MSc project, this dataset explicitly includes resection cavity.

Milestone 1 evaluation uses standard volumetric Dice for **NETC, SNFH, ET, RC, WT, TC**, averaged per subject. Both-empty regions score 1 and one-empty regions score 0. Threshold the four tissue predictions before deriving unions. Lesion-wise challenge metrics are not implemented.

The baseline now outputs four tissue logits using a 32/64/128/256 GroupNorm U-Net, with a reusable encoder and decoder. Synthetic forward/backward validation is implemented; real training has not started.

Initial scaffold includes data loading, explicit release-dependent mapping, a small U-Net, training/evaluation scripts, checkpointing, inspection, and synthetic tests. It does not establish baseline quality on real data.

TODO before real experiments:
- Verify modality identities, alignment, preprocessing, and file layout.
- Establish subject-disjoint splits and document the evaluation protocol, including empty-region handling.
- Run and assess the baseline; document settings, results, and limitations before advancing.

## Milestone 2: Healthy/pathological latent disentanglement

Future work, not implemented. Investigate separating healthy and pathological latent representations. Define objectives and validation criteria after assessing Milestone 1.

## Milestone 3: Latent diffusion modeling of the healthy representation

Future work, not implemented. Investigate latent diffusion of the healthy representation established in Milestone 2. No diffusion modules, objectives, samplers, or schedules are included now.

## Milestone 4: Fusion for final tissue segmentation

Future work, not implemented. Fuse disease features, normative diffusion features, and latent deviation features for final NETC/SNFH/ET/RC segmentation, deriving WT/TC for evaluation. Determine fusion design and comparisons after earlier milestones are validated.
