# D0-C3b professor summary

## Research question

Can a diffusion-derived deviation feature be spatially associated with tumor regions in the BraTS 2024 development-validation cohort?

## Method

A frozen primary timestep-495 DeltaZ map was compared with WT, TC and ET labels after exact 8×8×8 block mapping. Only MRI-supported cells were used; positive cells had any region voxel, while the comparator consisted of blocks containing only label 0. The patient, not the latent cell, was the unit for aggregate statistics.

## Dataset/evaluation

The analysis used 270 development-validation cases. WT was primary; TC and ET were secondary. Cases without both positive and clean-background supported cells were retained as non-evaluable.

## Key result

WT was evaluable in 270/270 cases, with median AUROC 0.614130 (95% CI 0.606392–0.621043); median AUPRC lift 0.030396; and median DeltaZ difference 0.010929. A positive median difference occurred in 252/270 evaluable WT cases (93.3%). These results suggest a modest spatial association, not diagnostic performance.

## Why it matters

The result motivates testing whether the diffusion-derived feature contributes information that is complementary to the supervised model. Association alone does not establish clinical or segmentation benefit.

## Next research step

Test whether DeltaZ provides complementary information when fused into a supervised segmentation model, with a separately specified evaluation design and independent testing.
