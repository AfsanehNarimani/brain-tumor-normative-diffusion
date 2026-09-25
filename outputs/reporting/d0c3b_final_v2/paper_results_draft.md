# Results

WT was the predeclared primary endpoint and was evaluable in 270/270 cases. The median case-level AUROC was 0.614130 [0.606392, 0.621043], median AUPRC was 0.128997 [0.109080, 0.143982], median AUPRC lift was 0.030396 [0.026699, 0.036569], and median case-level DeltaZ difference was 0.010929 [0.010256, 0.011505]. 252/270 evaluable cases (93.3%) had positive median DeltaZ difference. Holm-adjusted p-values were 6.062e-41, 2.604e-43, and 2.261e-35 for the three predeclared test families.

TC was evaluable in 200/270 cases and ET in 198/270. Their median AUROCs were 0.619183 and 0.619359; median AUPRC lifts were 0.009065 and 0.008520; median DeltaZ differences were 0.010578 and 0.010651, respectively.

# Discussion

The results indicate a modest but consistent spatial association between the diffusion-derived deviation feature and tumor-containing regions in the development-validation cohort. The association is descriptive and does not establish diagnostic or clinical performance. The result motivates evaluation of whether DeltaZ adds complementary information when fused into a supervised segmentation model.

# Limitations

The cohort is development-validation and is not an independent test set. The AUROC values indicate moderate association. Za is described as an anatomy/background representation, not as healthy or pathology-free. DeltaZ has not yet been shown to improve a supervised segmentation model. Sensitivity timesteps 327 and 663 remain deferred experiments.
