# E1-C anatomy-loss weight selection

The E1-C protocol fixes `lambda_rec = 1.0` and `lambda_anat = 0.5`. This
decision was made before E1-C training and without observing E1-C validation
segmentation performance.

The selection used only a 10-batch real-data loss-scale diagnostic evaluated at
reference weights `lambda_rec = 1.0` and `lambda_anat = 1.0`. The verified mean
losses were 1.762925 for segmentation, 0.571010 for joint reconstruction, and
0.552752 for anatomy reconstruction. At the selected anatomy weight, the mean
weighted anatomy contribution is 0.276376: approximately 15.7% of the initial
segmentation-loss magnitude and 48.4% of the existing reconstruction-loss
magnitude.

The mask `raw_mask == 0` includes both non-annotated brain and image background.
The auxiliary objective is therefore described as a non-abnormal or
anatomy/background reconstruction constraint. It does not construct a
pseudo-healthy target, prove pathology exclusion from the anatomy/background
representation Za, or establish Za as a normative latent. Latent-probe
evaluation remains required.

The machine-readable decision record is
[`lambda_selection_protocol.json`](lambda_selection_protocol.json). The
original server diagnostic JSON is intentionally absent here and will be
transferred separately so its exact bytes and SHA256 can be preserved.
