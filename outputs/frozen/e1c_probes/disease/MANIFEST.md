# E1-C Disease-Latent Probe — Frozen Scientific Artifact

## Purpose

Frozen artifacts for the E1-C disease-latent (`Zd`) probe experiment.

The experiment evaluates pathology accessibility from the disease representation of the E1-C anatomy/background-constrained model.

## Scientific protocol

- Source experiment: E1-C
- Source model: anatomy-constrained joint-reconstruction 3D U-Net
- Latent source: disease (`Zd`)
- Source checkpoint epoch: 11
- Probe architecture: latent_probe_decoder3d
- Probe trainable parameters: 1,742,852
- Training epochs: 30
- Batch size: 4
- Learning rate: 0.001
- Weight decay: 0.00001
- Seed: 42
- Validation threshold: 0.5
- Validation cases: 270

## Training execution

The scientific E1-C Zd probe run started cleanly from epoch 1 and completed all 30 requested epochs without resume.

An earlier launch terminated before completion of the first epoch and produced no checkpoint or completed epoch. That pre-epoch launch is not part of the scientific training run.

Best epoch: 30

Best validation metric:

`0.43147381465581763`

## Independent evaluation

Independent full-volume evaluation was performed on the fixed 270-case validation split.

Mean validation metric:

`0.43147343676893213`

Training-time and independent evaluation results agree to approximately 3.78e-7 in the mean validation metric.

## Source identity

E1-C frozen source checkpoint SHA256:

`cf5a04a2d845cecb45a0ea87d617abad0afab9ad4264abce8342a6045563ee53`

Validation split SHA256:

`ec5fe6b0bbcf190fe8b7c032f8da4ff1d495969c2d868561e7a3351c572cc5ca`

Scientific run Git commit:

`46b85e2c42761c2e5f524ed75df8d221d0b5bb85`

## Artifact hashes

### Training

best.pt:

`6ca1011152f58c56576042c54df83ae624506bd31c21dca945096ecd6bd02318`

latest.pt was retained on the training server but is not part of this frozen archive.

### Evaluation

per_case_metrics.csv:

`f68053782d926051c06ea2b6619652c3ee95f9f366f0065b163764d32515185b`

aggregate_metrics.json:

`e2ba6d1a1a9b115d9f60c4d5269860fcdcb7bf1c5f37739d84ecb8984a54b740`

evaluation_provenance.json:

`3cd912afafeb1a29338ea6ebc24a57848273b737b892f0afd9a94045e617d9f9`

## Interpretation policy

This artifact alone is not used to claim latent specialization.

Scientific interpretation requires the pre-specified paired comparison of E1-C `Za` and `Zd` probe performance on the same validation cases, including paired bootstrap confidence intervals, Wilcoxon signed-rank tests, Holm correction across regions, and win/tie/loss counts.
