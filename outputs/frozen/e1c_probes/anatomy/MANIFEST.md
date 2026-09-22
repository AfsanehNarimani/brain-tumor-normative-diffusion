# E1-C Anatomy Latent Probe — Frozen Artifact Manifest

## Experiment identity

- Experiment: E1-C latent probe
- Latent source: anatomy/background representation (`Z_a`)
- Scientific purpose: measure pathology accessibility from the E1-C anatomy/background latent representation.
- Probe architecture: `latent_probe_decoder3d`
- Trainable probe parameters: 1,742,852
- Validation threshold: 0.5
- Fixed validation cases: 270

## Frozen E1-C source model

- Source experiment: E1-C
- Source model: `anatomy_constrained_joint_reconstruction_unet3d`
- Source best epoch: 11
- Source checkpoint SHA256:
  `cf5a04a2d845cecb45a0ea87d617abad0afab9ad4264abce8342a6045563ee53`

The E1-C encoder and latent projections were frozen during probe training.

## Probe training protocol

- Epochs requested: 30
- Batch size: 4
- Learning rate: 0.001
- Weight decay: 0.00001
- Optimizer: AdamW
- Seed: 42
- Validation: every epoch
- Device: CUDA
- Best checkpoint selection: validation segmentation metric only

## Interruption and resume

Training was unintentionally interrupted after epoch 8.

Training was resumed from the epoch-8 `latest.pt` checkpoint and continued from epoch 9 through epoch 30.

The resume checkpoint preserved the probe model state, optimizer state, epoch, and best-model state.

Explicit random-number-generator states were not stored in the checkpoint. Therefore, bitwise equivalence to a completely uninterrupted 30-epoch run is not claimed.

The resume event and checkpoint identity are preserved in `training/run_events.jsonl`.

## Final training result

- Training completed: 30 / 30 epochs
- Best epoch: 29
- Best validation metric:
  `0.4335383031219832`

### Best probe checkpoint

SHA256:

`b9439640465108c392d58bc7525df9d2a75a212f26206df84e0365cfd4330e60`

## Independent evaluation

The frozen epoch-29 probe was independently evaluated on the fixed 270-case validation split.

Validation split SHA256:

`ec5fe6b0bbcf190fe8b7c032f8da4ff1d495969c2d868561e7a3351c572cc5ca`

Independent mean validation metric:

`0.43353932396406925`

Presence-aware case Dice:

- NETC: 0.17182311921032237
- SNFH: 0.6260963008282017
- ET: 0.2755190539238457
- RC: 0.5511099311376355
- WT: 0.6778241740953591
- TC: 0.29886336458905133

Positive validation cases:

- NETC: 114
- SNFH: 270
- ET: 198
- RC: 230
- WT: 270
- TC: 200

The independent evaluation reproduces the training-time best metric to approximately 1e-6.

## Evaluation provenance

Evaluator Git commit:

`8d3b3b1d811838f3221f4b7de9d944fc71d2990c`

Environment:

- Python: 3.11.10
- PyTorch: 2.14.0+cu130
- NumPy: 2.2.6
- nibabel: 5.4.2
- Device: NVIDIA CUDA GPU

The evaluation provenance records a dirty working tree caused by known server-local state:

- machine-specific dataset path in `configs/data.yaml`
- materialization of the Git-LFS E1-C source checkpoint
- untracked external training log

These are not scientific model-code modifications.

## Frozen artifact hashes

### Training

`training/best.pt`

`b9439640465108c392d58bc7525df9d2a75a212f26206df84e0365cfd4330e60`

### Evaluation

`evaluation/per_case_metrics.csv`

`d1328d3b800490cf56ee44532d185c9fd2e76cb9ca470fa713df809607212880`

`evaluation/aggregate_metrics.json`

`7b7814da717fb8cbb8a77a5eba70438e79aad897b35458541c2d96a97ad5ba6e`

`evaluation/evaluation_provenance.json`

`cb813eb59801b5c78215fe47a1b38d9078d18954c4e83d9ed86883e8bc02fa1d`

## Interpretation boundary

This experiment measures pathology accessibility from `Z_a`.

`Z_a` is referred to as the anatomy/background representation. It is not assumed to be a pure healthy or normative representation.

No conclusion about preferential latent specialization is made from the anatomy probe alone.

The primary E1-C latent-specialization analysis requires the matched `Z_d` probe and paired per-case statistical comparison between `Z_d` and `Z_a`.
