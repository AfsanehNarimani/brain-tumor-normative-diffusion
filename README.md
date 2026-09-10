# Brain tumor normative diffusion research

This project starts with multimodal 3D brain tumor segmentation. The research motivation is to establish a reliable segmentation reference before testing whether separate healthy and pathological representations, and a learned healthy prior, improve tumor delineation. These are future hypotheses, not demonstrated results.

Current scope is **Milestone 1: a strong 3D multimodal BraTS segmentation baseline**. This repository provides an initial, small, readable pure-PyTorch U-Net and baseline-ready training/evaluation organization; it has not yet been trained or validated on BraTS. Healthy/pathological disentanglement and diffusion are future milestones and are **not implemented**.

## Setup and synthetic tests

Use Python 3.11, from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Imports and synthetic tests require no BraTS data. Tests create temporary synthetic NIfTI files and tensors. The scripts are invoked as modules from the repository root.

## Inputs and outputs

- Input: float tensor `[B, 4, D, H, W]`, modality order **T1n, T1c, T2w, T2f**.
- Target: binary overlapping regions `[B, 3, D, H, W]`, order **WT, TC, ET**.
- Output: logits `[B, 3, D, H, W]`; apply `logits.sigmoid()` for region probabilities. There is no 4-class softmax.
- WT means whole tumor, TC tumor core, ET enhancing tumor. Exact raw-label membership is **dataset-release dependent** and must be supplied explicitly. No release mapping is assumed.

The loader preserves NIfTI voxel-axis order and verifies matching shapes and affines; tensor spatial names do not imply anatomical orientation. Spatial dimensions must each be at least 4. Normalization optionally z-scores nonzero voxels per modality, preserving zeros. No resampling, registration, cropping, augmentation, or sliding-window inference is implemented. Full-volume training can require substantial memory; a batch larger than one requires equal spatial shapes across subjects. Confirm preprocessing for the selected release before experiments.

## Configure and run

1. Obtain the selected dataset yourself; nothing downloads it automatically.
2. Follow [data/README.md](data/README.md) to create separate subject manifests.
3. Complete the TODOs in `configs/data.yaml`: exact release identifier, root, manifests, all known raw labels including background, and WT/TC/ET label memberships. The mapper rejects unknown labels and non-nested region definitions.
4. Adjust model width and training parameters in `configs/model.yaml` and `configs/train.yaml`.

Inspect raw files before choosing the mapping:

```bash
python -m scripts.inspect_dataset --root /path/to/data --manifest /path/to/train.csv --limit 2
python -m scripts.train_baseline --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml
python -m scripts.evaluate_baseline --checkpoint outputs/baseline/last.pt --data-config configs/data.yaml
```

All data, manifest, config, checkpoint, and output paths are configurable. Unfilled configuration fails with an actionable error. Evaluation uses the validation manifest and checkpoint model settings, requiring matching release, mapping, and normalization settings.

Training minimizes weighted soft Dice plus BCEWithLogits, with equal averaging across regions. Evaluation thresholds sigmoid probabilities at 0.5 by default and averages per-subject Dice separately for WT/TC/ET. Both-empty regions score 1; one-empty regions score 0. This is an explicit local convention, not a claim of matching an official challenge evaluator. Independent predictions are not forced into nested regions. Checkpoints record model and optimizer state, epoch, and configurations. The seed initializes PyTorch randomness; full cross-device determinism is not promised.

## Organization

- `configs/`: explicit data, model, and training settings.
- `datasets/`: manifest loading, grid checks, normalization, and release-dependent label conversion.
- `models/`: two-level encoder, skip-connected decoder, and baseline U-Net.
- `losses/`: soft region Dice and Dice + BCE.
- `training/`: one-epoch training and checkpoint helpers.
- `evaluation/`: region Dice and subject-averaged evaluation.
- `scripts/`: training, evaluation, and raw dataset inspection entry points.
- `tests/`: synthetic label, data, shape, gradient, and training smoke tests.
- `notebooks/`, `outputs/`: reserved for analysis and generated artifacts.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the staged research plan. The pre-existing `main.py` editor sample is unrelated to the baseline entry points.
