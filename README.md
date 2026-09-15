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
- Default dataset target: binary float32 `[4, D, H, W]`, order **NETC, SNFH, ET, RC** (batched: `[B, 4, D, H, W]`). Background is excluded; absent tissues retain zero-filled channels.
- Each sample also exposes `raw_mask`: int64 `[D, H, W]` with original BraTS label IDs. Conversion does not mutate the source segmentation.
- `segmentation_to_tissue_masks` and the named label constants live in `datasets/label_mapping.py`. `derive_wt`, `derive_tc`, `derive_et`, and `derive_rc` accept binary `[4,D,H,W]` tissue masks and return binary float32 `[D,H,W]` regions. ET and RC select their exact tissue channels.
- Raw IDs: Background = 0, NETC = 1, SNFH = 2, ET = 3, RC = 4.
- WT = NETC ∪ SNFH ∪ ET = labels {1, 2, 3}; TC = NETC ∪ ET = labels {1, 3}. RC is excluded from both, following [section 3 of the challenge manuscript](https://arxiv.org/html/2405.18368v1#S3). Memberships are centralized in `COMPOSITE_TISSUES` for explicit future adjustments.

The inspected dataset is **BraTS 2024 Adult Glioma Post-Treatment**. Unlike the older BraTS setup used in the MSc project, it includes a separate resection cavity (RC) class. ET means enhancing tissue, and SNFH can include treatment-related changes. Label 4 means RC in this release; older label conventions must not be reused. See [data/README.md](data/README.md) for the inspected layout and label table.

The baseline is a pure-PyTorch four-resolution U-Net: two Conv3d–GroupNorm–ReLU layers per block, encoder widths **32/64/128/256**, three max-pooling steps, trilinear upsampling to skip shapes, and a 1×1×1 output head. Inputs, outputs, and base width are configurable; width multipliers are fixed at [1,2,4,8]. It returns **raw logits** in [NETC,SNFH,ET,RC] order, without sigmoid. Default trainable parameter count: **5,840,132**. Encoder and decoder are separate reusable modules; no latent branches are implemented.

Run the default synthetic 96³ forward check with `python -m scripts.smoke_test_baseline_model`; add `--backward` for an optional synthetic Dice+BCE gradient check. Tests use reduced widths for speed and verify odd shapes, 96³ output geometry, raw logits, and finite gradients. No real training has been run.

Use `BraTSDataset(root=data_root)` to discover sorted immediate case directories dynamically. It validates every required filename during discovery and returns `case_id`, `image`, `target`, `raw_mask`, and `spacing`; `subject_id` is retained for compatibility. Hidden directories are ignored; incomplete case directories raise errors. The optional `(manifest, root)` interface remains available for explicit splits. Paths expand `~` automatically.

Run a real-data inspection of the first three cases without training:

```bash
source .venv/bin/activate
python -m scripts.smoke_test_brats_dataset --data-root ~/datasets/BraTS2024/train/training_data1_v2
```

The script prints shapes, dtypes, spacing, normalized per-modality population mean/std over original nonzero voxels, and tissue/WT/TC counts. The external root is also recorded in `configs/data.yaml`; no user-specific absolute path is embedded in Python.

The loader preserves NIfTI voxel-axis order and verifies matching 3D shapes, voxel spacing, and affines; tensor spatial names do not imply anatomical orientation. The model requires spatial dimensions of at least 8 and explicitly aligns odd skip sizes. Normalization defaults to independent nonzero-voxel z-scoring per modality using population standard deviation, preserving zero background. Constant foreground and all-zero modalities remain finite; segmentation is never normalized. No resampling, registration, or augmentation is implemented. Validation uses deterministic tiled inference. Full-volume training can require substantial memory; a batch larger than one requires equal spatial shapes across subjects. Confirm preprocessing for the selected release before experiments.

## Configure and run

1. Obtain the selected dataset yourself; nothing downloads it automatically.
2. Follow [data/README.md](data/README.md) to create separate subject manifests.
3. Set the external root and subject-disjoint train/validation manifests in `configs/data.yaml`. Targets use the validated four-tissue mapping.
4. Adjust model width and training parameters in `configs/model.yaml` and `configs/train.yaml`.

Baseline commands (not run as part of this dataset update):

```bash
python -m scripts.inspect_dataset --root /path/to/data --manifest /path/to/train.csv --limit 2
python -m scripts.train_baseline --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml
python -m scripts.evaluate_baseline --checkpoint outputs/baseline/last.pt --data-config configs/data.yaml
```

All data, manifest, config, checkpoint, and output paths are configurable. Unfilled configuration fails with an actionable error. Evaluation uses the validation manifest and checkpoint model settings, requiring matching release, mapping, and normalization settings.

The training scaffold minimizes weighted soft Dice plus BCEWithLogits. For Milestone 1, evaluation uses **standard volumetric Dice only**, with no lesion-wise challenge metrics. `region_dice(logits, target)` accepts `[B,4,D,H,W]` in NETC/SNFH/ET/RC order and returns `[B,6]` in **NETC, SNFH, ET, RC, WT, TC** order. It thresholds sigmoid probabilities at 0.5 by default before deriving binary WT/TC unions. The evaluator averages these scores across subjects. Legacy three-channel inputs retain WT/TC/ET scoring. Both-empty regions score 1; one-empty regions score 0. This is an explicit local convention, not a claim of matching an official challenge evaluator. Independent predictions are not forced into nested regions. Checkpoints record model and optimizer state, epoch, and configurations. The seed initializes PyTorch randomness; full cross-device determinism is not promised.

## Organization

- `configs/`: explicit data, model, and training settings.
- `datasets/`: manifest loading, grid checks, normalization, and BraTS 2024 tissue conversion and explicit legacy region conversion.
- `models/`: four-resolution encoder, skip-connected decoder, and baseline U-Net.
- `losses/`: soft region Dice and Dice + BCE.
- `training/`: one-epoch training and checkpoint helpers.
- `evaluation/`: region Dice and subject-averaged evaluation.
- `scripts/`: training, evaluation, and raw dataset inspection entry points.
- `tests/`: synthetic label, data, shape, gradient, and training smoke tests.
- `notebooks/`, `outputs/`: reserved for analysis and generated artifacts.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the staged research plan. The pre-existing `main.py` editor sample is unrelated to the baseline entry points.

## Training patches

`BraTSDataset(root=data_root, training=True)` returns one aligned patch per case access, defaulting to `[4,96,96,96]` images and NETC/SNFH/ET/RC targets. Configure `patch_size` and `foreground_sampling_probability` in `configs/data.yaml`. The training script applies these settings only to the training split; validation/evaluation retain full volumes. The model now outputs the same four tissue channels.

With probability 0.7, sampling selects a voxel uniformly from the union of all four tissues (including RC), then applies bounded random center jitter of up to one quarter patch width per axis. The selected voxel stays inside the crop. Otherwise, crop starts are uniform over all valid spatial starts. Empty foreground falls back to random sampling. Starts are clamped to image bounds, preserving exact configured dimensions; oversized patches raise an error rather than padding or resizing.

Full-volume normalization precedes cropping. Image, target, and raw segmentation share identical coordinates; training samples include `crop_start` in original voxel coordinates. Supply a seeded CPU `torch.Generator` for reproducible sampling sequences, or use `torch.manual_seed` with the default generator. For multiworker loading, prefer the default RNG so PyTorch worker seeds provide distinct streams; a supplied generator is intended for single-worker reproducibility. No augmentation or training run is included in this update.

## Tiny-batch memorization check

Before real training, run the standalone sanity check:

```bash
python -m scripts.overfit_tiny_batch --data-root ~/datasets/BraTS2024/train/training_data1_v2 --case-count 2 --steps 200 --learning-rate 1e-3 --seed 42 --device cpu
```

Use `--device cuda` or `--device mps` when available. The script caches one seeded foreground-centered 96³ patch from each of the first requested foreground-containing cases in sorted order. It skips entirely empty cases and accepts absent individual tissues. Cached images/targets are reused unchanged for every step. Adam optimizes the default U-Net with Dice+BCE; gradients accumulate across the fixed cases before each update to limit peak memory. There is no scheduler, augmentation, split, or checkpoint output.

Initial and final losses use identical cached samples. Every ten updates (and at the final update), it prints post-update loss and per-tissue volumetric Dice. A loss reduction strictly greater than 50% is reported as heuristic memorization success; this is not a generalization result or a required rare-class Dice threshold. Otherwise it reports `NOT YET MEMORIZED`. Deterministic algorithms are requested; reproducibility is scoped to the same supported device/software environment. This command starts only the explicitly requested tiny optimization experiment, not full training.

## Reproducible case split

```bash
python -m scripts.create_data_split --data-root ~/datasets/BraTS2024/train/training_data1_v2 --output-dir splits --train-ratio 0.8 --seed 42
```

This scans segmentation labels only and records NETC/SNFH/ET/RC presence vectors. Seeded shuffling within presence-pattern strata produces a fixed overall training count (rounded to the nearest case). Strata with at least two cases receive representation in both partitions when jointly feasible; singleton strata cannot be shared. Quota adjustments keep the requested total while minimizing departures from proportional allocations.

The utility writes sorted `train_cases.txt`, `val_cases.txt`, and `split_summary.json`, including each case's presence vector, pattern distributions, and per-tissue counts/percentages. Re-running with identical inputs and seed overwrites these files reproducibly. This is a **case-level** split; it does not group longitudinal scans by patient. Patient grouping requires verified patient identifiers before claiming patient-independent validation. The case-ID lists are persistent split artifacts; the existing training script's CSV manifest interface has not been changed by this utility.

## First baseline training pipeline

```bash
python -m scripts.train_baseline --data-root ~/datasets/BraTS2024/train/training_data1_v2 --train-cases splits/train_cases.txt --val-cases splits/val_cases.txt --config configs/train.yaml --output-dir outputs/baseline_unet3d_debug --device cpu --debug
```

Run debug before full training. Debug uses one training batch, one complete validation case, one epoch, zero workers, and base width 2; it exercises optimization, validation, CSV logging, and checkpointing. It retains 96³ training patches and full-volume validation coverage. It uses a separate output directory. The default non-debug architecture remains width 32. Remove `--debug` and choose a fresh output directory only when ready for full training.

The pipeline reads the fixed case-ID lists, rejects duplicates, unknown cases and overlap, and applies random 96³ patches with foreground probability 0.7 only to training. Validation uses batch size 1 and deterministic covering tiles, averaging raw logits in overlap regions before computing full-volume Dice+BCE and Dice. Reconstruction and loss/metrics are on CPU to limit accelerator memory. No random validation crops are used.

`configs/train.yaml` defines 100 epochs, batch size 1, AdamW learning rate 1e-3, weight decay 1e-5, seed 42, zero workers, device, output/checkpoint directory, and validation frequency. Python, NumPy and PyTorch are seeded; loader workers receive seeded RNG streams. Exact determinism can depend on backend, device and software versions. No scheduler or augmentation is used.

`history.csv` records each epoch's training loss, validation loss, six Dice metrics and learning rate. Validation columns remain blank on skipped validation epochs; the last epoch is always validated. `latest.pt` is saved each epoch. `best.pt` maximizes the equal-weight mean of target-positive case Dice for NETC/SNFH/ET/RC/WT/TC, including only regions with at least one positive target case and retaining the earlier checkpoint on ties. Both contain model/optimizer state, epoch, effective configuration, best metric and seed. Existing run outputs are rejected to prevent accidental overwrite; resume is not implemented.

Low-level Dice uses both-empty=1 and one-empty=0. Validation checkpoint selection excludes target-empty cases, so correct empty predictions cannot raise its score. No lesion-wise challenge metric is claimed. The case-level split is not patient-grouped. The full training run has not been started by this implementation.

## Short real run

```bash
python -m scripts.train_baseline --epochs 2 --max-train-cases 20 --max-val-cases 5 --output-dir outputs/baseline_unet3d_short --device cpu
```

These optional CLI settings override the training configuration. Limits select the first N IDs in each split file, without reordering; limits larger than the split use all available cases. Omitted limits retain the full split. Non-debug short runs retain the default model, optimizer, patch sampling, and deterministic full-volume tiled validation. Debug mode remains a one-epoch diagnostic override.

Startup prints device, effective case counts, epochs, patch size, batch size, trainable parameters, and output directory. History adds `epoch_duration_seconds`, `training_duration_seconds`, and `validation_duration_seconds`; skipped validation has duration zero. Epoch timing includes checkpoint writes but excludes that row's CSV write and console output. Accelerator work is synchronized at measurement boundaries. Final output includes total duration (including setup), average epoch duration, best validation metric and best epoch. The short real run is not launched automatically.

## Presence-aware validation aggregation

For each of NETC/SNFH/ET/RC/WT/TC, validation logs:

- `<region>_case_dice`: arithmetic mean case Dice only where that region has positive target voxels. Empty-target cases are excluded regardless of prediction. With no eligible cases, this is undefined (`None`, blank CSV).
- `<region>_dataset_dice`: `2 * total_intersection / (total_predicted_foreground + total_target_foreground)` over all validation cases. False positives on target-empty cases remain in the denominator. When the region has zero target-positive cases, dataset Dice is unavailable (`None`, blank CSV), regardless of false-positive predictions. When positive cases exist, false positives from empty-target cases still contribute to the pooled denominator.
- `<region>_positive_cases`: number of target-positive validation cases.

Best-checkpoint selection averages the eligible region case Dice equally. It does not use pooled Dice or both-empty successes. If the entire validation set has no positive targets in any region, the selection score is undefined and no best checkpoint is selected; latest checkpoints are still written. Validation loss is unchanged. Historical all-case scores are not directly comparable to this new selection score.

Rerun the 20/5/2 experiment in a fresh directory using `--epochs 2 --max-train-cases 20 --max-val-cases 5 --output-dir outputs/baseline_unet3d_short_presence`.

## Conservative training augmentation

`configs/data.yaml` now enables training-only augmentation after full-volume normalization and patch extraction: independent spatial flips (probability 0.5 per axis), then independent per-modality scaling U[0.9,1.1], shift U[-0.1,0.1], and Gaussian noise with independently sampled per-modality standard deviation U[0,0.05]. Spatial flips apply equally to images, targets and raw segmentation. Intensity changes affect MRI only, using a fixed nonzero mask from the normalized patch after flipping. This preserves all input zeros, including tissue voxels normalized exactly to zero. Target values stay binary; shapes and float32 dtypes are preserved.

The whole pipeline and individual operations can be disabled in config. Direct dataset consumers default to no augmentation unless a configuration is supplied; the training entry point passes the configured augmentation. Validation always ignores augmentation and returns original normalized full volumes. Augmentation uses the existing seeded PyTorch/worker RNG, or an explicitly supplied CPU generator. No rotations, affine/elastic transforms, gamma, or bias fields are included. `crop_start` still locates the pre-augmentation crop in the original volume; flipped tensors are in augmented coordinates.

```bash
python -m scripts.smoke_test_augmentations --data-root ~/datasets/BraTS2024/train/training_data1_v2 --seed 42
```

The smoke test loads one real training patch, reports before/after statistics, and checks binary targets, spatial alignment and preserved zero background. Prior baseline results were obtained without augmentation; new training runs use the enabled augmentation configuration.

## Resume from checkpoint

`--resume PATH` restores model/AdamW state, the last completed epoch, best metric, and best epoch (when stored). The CLI takes model/data and training settings from the checkpoint; explicit data paths, device and epoch overrides are honored subject to compatibility checks. The requested epoch count is the final desired epoch, not an additional count. A checkpoint already at or beyond that epoch exits without modifying files.

```bash
python -m scripts.train_baseline --epochs 1 --max-train-cases 2 --max-val-cases 1 --output-dir outputs/resume_smoke --device mps
python -m scripts.train_baseline --resume outputs/resume_smoke/latest.pt --epochs 2 --device mps
```

Resume appends to the existing history in the checkpoint directory. It checks model config, debug mode, patch size, normalization, augmentation, sampling settings, split paths, case limits, batch size, seed, learning rate, weight decay, threshold, and selection metric. Model loading additionally checks state tensor compatibility. History must have the current columns and contiguous epochs ending at the checkpoint epoch; mismatches fail explicitly without truncating history. Use the latest consistent checkpoint/history pair. Saving checkpoints uses a temporary file followed by replacement to avoid partial checkpoint files.

The previous best metric is preserved; `best.pt` changes only on a strictly better eligible validation score. Older compatible checkpoints without `best_epoch` can recover it from an existing best checkpoint. Resume does not provide bitwise continuation of the sampling stream: RNGs are reseeded, and prior RNG/loader positions are not restored. Backend/device determinism limits still apply. The model, optimizer type, loss, and augmentation definitions are unchanged.

## Research artifacts

Training writes the resolved nested data/model/train configuration to `config.yaml`, environment/version/git metadata to `environment.json`, and final run statistics to `summary.json`. Optional environment fields are null when unavailable. Summary duration accumulates completed invocation durations across resume when a prior summary exists; downtime is excluded. Existing history and checkpoint behavior are unchanged. Compatible resumes archive prior snapshots as `config.before_epoch_N.yaml` and `environment.before_epoch_N.json`; incompatible snapshots fail without overwriting the config.

Final full-validation evaluation is explicit (it is not automatically run after each training invocation):

```bash
python -m scripts.evaluate_baseline --output-dir outputs/baseline_unet3d --device mps
# Optional prediction export for selected validation case IDs:
python -m scripts.evaluate_baseline --output-dir outputs/baseline_unet3d --device mps --save-predictions CASE_ID
```

This defaults to `best.pt`, uses its saved configuration and threshold, and ignores short-run validation limits. `--checkpoint` can explicitly select a checkpoint; artifacts are placed beside it in `evaluation/`. Cases retain split-file order, without shuffling or augmentation. Per-case absent-target Dice is blank, and aggregate unavailable values are JSON null. The reported mean uses the unchanged presence-aware checkpoint selection rule.

Predictions are disabled by default. Selected cases produce four binary NIfTI files in NETC/SNFH/ET/RC order, named by case and tissue, with the reference segmentation's affine and header geometry. Separate masks preserve potentially overlapping sigmoid predictions without introducing argmax or changing metric semantics. Existing evaluation reports are replaced on rerun; previously exported masks are retained.

```text
<output_dir>/
  config.yaml
  environment.json
  history.csv
  summary.json
  best.pt
  latest.pt
  config.before_epoch_N.yaml          # on resume
  environment.before_epoch_N.json     # on resume
  evaluation/
    aggregate_metrics.json
    per_case_metrics.csv
    predictions/                     # only when explicitly requested
      <case>-NETC.nii.gz
      <case>-SNFH.nii.gz
      <case>-ET.nii.gz
      <case>-RC.nii.gz
```
