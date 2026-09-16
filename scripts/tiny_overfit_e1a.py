"""DEVELOPMENT/SANITY: memorize one fixed real training patch, not paper results."""
import argparse
import csv
import hashlib
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from datasets.brats_dataset import BraTSDataset
from datasets.label_mapping import TISSUE_NAMES
from losses.segmentation_loss import SegmentationLoss
from models.latent_split_unet3d import LatentSplitUNet3D
from scripts.train_baseline import read_case_ids
from training.artifacts import environment_metadata, write_json
from training.checkpointing import save_checkpoint

PATCH_SIZE = (96, 96, 96)
EXPECTED_PARAMETERS = 5_971_716
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.00001


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Fingerprint contiguous tensor bytes; shape/dtype are recorded separately."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def cache_patch(data_root: str, train_cases: str, seed: int) -> dict[str, Any]:
    """Load the first split ID exactly once; fail if it cannot supply foreground.

    No search over validation data or multiple candidate crops. The existing
    sampler keeps the chosen foreground voxel in the crop, including its jitter.
    """
    ids = read_case_ids(train_cases)
    dataset = BraTSDataset(root=data_root, normalize=True, training=True,
                           patch_size=PATCH_SIZE, foreground_sampling_probability=1.0,
                           generator=torch.Generator().manual_seed(seed), augmentation={'enabled': False})
    records = {record['subject_id']: record for record in dataset.records}
    missing = set(ids) - records.keys()
    if missing:
        raise ValueError(f'Unknown frozen training split IDs: {sorted(missing)}')
    dataset.records = [records[ids[0]]]
    sample = dataset[0]
    if sample['case_id'] != ids[0]:
        raise ValueError('Loaded case differs from selected frozen training ID.')
    image, target = sample['image'].clone(), sample['target'].clone()
    if image.shape != (4, *PATCH_SIZE) or target.shape != image.shape:
        raise ValueError('Expected a single aligned [4,96,96,96] patch.')
    if image.dtype != torch.float32 or target.dtype != torch.float32:
        raise ValueError('Expected float32 image and target.')
    if not torch.isfinite(image).all() or not ((target == 0) | (target == 1)).all():
        raise ValueError('Expected finite image and binary target.')
    if not target.bool().any():
        raise ValueError('First training case has no foreground; cannot satisfy foreground-centered protocol.')
    return dict(case_id=sample['case_id'], crop_start=list(sample['crop_start']),
                image=image.unsqueeze(0), target=target.unsqueeze(0))


def check_gradients(model: LatentSplitUNet3D) -> dict[str, bool]:
    """Require finite, nonzero task gradients; anatomy must have no gradient."""
    checks = {}
    for name, module in (('encoder', model.encoder), ('disease_projector', model.projection.disease_projector),
                         ('segmentation_decoder', model.decoder)):
        gradients = [p.grad for p in module.parameters()]
        checks[name] = (all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
                        and any(g is not None and bool(g.abs().sum() > 0) for g in gradients))
    checks['anatomy_projector_has_no_task_gradient'] = all(p.grad is None for p in model.projection.anatomy_projector.parameters())
    if not all(checks.values()):
        raise RuntimeError(f'E1-A segmentation gradient structure violated: {checks}')
    return checks


def verify_shapes(result: Any) -> dict[str, list[int]]:
    shapes = {name: list(getattr(result, name).shape) for name in ('logits', 'anatomy_latent', 'disease_latent')}
    if shapes != dict(logits=[1,4,96,96,96], anatomy_latent=[1,256,12,12,12], disease_latent=[1,256,12,12,12]):
        raise ValueError(f'Unexpected E1-A tensor shapes: {shapes}')
    return shapes


def optimize_fixed_patch(model: LatentSplitUNet3D, image: torch.Tensor, target: torch.Tensor,
                         optimizer: torch.optim.Optimizer, criterion: SegmentationLoss,
                         steps: int, history_path: Path) -> tuple[list[float], dict[str, bool]]:
    """Reuse supplied tensors by identity; history loss is before each update."""
    if steps < 1:
        raise ValueError('steps must be positive.')
    losses, checks = [], {}
    with history_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'loss', 'learning_rate'])
        writer.writeheader()
        for step in range(1, steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at step {step}.')
            loss.backward()
            if step == 1:
                checks = check_gradients(model)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f'Non-finite gradients at step {step}.')
            optimizer.step()
            value = loss.item()
            losses.append(value)
            writer.writerow(dict(step=step, loss=value, learning_rate=optimizer.param_groups[0]['lr']))
            stream.flush()
            print(f'step={step} loss={value:.6f} learning_rate={optimizer.param_groups[0]["lr"]}', flush=True)
    return losses, checks


def run(data_root: str, train_cases: str, device: str, steps: int, seed: int, output_dir: str) -> dict[str, Any]:
    if steps < 1:
        raise ValueError('steps must be positive.')
    output = Path(output_dir).expanduser().resolve()
    # Protect experiment roots even when their directories have not been created.
    for root in ('outputs/baseline_unet3d_full', 'outputs/e1a_latent_split'):
        protected = Path(root).resolve()
        if output == protected or protected in output.parents:
            raise ValueError('Use a separate DEVELOPMENT/SANITY output directory.')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite a nonempty sanity output directory.')
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sample = cache_patch(data_root, train_cases, seed)
    image, target = sample.pop('image').to(device), sample.pop('target').to(device)
    identities = dict(image_sha256=tensor_sha256(image), target_sha256=tensor_sha256(target))
    model = LatentSplitUNet3D(in_channels=4, out_channels=4, base_channels=32).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise ValueError(f'E1-A parameter count mismatch: {parameter_count} != {EXPECTED_PARAMETERS}')
    criterion = SegmentationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    metadata = environment_metadata(parameter_count)
    counts = target.sum((0,2,3,4)).tolist()
    config = dict(artifact_type='DEVELOPMENT/SANITY; not a paper experiment result', **sample,
                  data_root=str(Path(data_root).expanduser().resolve()), train_cases=str(Path(train_cases).expanduser().resolve()),
                  train_split_sha256=hashlib.sha256(Path(train_cases).expanduser().read_bytes()).hexdigest(),
                  seed=seed, device=device, steps=steps, patch_size=list(PATCH_SIZE), augmentation=False,
                  foreground_sampling_probability=1.0, normalize=True, **identities,
                  model=dict(name='latent_split_unet3d', base_channels=32, in_channels=4, out_channels=4,
                             latent_projection='conv1x1x1', latent_channels=256), model_parameter_count=parameter_count,
                  optimizer='AdamW', learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                  loss=dict(name='SegmentationLoss', dice_weight=1., bce_weight=1., dice_smooth=1e-5),
                  environment=metadata, history_loss_semantics='before each optimizer step; final_loss after last update',
                  determinism='Seeded Python/NumPy/PyTorch and deterministic cuDNN; exact reproducibility depends on backend/device.')
    write_json(output/'config.json', config)
    model.train()
    with torch.no_grad():
        result = model.forward_with_latents(image)
        shapes = verify_shapes(result)
        initial = criterion(result.logits, target).item()
    del result
    if not np.isfinite(initial):
        raise RuntimeError('Non-finite initial loss.')
    save_checkpoint(output/'initial.pt', model, optimizer, 0, config, seed=seed)
    losses, checks = optimize_fixed_patch(model, image, target, optimizer, criterion, steps, output/'history.csv')
    with torch.no_grad():
        final = criterion(model(image), target).item()
    if not np.isfinite(final):
        raise RuntimeError('Non-finite final loss.')
    if identities != dict(image_sha256=tensor_sha256(image), target_sha256=tensor_sha256(target)):
        raise RuntimeError('Fixed image/target tensors changed during optimization.')
    save_checkpoint(output/'final.pt', model, optimizer, steps, config, seed=seed)
    reduction = initial - final
    summary = dict(**sample, artifact_type=config['artifact_type'], seed=seed, patch_size=list(PATCH_SIZE),
                   foreground_voxel_count=int(target.bool().any(1).sum()),
                   target_voxel_counts=dict(zip(TISSUE_NAMES, (int(c) for c in counts))),
                   target_positive_channels=[name for name, count in zip(TISSUE_NAMES, counts) if count > 0],
                   target_positive_regions=[name for name, present in zip((*TISSUE_NAMES, 'WT', 'TC'),
                       [*(c > 0 for c in counts), sum(counts[:3]) > 0, counts[0]+counts[2] > 0]) if present],
                   model_parameter_count=parameter_count, steps=steps, optimizer='AdamW', learning_rate=LEARNING_RATE,
                   weight_decay=WEIGHT_DECAY, initial_loss=initial, final_loss=final, minimum_loss=min(initial, final, *losses),
                   absolute_loss_reduction=reduction, relative_loss_reduction=reduction/initial if initial else None,
                   relative_loss_reduction_units='fraction; null if initial loss is zero', gradient_checks=checks,
                   tensor_shapes={**shapes, 'image':list(image.shape), 'target':list(target.shape)}, fixed_tensor_hashes=identities,
                   duration_seconds=time.perf_counter()-started, git_commit=metadata['git_commit'], git_dirty_status=metadata['git_dirty'])
    write_json(output/'summary.json', summary)
    print(f'Initial loss={initial:.6f}; final loss={final:.6f}; reduction={reduction:.6f}. No scientific pass threshold.', flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--train-cases', default='splits/train_cases.txt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='outputs/e1a_tiny_overfit_sanity')
    args = parser.parse_args()
    try:
        run(**vars(args))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'E1-A tiny-overfit sanity test failed: {error}\n')


if __name__ == '__main__':
    main()
