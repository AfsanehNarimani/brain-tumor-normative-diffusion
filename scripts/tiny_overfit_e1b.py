"""DEVELOPMENT/SANITY: optimize E1-B on one fixed real BraTS training patch."""
import argparse
import csv
import math
from pathlib import Path
import random

import numpy as np
import torch

from losses.segmentation_loss import SegmentationLoss
from models.joint_reconstruction_unet3d import JointReconstructionUNet3D
from scripts.tiny_overfit_e1a import cache_patch, tensor_sha256
from training.artifacts import write_json
from training.e1b_joint import compute_joint_loss, train_e1b_epoch


LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.00001
DEFAULT_OUTPUT = 'outputs/e1b_tiny_overfit'


def _loss_row(model, batch, criterion, lambda_rec, step):
    model.train()
    with torch.no_grad():
        losses = compute_joint_loss(model(batch['image']), batch['image'], batch['target'],
                                    criterion, lambda_rec)
    values = {name: getattr(losses, name).item() for name in
              ('total_loss', 'segmentation_loss', 'reconstruction_loss')}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(f'Non-finite E1-B sanity loss at step {step}: {values}')
    return dict(step=step, **values,
                weighted_reconstruction_loss=lambda_rec * values['reconstruction_loss'],
                lambda_rec=lambda_rec)


def optimize_fixed_sample(model, batch, optimizer, criterion, lambda_rec, steps, device):
    """Use the production E1-B update on the same one-batch iterable every step.

    Trajectory rows are measured before training (step 0) and after each update.
    Gradient hooks only validate finite values; they return gradients unchanged.
    """
    if steps < 1:
        raise ValueError('steps must be positive')
    if not math.isfinite(lambda_rec) or lambda_rec < 0:
        raise ValueError('lambda_rec must be finite and nonnegative')
    hooks = []

    def check_gradient(gradient):
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError('Non-finite E1-B sanity gradient')
        return gradient

    for parameter in model.parameters():
        if parameter.requires_grad:
            hooks.append(parameter.register_hook(check_gradient))
    try:
        rows = [_loss_row(model, batch, criterion, lambda_rec, 0)]
        for step in range(1, steps + 1):
            train_e1b_epoch(model, [batch], optimizer, criterion, lambda_rec, device)
            rows.append(_loss_row(model, batch, criterion, lambda_rec, step))
            print(f"step={step} total={rows[-1]['total_loss']:.6f} "
                  f"seg={rows[-1]['segmentation_loss']:.6f} "
                  f"rec={rows[-1]['reconstruction_loss']:.6f}", flush=True)
        return rows
    finally:
        for hook in hooks:
            hook.remove()


def run(data_root, train_cases='splits/train_cases.txt', device='cpu', steps=50,
        seed=42, lambda_rec=1.0, output_dir=DEFAULT_OUTPUT):
    output = Path(output_dir).expanduser().resolve()
    protected = [Path(path).resolve() for path in
                 ('outputs/e1b_joint_reconstruction', 'outputs/frozen')]
    if any(output == path or path in output.parents for path in protected):
        raise ValueError('Sanity output must be separate from experiment and frozen outputs')
    if steps < 1:
        raise ValueError('steps must be positive')
    if not math.isfinite(lambda_rec) or lambda_rec < 0:
        raise ValueError('lambda_rec must be finite and nonnegative')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite sanity output: {output}')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sample = cache_patch(data_root, train_cases, seed)
    batch = {key: sample[key].to(device) for key in ('image', 'target')}
    hashes = {key: tensor_sha256(value) for key, value in batch.items()}
    model = JointReconstructionUNet3D(in_channels=4, out_channels=4, base_channels=32).to(device)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    criterion = SegmentationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rows = optimize_fixed_sample(model, batch, optimizer, criterion, lambda_rec, steps, device)
    if hashes != {key: tensor_sha256(value) for key, value in batch.items()}:
        raise RuntimeError('Fixed E1-B sanity sample changed during optimization')
    parameter_changed = any(not torch.equal(old, current) for old, current in zip(before, model.parameters()))
    if not parameter_changed:
        raise RuntimeError('E1-B sanity optimizer did not update model parameters')
    summary = dict(artifact_type='DEVELOPMENT/SANITY; not a paper experiment result',
                   case_id=sample['case_id'], crop_start=sample['crop_start'], seed=seed,
                   number_of_steps=steps, lambda_rec=lambda_rec, optimizer='AdamW',
                   learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                   fixed_tensor_sha256=hashes, parameter_changed=parameter_changed,
                   all_losses_finite=True, all_gradients_finite=True,
                   initial_total_loss=rows[0]['total_loss'], final_total_loss=rows[-1]['total_loss'],
                   initial_segmentation_loss=rows[0]['segmentation_loss'],
                   final_segmentation_loss=rows[-1]['segmentation_loss'],
                   initial_reconstruction_loss=rows[0]['reconstruction_loss'],
                   final_reconstruction_loss=rows[-1]['reconstruction_loss'],
                   total_loss_decreased=rows[-1]['total_loss'] < rows[0]['total_loss'],
                   trajectory_semantics='step 0 before optimization; later steps after each optimizer update')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'history.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'summary.json', summary)
    if not summary['total_loss_decreased']:
        raise RuntimeError('E1-B sanity joint objective did not decrease on the fixed sample')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--train-cases', default='splits/train_cases.txt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda-rec', type=float, default=1.0)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == '__main__':
    main()
