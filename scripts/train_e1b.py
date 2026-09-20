"""Train E1-B joint reconstruction; or inspect raw loss magnitudes first."""
import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.metrics import VALIDATION_METRIC_FIELDS, best_validation_metric
from evaluation.tiled import validate
from losses.segmentation_loss import SegmentationLoss
from models.joint_reconstruction_unet3d import JointReconstructionUNet3D
from scripts.train_baseline import make_dataset, read_case_ids, read_config, seed_worker, synchronize
from training.artifacts import snapshot, write_json
from training.checkpointing import load_checkpoint, save_checkpoint
from training.e1b_joint import diagnose_loss_magnitudes, train_e1b_epoch


SELECTION_METRIC = 'mean target-positive case Dice over NETC,SNFH,ET,RC,WT,TC; exclude regions with zero positive cases'
HISTORY_FIELDS = ['epoch', 'total_loss', 'segmentation_loss', 'reconstruction_loss',
                  'lambda_rec', 'validation_loss', *VALIDATION_METRIC_FIELDS,
                  'learning_rate', 'epoch_duration_seconds', 'training_duration_seconds',
                  'validation_duration_seconds']


class SegmentationView(nn.Module):
    """Expose only raw E1-A-equivalent logits to existing tiled validation."""

    def __init__(self, joint_model):
        super().__init__()
        self.joint_model = joint_model

    def forward(self, image):
        return self.joint_model.segmentation_only(image)


def select_checkpoint(metrics, best, epoch):
    """Use only the existing segmentation presence-aware selection metric."""
    score = best_validation_metric(metrics)
    improved = score is not None and (best is None or score > best)
    return (score if improved else best), (epoch if improved else None), improved


def _validate_protocol(train, model_config, data, lambda_rec):
    if model_config.get('name') != 'joint_reconstruction_unet3d':
        raise ValueError('E1-B requires joint_reconstruction_unet3d')
    reconstruction = train.get('reconstruction', {})
    if reconstruction.get('enabled') is not True or reconstruction.get('loss_type') != 'L1':
        raise ValueError('E1-B requires enabled L1 reconstruction')
    if train.get('segmentation_loss') != 'Dice+BCE':
        raise ValueError('E1-B must retain Dice+BCE segmentation loss')
    for key, expected in train.get('data_protocol', {}).items():
        if data.get(key) != expected:
            raise ValueError(f'E1-B data protocol mismatch: {key}')
    if lambda_rec is None or not np.isfinite(lambda_rec) or lambda_rec < 0:
        raise ValueError('Set a finite nonnegative lambda_rec explicitly for this run')
    if any(train[key] < 1 for key in ('epochs', 'batch_size', 'validation_frequency')) or train['num_workers'] < 0:
        raise ValueError('Invalid epoch, batch, validation frequency, or worker count')
    if len(read_case_ids(data['train_cases'])) != 1080 or len(read_case_ids(data['val_cases'])) != 270:
        raise ValueError('E1-B requires the fixed 1080/270 train/validation split')


def run_training(train, model_config, data, *, lambda_rec=None, diagnostic_batches=0, resume=None):
    train, model_config = dict(train), dict(model_config)
    selected_lambda = lambda_rec if lambda_rec is not None else train['reconstruction']['lambda_rec']
    _validate_protocol(train, model_config, data, selected_lambda)
    train['reconstruction'] = {**train['reconstruction'], 'lambda_rec': selected_lambda}
    if diagnostic_batches < 0:
        raise ValueError('Diagnostic batch count must be nonnegative')
    output = Path(train['output_dir']).expanduser().resolve()
    frozen = Path('outputs/frozen').resolve()
    protected = (frozen, Path('outputs/e1a_latent_split').resolve(),
                 Path('outputs/baseline_unet3d_full').resolve())
    if any(output == path or path in output.parents for path in protected):
        raise ValueError('E1-B output must be separate from frozen and earlier experiments')

    seed = train['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_data, val_data = make_dataset(data, 'train'), make_dataset(data, 'val')
    if {record['subject_id'] for record in train_data.records} & {record['subject_id'] for record in val_data.records}:
        raise ValueError('Training and validation split IDs overlap')
    train_loader = DataLoader(train_data, batch_size=train['batch_size'], shuffle=True,
                              num_workers=train['num_workers'], worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(seed))
    model = JointReconstructionUNet3D(**{key: value for key, value in model_config.items() if key != 'name'}).to(train['device'])
    criterion = SegmentationLoss()

    if diagnostic_batches:
        rows = diagnose_loss_magnitudes(model, train_loader, criterion, selected_lambda,
                                        diagnostic_batches, train['device'])
        print(json.dumps({'purpose': 'diagnostic loss magnitudes only; no optimizer steps',
                          'lambda_rec': selected_lambda, 'batches': rows}, indent=2))
        return rows

    val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                            num_workers=train['num_workers'], worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(seed + 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train['learning_rate'],
                                  weight_decay=train['weight_decay'])
    config = dict(data=data, model=model_config, train=train,
                  experiment_name='E1-B joint MRI reconstruction ablation',
                  selection_metric=SELECTION_METRIC)
    state = None
    if resume:
        resume_path = Path(resume).expanduser().resolve()
        if resume_path.parent != output:
            raise ValueError('Resume checkpoint must be in the E1-B output directory')
        state = torch.load(resume_path, map_location='cpu', weights_only=True)
        saved = state['config']
        if (saved['data'] != data or saved['model'] != model_config or
            {k: v for k, v in saved['train'].items() if k != 'epochs'} !=
            {k: v for k, v in train.items() if k != 'epochs'}):
            raise ValueError('E1-B resume configuration mismatch')
        if state['epoch'] >= train['epochs']:
            return None
        load_checkpoint(resume_path, model, optimizer)

    history = output / 'history.csv'
    if state is None and any((output / name).exists() for name in ('history.csv', 'best.pt', 'latest.pt')):
        raise FileExistsError(f'E1-B output already contains a run: {output}')
    if state is not None and not history.is_file():
        raise FileNotFoundError(f'E1-B resume history missing: {history}')
    output.mkdir(parents=True, exist_ok=True)
    snapshot(output, config, sum(p.numel() for p in model.parameters() if p.requires_grad),
             state['epoch'] if state else None)
    best = state.get('best_metric') if state else None
    best_epoch = state.get('best_epoch') if state else None
    start_epoch = state['epoch'] + 1 if state else 1
    previous_duration = 0.0
    if state and (output / 'summary.json').exists():
        previous_duration = json.loads((output / 'summary.json').read_text()).get('total_run_duration', 0.0)
    started = time.perf_counter()
    with history.open('a' if state else 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        if not state:
            writer.writeheader()
        for epoch in range(start_epoch, train['epochs'] + 1):
            synchronize(train['device'])
            epoch_started = time.perf_counter()
            losses = train_e1b_epoch(model, train_loader, optimizer, criterion, selected_lambda, train['device'])
            synchronize(train['device'])
            training_duration = time.perf_counter() - epoch_started
            row = dict(epoch=epoch, **losses, learning_rate=optimizer.param_groups[0]['lr'])
            validation_duration = 0.0
            if epoch % train['validation_frequency'] == 0 or epoch == train['epochs']:
                validation_started = time.perf_counter()
                val_loss, metrics = validate(SegmentationView(model), val_loader, criterion,
                                             train['device'], tuple(data['patch_size']), train['threshold'])
                synchronize(train['device'])
                validation_duration = time.perf_counter() - validation_started
                row.update(validation_loss=val_loss, **metrics)
                updated, selected_epoch, improved = select_checkpoint(metrics, best, epoch)
                if improved:
                    best, best_epoch = updated, selected_epoch
                    save_checkpoint(output / 'best.pt', model, optimizer, epoch, config, best, seed, best_epoch)
            save_checkpoint(output / 'latest.pt', model, optimizer, epoch, config, best, seed, best_epoch)
            duration = time.perf_counter() - epoch_started
            row.update(epoch_duration_seconds=duration, training_duration_seconds=training_duration,
                       validation_duration_seconds=validation_duration)
            writer.writerow(row)
            stream.flush()
            print(row, flush=True)
    write_json(output / 'summary.json', dict(total_epochs_completed=epoch, best_epoch=best_epoch,
               best_validation_metric=best, final_total_loss=row['total_loss'],
               final_segmentation_loss=row['segmentation_loss'],
               final_reconstruction_loss=row['reconstruction_loss'], lambda_rec=selected_lambda,
               train_case_count=len(train_data), validation_case_count=len(val_data),
               total_run_duration=previous_duration + time.perf_counter() - started))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/train_e1b.yaml')
    parser.add_argument('--model-config', default='configs/model_e1b.yaml')
    parser.add_argument('--lambda-rec', type=float)
    parser.add_argument('--diagnostic-batches', type=int, default=0)
    parser.add_argument('--device')
    parser.add_argument('--output-dir')
    parser.add_argument('--resume')
    args = parser.parse_args()
    train = read_config(args.config)
    model_config = read_config(args.model_config)
    data = read_config(train['data_config'])
    if args.device:
        train['device'] = args.device
    if args.output_dir:
        train['output_dir'] = args.output_dir
    run_training(train, model_config, data, lambda_rec=args.lambda_rec,
                 diagnostic_batches=args.diagnostic_batches, resume=args.resume)


if __name__ == '__main__':
    main()
