"""Train the D0-C0 latent diffusion model under the frozen D0-B4 protocol.

No training starts on import. Scientific mode requires CUDA and a new output
folder (or an exact-provenance epoch-boundary resume). ``--smoke`` is explicitly
non-scientific, uses synthetic CPU tensors, and writes only smoke_summary.json
under a separate smoke directory.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.transforms import normalize_nonzero
from models.latent_diffusion_unet3d import LatentDiffusionUNet3D, cosine_schedule, masked_epsilon_mse
from models.latent_probe import E1C_SOURCE_SHA256, file_sha256, load_source
from scripts.audit_d0b1_abnormal_occupancy import abnormal_mask, block_occupancy, load_case, support_mask
from scripts.audit_d0b3_corruption_operator import corrupt_input, donor_pools
from scripts.freeze_d0b4d_validation_manifest import verify as verify_b4d
from scripts.train_baseline import read_case_ids, read_config

FINAL_SHA256 = '05d13292994b59f3c45d70c98591c306970269673e475797620c644446aa86d8'
B4C_SHA256 = '861ea070f7bfa2677b9345403905e70400190012e56efac722ccbb892af21f75'
B4B_SHA256 = '29cf659528f6257cd9f8f218e9f4eaf8fabdc6d8a0e4905505aa8548ae68bde4'
VAL_MANIFEST_SHA256 = '2a3a10d0d7b6b34bc412cfee5e30f45912161528ff509e69193f3516af3baee4'
VAL_AUDIT_SHA256 = 'fe9b055b64c51fa1dafddbe6cb260c73310ef7468ef10ccb6e80477a3cdf2257'
TRAIN_SHA256 = '581dc6326a22cb0b9582fa249b8e13fd948597a60c5f454cb5ade0493fa7c18e'
VAL_SHA256 = 'ec5fe6b0bbcf190fe8b7c032f8da4ff1d495969c2d868561e7a3351c572cc5ca'
D0A_SHA256 = 'bca5c0f8f1617d73fc4d0850fe5b6f45e4b2bcbacbb4018f1d5d825de49ad792'
D0B0_SHA256 = '9c4ba67fc195537a5500229722ce55e4772ea5b7d135b351b3a2fc4b857f44b9'
D0B3_SHA256 = '095e1f9671411eca0d12a2f834833d78477a14664504bde3e6e1b46766c6cc2f'
B4A_SHA256 = '07c268d97c7a39f389548344be7245cee4eba3a390a91d29b6533ab466c5254b'
DEFAULT_CONFIG = ROOT / 'configs/data.yaml'
DEFAULT_CONTRACT = ROOT / 'configs/d0b4c_training_validation_contract.json'
DEFAULT_FINAL_FREEZE = ROOT / 'configs/d0b4_final_design_freeze.json'
DEFAULT_B4B = ROOT / 'configs/d0b4b_inference_contract.json'
DEFAULT_VAL_MANIFEST = ROOT / 'outputs/frozen/d0b4d_validation_manifest/validation_manifest.csv'
DEFAULT_VAL_AUDIT = DEFAULT_VAL_MANIFEST.with_name('validation_manifest_audit.csv')
DEFAULT_E1C = ROOT / 'outputs/frozen/e1c_anatomy_constraint/training/best.pt'
DEFAULT_D0A = ROOT / 'outputs/frozen/d0_latent_distribution_audit/d0_za_normalization_contract.json'
DEFAULT_D0B3 = ROOT / 'outputs/frozen/d0b3_corruption_operator_audit/decision/corruption_operator_decision.json'
DEFAULT_D0B0 = ROOT / 'outputs/frozen/d0b_support_mask_audit/patch_manifest.csv'
DEFAULT_B4A = ROOT / 'outputs/frozen/d0b4a_diffusion_memory_preflight/decision/microbatch_decision.json'
DEFAULT_TRAIN_SPLIT = ROOT / 'splits/train_cases.txt'
DEFAULT_VAL_SPLIT = ROOT / 'splits/val_cases.txt'
DEFAULT_OUTPUT = ROOT / 'outputs/d0_latent_diffusion'
DEFAULT_SMOKE_OUTPUT = ROOT / 'outputs/d0_latent_diffusion_smoke'
GPU_SMOKE_BATCH_SIZE = 8
GPU_SMOKE_DEFAULT_TRAIN_BATCHES = 2
GPU_SMOKE_DEFAULT_VALIDATION_PATCHES = 8
GPU_SMOKE_FORBIDDEN_FILES = ('best.pt', 'latest.pt', 'training_history.csv',
    'validation_per_case.csv', 'training_provenance.json', 'validation_summary.json')
LATENT_SHAPE = (256, 12, 12, 12)
RAW_SHAPE = (4, 96, 96, 96)
HISTORY_FIELDS = ('epoch_index', 'human_epoch', 'samples_presented', 'successful_optimizer_steps',
    'mean_training_loss', 'global_grad_norm_mean', 'global_grad_norm_max', 'gradscaler_skipped_steps',
    'learning_rate', 'elapsed_seconds', 'donor_fallback_modality_count', 'primary_validation_metric',
    'secondary_global_support_weighted_mse', 'is_best')
CASE_FIELDS = ('human_epoch', 'epoch_index', 'case_id', 'patch_count', 'case_mean_epsilon_mse')
VAL_FIELDS = ('case_id', 'validation_patch_index', 'crop_start_d', 'crop_start_h', 'crop_start_w',
    'crop_stop_d', 'crop_stop_h', 'crop_stop_w', 'coordinate_generation_version', 'seed', 'split_sha256',
    'timestep', 'timestep_seed', 'epsilon_seed', 'corruption_operator', 'corruption_global_seed',
    'corruption_patch_index')
CROP_FIELDS = ('case_id', 'epoch_index', 'human_epoch', 'crop_start_d', 'crop_start_h',
    'crop_start_w', 'crop_stop_d', 'crop_stop_h', 'crop_stop_w',
    'coordinate_generation_version', 'training_global_seed', 'case_order_position')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def read_sha(path, expected, label):
    actual = file_sha256(path)
    _require(actual == expected, f'{label} SHA256 mismatch: {actual} != {expected}')
    return actual


def verify_startup(*, final_freeze=DEFAULT_FINAL_FREEZE, contract_path=DEFAULT_CONTRACT,
                   inference_contract=DEFAULT_B4B, manifest=DEFAULT_VAL_MANIFEST,
                   audit=DEFAULT_VAL_AUDIT, train_split=DEFAULT_TRAIN_SPLIT,
                   val_split=DEFAULT_VAL_SPLIT, e1c=DEFAULT_E1C, d0a=DEFAULT_D0A,
                   d0b3=DEFAULT_D0B3, d0b0=DEFAULT_D0B0, b4a=DEFAULT_B4A):
    """Fail closed on the complete frozen provenance chain before any run."""
    read_sha(final_freeze, FINAL_SHA256, 'D0-B4 final freeze')
    read_sha(contract_path, B4C_SHA256, 'D0-B4c contract')
    read_sha(inference_contract, B4B_SHA256, 'D0-B4b contract')
    read_sha(manifest, VAL_MANIFEST_SHA256, 'D0-B4d validation manifest')
    read_sha(audit, VAL_AUDIT_SHA256, 'D0-B4d validation audit')
    read_sha(train_split, TRAIN_SHA256, 'training split')
    read_sha(val_split, VAL_SHA256, 'validation split')
    read_sha(e1c, E1C_SOURCE_SHA256, 'E1-C checkpoint')
    read_sha(d0a, D0A_SHA256, 'D0-A normalization contract')
    read_sha(d0b3, D0B3_SHA256, 'D0-B3 decision')
    d0b0_sha = read_sha(d0b0, D0B0_SHA256, 'D0-B0 patch manifest')
    b4a_sha = read_sha(b4a, B4A_SHA256, 'D0-B4a decision')
    freeze = json.loads(Path(final_freeze).read_text())
    contract = json.loads(Path(contract_path).read_text())
    b4b = json.loads(Path(inference_contract).read_text())
    _require(freeze['status'] == 'COMPLETE_DESIGN_FROZEN_PRE_TRAINING'
             and freeze['training_start_gate']['scientific_diffusion_training_allowed'] is True
             and freeze['training_start_gate']['all_prerequisites_frozen_and_verified'] is True,
             'D0-B4 final freeze training gate is not satisfied')
    _require(freeze['d0b4a']['sha256'] == B4A_SHA256 and freeze['d0b4a']['sha256_verified'] is True,
             'Final freeze D0-B4a reference mismatch')
    _require(freeze['root_scientific_provenance']['d0b0_manifest']['sha256'] == D0B0_SHA256
             and freeze['root_scientific_provenance']['d0b0_manifest']['sha256_verified'] is True,
             'Final freeze D0-B0 reference mismatch')
    for stage, sha in (('d0b4b', B4B_SHA256), ('d0b4c', B4C_SHA256)):
        _require(freeze[stage]['sha256'] == sha and freeze[stage]['sha256_verified'] is True,
                 f'Final freeze {stage} reference mismatch')
    validation = contract['validation']
    for key, expected in (('case_count', 270), ('patches_per_case', 4), ('total_patches', 1080),
                          ('validation_global_seed', 20260923)):
        _require(validation[key] == expected, f'D0-B4c validation {key} mismatch')
    _require(contract['validation_interpretation']['split_role'] == 'DEVELOPMENT_VALIDATION',
             'Validation split is not DEVELOPMENT_VALIDATION')
    _require(b4b['partial_noise']['primary_timestep'] == 495, 'D0-B4b primary timestep mismatch')
    _require(file_sha256(DEFAULT_VAL_MANIFEST.parent / 'MANIFEST.sha256') == freeze['d0b4d']['manifest_sha256'],
             'D0-B4d MANIFEST.sha256 file digest mismatch')
    summary = json.loads((DEFAULT_VAL_MANIFEST.parent / 'validation_manifest_summary.json').read_text())
    for key, expected in (('case_count', 270), ('patches_per_case', 4), ('total_patches', 1080),
                          ('zero_support_patch_count', 0), ('validation_role', 'DEVELOPMENT_VALIDATION')):
        _require(summary[key] == expected, f'D0-B4d {key} mismatch')
    ids, val_ids = read_case_ids(train_split), read_case_ids(val_split)
    _require(len(ids) == 1080 and len(val_ids) == 270 and not set(ids) & set(val_ids),
             'Frozen split count or disjointness mismatch')
    # Full manifest, audit CSV, summary, provenance, and manifest inventory audit.
    verify_b4d(DEFAULT_VAL_MANIFEST.parent)
    d0a_payload = json.loads(Path(d0a).read_text())
    _require(d0a_payload['channel_count'] == 256 and len(d0a_payload['channels']) == 256,
             'D0-A channel contract mismatch')
    means = torch.tensor([c['mean'] for c in d0a_payload['channels']], dtype=torch.float32)
    stds = torch.tensor([c['population_std'] for c in d0a_payload['channels']], dtype=torch.float32)
    _require(torch.isfinite(means).all().item() and torch.isfinite(stds).all().item()
             and (stds > 0).all().item(), 'Invalid D0-A standardization values')
    return dict(final_freeze_sha256=FINAL_SHA256, b4c_contract_sha256=B4C_SHA256,
        b4b_contract_sha256=B4B_SHA256, validation_manifest_sha256=VAL_MANIFEST_SHA256,
        validation_manifest_audit_sha256=VAL_AUDIT_SHA256,
        training_split_sha256=TRAIN_SHA256, validation_split_sha256=VAL_SHA256,
        e1c_checkpoint_sha256=E1C_SOURCE_SHA256, d0a_contract_sha256=D0A_SHA256,
        d0b3_decision_sha256=D0B3_SHA256, validation_case_count=270,
        d0b4d_manifest_sha256=freeze['d0b4d']['manifest_sha256'],
        d0b0_artifact_path=str(Path(d0b0).expanduser().resolve()), d0b0_artifact_sha256=d0b0_sha,
        b4a_decision_path=str(Path(b4a).expanduser().resolve()), b4a_decision_sha256=b4a_sha,
        validation_patches_per_case=4, validation_rows=1080,
        validation_role='DEVELOPMENT_VALIDATION', d0a_means=means, d0a_stds=stds,
        train_ids=ids, validation_ids=val_ids, contract=contract, freeze=freeze)


def stable_payload(values):
    return json.dumps(values, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def hash_integer(values):
    return int.from_bytes(hashlib.sha256(stable_payload(values)).digest(), 'big', signed=False)


def generator_seed(values):
    return int.from_bytes(hashlib.sha256(stable_payload(values)).digest()[:8], 'big', signed=False) & ((1 << 63) - 1)


def deterministic_case_order(case_ids, epoch_index, seed=20260923):
    if len(case_ids) != len(set(case_ids)):
        raise ValueError('Training case IDs must be unique')
    return [case for _, case in sorted((hashlib.sha256(stable_payload(
        [seed, int(epoch_index), case, 'training_case_order'])).hexdigest(), case) for case in case_ids)]


def deterministic_crop(case_id, epoch_index, dimensions, seed=20260923, patch_shape=(96, 96, 96)):
    if len(dimensions) != 3 or any(int(n) < p for n, p in zip(dimensions, patch_shape)):
        raise ValueError(f'Case {case_id}: volume smaller than required crop {patch_shape}: {dimensions}')
    row = {'case_id': case_id, 'epoch_index': int(epoch_index)}
    for axis, size, patch in zip('dhw', dimensions, patch_shape):
        start = hash_integer([seed, int(epoch_index), case_id, axis, 'training_crop']) % (int(size) - patch + 1)
        row[f'crop_start_{axis}'] = start
        row[f'crop_stop_{axis}'] = start + patch
    return row


def training_epoch_crop_rows(case_ids, epoch_index, dimensions_by_case, seed=20260923):
    """Build one coordinate record per case from the exact training sampler."""
    ordered = deterministic_case_order(case_ids, epoch_index, seed)
    rows = []
    for position, case_id in enumerate(ordered):
        if case_id not in dimensions_by_case:
            raise ValueError(f'Missing loaded spatial dimensions for training case {case_id}')
        row = deterministic_crop(case_id, epoch_index, dimensions_by_case[case_id], seed)
        rows.append(dict(case_id=case_id, epoch_index=int(epoch_index), human_epoch=int(epoch_index) + 1,
            **{key: row[key] for key in CROP_FIELDS if key.startswith('crop_')},
            coordinate_generation_version=1, training_global_seed=int(seed),
            case_order_position=position))
    return rows


def _canonical_csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows({field: row.get(field, '') for field in fields} for row in rows)
    return stream.getvalue().encode('utf-8')


def _read_crop_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CROP_FIELDS):
            raise ValueError('Training crop manifest schema mismatch')
        rows = list(reader)
    keys = [(int(row['epoch_index']), row['case_id']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate (epoch_index, case_id) in training crop manifest')
    positions = [(int(row['epoch_index']), int(row['case_order_position'])) for row in rows]
    if positions != sorted(positions):
        raise ValueError('Training crop manifest rows are not in canonical order')
    return rows


def _crop_row_strings(rows):
    return [{field: str(row[field]) for field in CROP_FIELDS} for row in rows]


def ensure_training_crop_epoch(path, expected_rows):
    """Atomically add one complete epoch, or verify its existing exact rows."""
    path = Path(path)
    expected = _crop_row_strings(expected_rows)
    if len(expected) != 1080 or len({row['case_id'] for row in expected}) != 1080:
        raise ValueError('Training crop epoch must have exactly 1080 unique cases')
    epoch = int(expected[0]['epoch_index'])
    if any(int(row['epoch_index']) != epoch for row in expected):
        raise ValueError('Expected training crop rows span multiple epochs')
    existing = _read_crop_rows(path)
    grouped = {}
    for row in existing:
        grouped.setdefault(int(row['epoch_index']), []).append(row)
    for old_epoch, old_rows in grouped.items():
        if len(old_rows) != 1080:
            raise ValueError(f'Incomplete pre-existing crop manifest epoch {old_epoch}')
        if old_epoch > epoch:
            raise ValueError(f'Unexpected future crop manifest epoch {old_epoch}')
    if any(old_epoch not in grouped for old_epoch in range(epoch)):
        raise ValueError('Training crop manifest is missing a prior epoch')
    if epoch in grouped:
        if grouped[epoch] != expected:
            raise ValueError(f'Pre-existing training crop rows mismatch for epoch_index={epoch}')
        return hashlib.sha256(path.read_bytes()).hexdigest(), len(existing)
    if grouped and max(grouped) != epoch - 1:
        raise ValueError('Training crop manifest epochs are not contiguous')
    updated = existing + expected
    data = _canonical_csv_bytes(updated, CROP_FIELDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return hashlib.sha256(data).hexdigest(), len(updated)


def verify_training_crop_resume(path, *, completed_epoch_index, checkpoint_sha256,
                                checkpoint_row_count, expected_next_rows=None):
    rows = _read_crop_rows(path)
    completed = [row for row in rows if int(row['epoch_index']) <= completed_epoch_index]
    expected_count = (completed_epoch_index + 1) * 1080
    if len(completed) != expected_count or checkpoint_row_count != expected_count:
        raise ValueError('Training crop manifest row count disagrees with completed checkpoint epoch')
    prefix_sha = hashlib.sha256(_canonical_csv_bytes(completed, CROP_FIELDS)).hexdigest()
    if prefix_sha != checkpoint_sha256:
        raise ValueError('Training crop manifest prefix SHA disagrees with checkpoint')
    later = [row for row in rows if int(row['epoch_index']) > completed_epoch_index]
    if later:
        if expected_next_rows is None or [dict(row) for row in later] != _crop_row_strings(expected_next_rows):
            raise ValueError('Unexpected or mismatched prewritten next-epoch crop rows')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest(), len(rows)


def sample_identity(row):
    return [row[k] for k in ('case_id', 'epoch_index', 'crop_start_d', 'crop_start_h', 'crop_start_w',
                              'crop_stop_d', 'crop_stop_h', 'crop_stop_w')]


def sample_randomness(row, seed=20260923, shape=LATENT_SHAPE):
    identity = sample_identity(row)
    t_seed = generator_seed([seed, identity, 'training_timestep'])
    e_seed = generator_seed([seed, identity, 'training_epsilon'])
    timestep = int(torch.randint(0, 1000, (1,), generator=torch.Generator(device='cpu').manual_seed(t_seed)).item())
    epsilon = torch.randn(shape, dtype=torch.float32, device='cpu',
                          generator=torch.Generator(device='cpu').manual_seed(e_seed))
    return timestep, epsilon, t_seed, e_seed


def forward_diffuse(z0, timesteps, epsilon, support, schedule):
    if z0.ndim != 5 or z0.shape[1:] != LATENT_SHAPE:
        raise ValueError('Expected z0 [B,256,12,12,12]')
    if support.shape != (z0.shape[0], 1, *z0.shape[2:]) or epsilon.shape != z0.shape:
        raise ValueError('Incompatible support or epsilon shape')
    mask = support.to(device=z0.device, dtype=torch.float32)
    if bool((mask.flatten(1).sum(dim=1) == 0).any()):
        raise ValueError('Zero valid latent support in batch sample')
    ab = schedule['alpha_bar'].to(device=z0.device, dtype=torch.float32)[timesteps.long()]
    if ab.shape != (z0.shape[0],):
        raise ValueError('Expected one timestep per sample')
    alpha = ab[:, None, None, None, None]
    epsilon_supported = epsilon.float() * mask
    zt = (alpha.sqrt() * z0.float() + (1 - alpha).sqrt() * epsilon_supported) * mask
    return zt, epsilon_supported


def support_for_patch(raw_patch, threshold=0.50):
    if tuple(raw_patch.shape) != RAW_SHAPE:
        raise ValueError(f'Expected raw patch {RAW_SHAPE}; got {tuple(raw_patch.shape)}')
    occupancy = block_occupancy(support_mask(raw_patch))
    latent = occupancy.ge(threshold)
    if int(latent.sum()) == 0:
        raise ValueError('Zero valid latent support')
    return latent[None].float()


def weighted_batch_loss(prediction, target, support):
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError('Prediction and target shapes must match [B,C,D,H,W]')
    if support.shape != (prediction.shape[0], 1, *prediction.shape[2:]):
        raise ValueError('Invalid support mask shape')
    if bool((support.float().flatten(1).sum(dim=1) == 0).any()):
        raise ValueError('Zero valid latent support in batch sample')
    # Explicit FP32 reduction even when the forward ran under CUDA autocast.
    return masked_epsilon_mse(prediction.float(), target.float(), support.float())


def validation_case_metrics(patch_rows):
    grouped = {}
    for row in patch_rows:
        grouped.setdefault(row['case_id'], []).append(float(row['loss']))
    if any(len(losses) != 4 for losses in grouped.values()):
        raise ValueError('Every validation case must have exactly four patch losses')
    return {case: sum(losses) / 4 for case, losses in sorted(grouped.items())}


def primary_validation_metric(case_metrics):
    if len(case_metrics) != 270:
        raise ValueError('Primary validation requires exactly 270 case metrics')
    values = list(case_metrics.values())
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Non-finite primary validation metric')
    return sum(values) / 270


def checkpoint_improves(metric, best_metric, tolerance=1e-12):
    if not math.isfinite(metric):
        raise ValueError('Non-finite validation metric')
    return best_metric is None or metric < best_metric - tolerance


def optimizer_step_succeeded(old_scale, new_scale):
    """GradScaler lowers its scale exactly when it skips a non-finite update."""
    return new_scale >= old_scale


def ema_update(ema_model, model, decay=0.999):
    with torch.no_grad():
        for ema, current in zip(ema_model.parameters(), model.parameters()):
            ema.mul_(decay).add_(current.detach(), alpha=1 - decay)
        for ema, current in zip(ema_model.buffers(), model.buffers()):
            ema.copy_(current)


def make_ema(model):
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    return ema


def sample_to_corrupted(raw_image, raw_mask, coords, *, case_id, patch_index, seed=20260923):
    slices = tuple(slice(coords[f'crop_start_{axis}'], coords[f'crop_stop_{axis}']) for axis in 'dhw')
    raw_patch = raw_image[(slice(None), *slices)].float()
    if tuple(raw_patch.shape) != RAW_SHAPE:
        raise ValueError(f'Case {case_id}: unexpected raw patch shape {tuple(raw_patch.shape)}')
    normalized_full = normalize_nonzero(raw_image.float())
    return corrupt_loaded_patch(raw_patch, raw_mask[slices], normalized_full[(slice(None), *slices)],
                                case_id=case_id, patch_index=patch_index, seed=seed)


def corrupt_loaded_patch(raw_patch, raw_mask_patch, normalized_patch, *, case_id, patch_index, seed=20260923):
    if tuple(raw_patch.shape) != RAW_SHAPE:
        raise ValueError(f'Case {case_id}: unexpected raw patch shape {tuple(raw_patch.shape)}')
    raw_support = support_mask(raw_patch)
    occupancy = block_occupancy(raw_support)
    latent_support = occupancy.ge(0.50)
    if int(latent_support.sum()) == 0:
        raise ValueError(f'Zero valid latent support: case_id={case_id}, patch_index={patch_index}')
    support = latent_support[None].float()
    abnormal = abnormal_mask(raw_mask_patch)  # GT is consumed only by frozen corruption.
    pools, fallback = donor_pools(raw_patch, abnormal, raw_support)
    corrupted = corrupt_input(normalized_patch, abnormal, pools, 'local_empirical_resample',
        global_seed=seed, case_id=case_id, patch_index=patch_index)
    if tuple(corrupted.shape) != RAW_SHAPE or not torch.isfinite(corrupted).all():
        raise ValueError(f'Case {case_id}: invalid corrupted patch')
    return corrupted.float(), support, int(sum(bool(x) for x in fallback))


def extract_z0(source, images, means, stds, device):
    images = images.to(device=device, dtype=torch.float32)
    means = means.to(device=device, dtype=torch.float32)
    stds = stds.to(device=device, dtype=torch.float32)
    source.eval().requires_grad_(False)
    with torch.no_grad():
        za = source.projection.anatomy_projector(source.encoder(images)[-1]).float()
        z0 = (za - means[None, :, None, None, None]) / stds[None, :, None, None, None]
    if z0.shape[1:] != LATENT_SHAPE or not torch.isfinite(z0).all():
        raise ValueError(f'Invalid standardized anatomy/background representation: {tuple(z0.shape)}')
    return z0


def load_validation_manifest(path=DEFAULT_VAL_MANIFEST):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != VAL_MANIFEST_SHA256:
        raise ValueError('Frozen validation manifest SHA256 mismatch')
    rows = list(csv.DictReader(io.StringIO(data.decode('utf-8'))))
    if len(rows) != 1080 or len({r['case_id'] for r in rows}) != 270:
        raise ValueError('Frozen validation manifest row/case count mismatch')
    for row in rows:
        row['validation_patch_index'] = int(row['validation_patch_index'])
        for key in ('crop_start_d','crop_start_h','crop_start_w','crop_stop_d','crop_stop_h','crop_stop_w',
                    'coordinate_generation_version','seed','timestep','timestep_seed','epsilon_seed',
                    'corruption_global_seed','corruption_patch_index'):
            row[key] = int(row[key])
        if row['corruption_operator'] != 'local_empirical_resample' or row['corruption_patch_index'] != row['validation_patch_index']:
            raise ValueError('Frozen validation corruption identity mismatch')
    counts = {case: 0 for case in {r['case_id'] for r in rows}}
    for row in rows:
        counts[row['case_id']] += 1
    if set(counts.values()) != {4}:
        raise ValueError('Frozen validation manifest must contain 4 patches per case')
    return rows, data


def validation_manifest_copy(output, manifest_data):
    path = Path(output) / 'validation_manifest.csv'
    if path.exists():
        if path.read_bytes() != manifest_data:
            raise ValueError('Run validation manifest differs from frozen bytes')
        return file_sha256(path)
    path.write_bytes(manifest_data)
    actual = file_sha256(path)
    if actual != VAL_MANIFEST_SHA256:
        raise ValueError('Copied validation manifest SHA256 mismatch')
    return actual


def append_csv(path, fields, row):
    path = Path(path)
    exists = path.exists()
    if exists:
        with path.open(newline='') as stream:
            if next(csv.reader(stream), None) != list(fields):
                raise ValueError(f'CSV schema mismatch: {path}')
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, '') for field in fields})
        stream.flush()
        os.fsync(stream.fileno())


def upsert_csv(path, fields, row, key_fields):
    upsert_csv_many(path, fields, [row], key_fields)


def upsert_csv_many(path, fields, new_rows, key_fields):
    """Keep one durable row per epoch or validation case and epoch."""
    path = Path(path)
    new_rows = list(new_rows)
    if not path.exists():
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows({field: row.get(field, '') for field in fields} for row in new_rows)
            stream.flush(); os.fsync(stream.fileno())
        return
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):
            raise ValueError(f'CSV schema mismatch: {path}')
        existing = list(reader)
    keys = {tuple(str(row[k]) for k in key_fields) for row in new_rows}
    kept = [old for old in existing if tuple(old[k] for k in key_fields) not in keys]
    kept.extend({k: str(row.get(k, '')) for k in fields} for row in new_rows)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(kept)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def truncate_csv_after_epoch(path, fields, epoch_index):
    path = Path(path)
    if not path.exists():
        return
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):
            raise ValueError(f'CSV schema mismatch: {path}')
        rows = [row for row in reader if int(row['epoch_index']) <= epoch_index]
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def save_checkpoint(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def state_payload(model, ema, optimizer, scaler, *, epoch_index, successful_steps,
                  best_metric, best_epoch, manifest_sha, provenance_sha,
                  crop_manifest_sha, crop_manifest_row_count):
    return dict(raw_model_state=model.state_dict(), ema_state=ema.state_dict(),
        optimizer_state=optimizer.state_dict(), gradscaler_state=scaler.state_dict(),
        epoch_index=epoch_index, human_epoch=epoch_index + 1,
        successful_optimizer_step=successful_steps,
        next_sample_cursor=1080, resume_granularity='epoch_boundary_only',
        best_primary_validation_metric=best_metric, best_epoch=best_epoch,
        python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state(),
        torch_cpu_rng_state=torch.get_rng_state(),
        torch_cuda_rng_state=torch.cuda.get_rng_state_all(),
        sampler_state=dict(epoch_index=epoch_index, next_sample_cursor=1080),
        architecture_provenance=dict(parameter_count=sum(p.numel() for p in model.parameters()),
                                     model='LatentDiffusionUNet3D'),
        training_contract_provenance=provenance_sha,
        validation_manifest_sha256=manifest_sha,
        training_crop_manifest_sha256=crop_manifest_sha,
        training_crop_manifest_row_count=crop_manifest_row_count)


def validate_resume_payload(payload, *, model, provenance_sha, manifest_sha):
    expected_params = sum(p.numel() for p in model.parameters())
    if payload.get('architecture_provenance') != dict(parameter_count=expected_params, model='LatentDiffusionUNet3D'):
        raise ValueError('Resume architecture mismatch')
    if payload.get('training_contract_provenance') != provenance_sha:
        raise ValueError('Resume contract/provenance mismatch')
    if payload.get('validation_manifest_sha256') != manifest_sha:
        raise ValueError('Resume validation manifest mismatch')
    cursor = payload.get('next_sample_cursor')
    if cursor != 1080 or payload.get('resume_granularity') != 'epoch_boundary_only':
        raise ValueError('Only safe epoch-boundary resume is supported')
    required = ('raw_model_state','ema_state','optimizer_state','gradscaler_state',
                'python_rng_state','numpy_rng_state','torch_cpu_rng_state','torch_cuda_rng_state',
                'successful_optimizer_step','epoch_index','best_primary_validation_metric','best_epoch',
                'training_crop_manifest_sha256','training_crop_manifest_row_count')
    if any(key not in payload for key in required):
        raise ValueError('Resume checkpoint is missing required state')


def restore_resume(payload, model, ema, optimizer, scaler):
    model.load_state_dict(payload['raw_model_state'], strict=True)
    ema.load_state_dict(payload['ema_state'], strict=True)
    optimizer.load_state_dict(payload['optimizer_state'])
    scaler.load_state_dict(payload['gradscaler_state'])
    random.setstate(payload['python_rng_state'])
    np.random.set_state(payload['numpy_rng_state'])
    torch.set_rng_state(payload['torch_cpu_rng_state'])
    torch.cuda.set_rng_state_all(payload['torch_cuda_rng_state'])


def _is_path_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def verify_output_dir(path, *, smoke=False, gpu_smoke=False, resume=False, scientific_path=DEFAULT_OUTPUT):
    path = Path(path).expanduser().resolve()
    frozen = (ROOT / 'outputs/frozen').resolve()
    if path == frozen or frozen in path.parents:
        raise ValueError('Run output cannot be inside frozen artifacts')
    scientific = Path(scientific_path).expanduser().resolve()
    if smoke and gpu_smoke:
        raise ValueError('CPU smoke and GPU smoke modes are mutually exclusive')
    if (smoke or gpu_smoke) and (_is_path_within(path, scientific) or _is_path_within(scientific, path)):
        raise ValueError('Smoke output must be completely separate from the scientific output path')
    if smoke and 'smoke' not in path.name.lower():
        raise ValueError('Smoke output must use a distinct path containing "smoke"')
    if not smoke and path == DEFAULT_SMOKE_OUTPUT.resolve():
        raise ValueError('Scientific mode cannot use the smoke output path')
    if path.exists() and not resume:
        raise FileExistsError('Output exists; use --resume for a compatible epoch-boundary run')
    if resume and not path.is_dir():
        raise FileNotFoundError('Resume output directory does not exist')
    return path


def configure_logging(output):
    logger = logging.getLogger('d0c0')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(output) / 'training.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def runtime_provenance(config, verified, dataset_root, device, manifest_sha):
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    contract = json.loads(DEFAULT_CONTRACT.read_text())
    return dict(stage='D0-C0', run_type='scientific diffusion training',
        training_start_timestamp=datetime.now(timezone.utc).isoformat(),
        runtime_git_branch=git('rev-parse','--abbrev-ref','HEAD'),
        runtime_git_commit=git('rev-parse','HEAD'), git_status_short=git('status','--short').splitlines(),
        git_working_tree_dirty=bool(git('status','--porcelain').strip()),
        dataset_root=str(Path(dataset_root).expanduser().resolve()),
        training_split_sha256=TRAIN_SHA256, validation_split_sha256=VAL_SHA256,
        e1c_checkpoint_path=str(DEFAULT_E1C.resolve()), e1c_checkpoint_sha256=E1C_SOURCE_SHA256,
        d0a_contract_sha256=D0A_SHA256,d0b3_corruption_decision_sha256=D0B3_SHA256,
        d0b4_final_freeze_sha256=FINAL_SHA256,d0b4c_contract_sha256=B4C_SHA256,
        d0b4b_contract_sha256=B4B_SHA256, validation_manifest_sha256=manifest_sha,
        architecture=contract['model'], parameter_count=contract['model']['parameter_count'],
        optimizer=contract['optimizer'],amp=contract['precision'],ema=contract['ema'],
        seed_semantics=contract['stable_hashing'], device=str(device),cuda_version=torch.version.cuda,
        pytorch_version=str(torch.__version__),validation_role='DEVELOPMENT_VALIDATION',
        independent_final_evaluation='separate/TBD', scientific_training=True,
        d0b0_artifact_path=verified['d0b0_artifact_path'],
        d0b0_artifact_sha256=verified['d0b0_artifact_sha256'],
        d0b4a_decision_path=verified['b4a_decision_path'],
        d0b4a_decision_sha256=verified['b4a_decision_sha256'],
        resume_granularity='epoch_boundary_only',
        donor_fallback_diagnostic='D0-B3 fallback modality counts recorded per completed epoch; never change samples.',
        verified_source_hashes={k:v for k,v in verified.items() if k.endswith('sha256')})


def run_smoke(output_dir=DEFAULT_SMOKE_OUTPUT):
    """Exercise pure diffusion utilities with synthetic CPU tensors only."""
    output = verify_output_dir(output_dir, smoke=True)
    output.mkdir(parents=True, exist_ok=False)
    logger = configure_logging(output)
    model = LatentDiffusionUNet3D()
    count = sum(p.numel() for p in model.parameters())
    if count != 22825280:
        raise ValueError(f'Unexpected model parameter count: {count}')
    z = torch.zeros(2, *LATENT_SHAPE)
    support = torch.ones(2, 1, 12, 12, 12)
    noise = torch.ones_like(z)
    schedule = cosine_schedule()
    zt, epsilon = forward_diffuse(z, torch.tensor([0, 999]), noise, support, schedule)
    loss = weighted_batch_loss(torch.zeros_like(epsilon), epsilon, support)
    report = dict(stage='D0-C0', smoke=True, scientific_training=False,
        synthetic_cpu_only=True, model_parameter_count=count, forward_shape=list(zt.shape),
        loss=float(loss), no_scientific_artifacts_written=True)
    (output / 'smoke_summary.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    logger.info('Non-scientific CPU smoke passed; no scientific training performed')
    return report


def gpu_smoke_train_cases(case_ids, train_batches, batch_size=GPU_SMOKE_BATCH_SIZE, epoch_index=0):
    if batch_size != GPU_SMOKE_BATCH_SIZE:
        raise ValueError('GPU smoke physical batch size is locked at 8; fallback is forbidden')
    if not isinstance(train_batches, int) or isinstance(train_batches, bool) or train_batches < 1:
        raise ValueError('GPU smoke train batches must be a positive integer')
    ordered = deterministic_case_order(case_ids, epoch_index)
    cases = ordered[:train_batches * batch_size]
    if len(cases) != train_batches * batch_size:
        raise ValueError('Locked training split does not contain enough cases for requested full batches')
    return cases


def gpu_smoke_validation_rows(rows, validation_patches):
    if not isinstance(validation_patches, int) or isinstance(validation_patches, bool) or not 1 <= validation_patches <= 1080:
        raise ValueError('GPU smoke validation patch count must be in [1,1080]')
    selected = list(rows[:validation_patches])
    if len(selected) != validation_patches:
        raise ValueError('Frozen validation manifest has too few rows')
    return selected


def frozen_source_fingerprint(source):
    """Hash E1-C Encoder and P_a values without changing their tensors."""
    digest = hashlib.sha256()
    modules = (('encoder', source.encoder), ('anatomy_projector', source.projection.anatomy_projector))
    for module_name, module in modules:
        for name, parameter in module.named_parameters():
            tensor = parameter.detach().to(device='cpu').contiguous()
            digest.update(f'{module_name}.{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0'.encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def source_fingerprint_unchanged(before, source):
    return before == frozen_source_fingerprint(source)


def validate_cli_args(args, parser=None):
    if args.smoke and args.gpu_smoke:
        if parser:
            parser.error('--smoke and --gpu-smoke are mutually exclusive')
        raise ValueError('--smoke and --gpu-smoke are mutually exclusive')
    if not args.gpu_smoke and (args.gpu_smoke_train_batches is not None
                               or args.gpu_smoke_validation_patches is not None):
        if parser:
            parser.error('GPU smoke limits are legal only with --gpu-smoke')
        raise ValueError('GPU smoke limits are legal only with --gpu-smoke')
    if args.gpu_smoke:
        if args.dataset_root is None or not args.output_dir_explicit:
            if parser:
                parser.error('--gpu-smoke requires --dataset-root and an explicit --output-dir')
            raise ValueError('--gpu-smoke requires --dataset-root and an explicit --output-dir')
        if args.gpu_smoke_train_batches is None:
            args.gpu_smoke_train_batches = GPU_SMOKE_DEFAULT_TRAIN_BATCHES
        if args.gpu_smoke_validation_patches is None:
            args.gpu_smoke_validation_patches = GPU_SMOKE_DEFAULT_VALIDATION_PATCHES
        if args.gpu_smoke_train_batches < 1 or args.gpu_smoke_validation_patches < 1:
            if parser:
                parser.error('GPU smoke limits must be positive integers')
            raise ValueError('GPU smoke limits must be positive integers')
    return args


def build_arg_parser(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true', help='Synthetic CPU utility smoke; non-scientific.')
    mode.add_argument('--gpu-smoke', action='store_true', help='Bounded real-data CUDA integration smoke; non-scientific.')
    parser.add_argument('--gpu-smoke-train-batches', type=int, default=None)
    parser.add_argument('--gpu-smoke-validation-patches', type=int, default=None)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    args.output_dir_explicit = any(arg == '--output-dir' or arg.startswith('--output-dir=') for arg in argv)
    return validate_cli_args(args, parser)


def configure_gpu_smoke_logging(output):
    logger = logging.getLogger('d0c1-gpu-smoke')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(output) / 'gpu_smoke.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def gpu_smoke_runtime_identity():
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    return dict(runtime_git_branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
        runtime_git_commit=git('rev-parse', 'HEAD'), git_status_short=git('status', '--short').splitlines())


def _gpu_smoke_references(verified):
    keys = ('final_freeze_sha256', 'b4c_contract_sha256', 'b4b_contract_sha256',
        'validation_manifest_sha256', 'validation_manifest_audit_sha256', 'training_split_sha256',
        'validation_split_sha256', 'e1c_checkpoint_sha256', 'd0a_contract_sha256', 'd0b4d_manifest_sha256',
        'd0b0_artifact_sha256', 'b4a_decision_sha256', 'd0b3_decision_sha256')
    result = {key: verified[key] for key in keys}
    result.update(d0b0_artifact_path=verified['d0b0_artifact_path'],
        d0b4a_decision_path=verified['b4a_decision_path'])
    return result


def gpu_smoke_summary_template(verified, dataset_root, train_batches, validation_patches):
    return dict(stage='D0-C1', status='FAIL', scientific_training=False,
        scientific_training_started=False, run_type='non_scientific_gpu_smoke',
        dataset_root=str(Path(dataset_root).expanduser().resolve()),
        verified_frozen_references=_gpu_smoke_references(verified),
        train_batches_requested=train_batches, train_batches_completed=0, samples_processed=0,
        physical_batch_size=GPU_SMOKE_BATCH_SIZE, loss_values=[], gradient_norms=[],
        successful_optimizer_steps=0, ema_updates=0, validation_patches_requested=validation_patches,
        validation_patches_completed=0, validation_patch_losses=[], checkpoint_selection_performed=False,
        source_model_unchanged=False, peak_memory_allocated=0, peak_memory_reserved=0,
        peak_reserved_fraction=0.0, total_cuda_memory=0,
        interpretation='Engineering integration check only; losses are not scientific results.')


def validate_gpu_smoke_artifact_inventory(output):
    allowed = {'gpu_smoke_summary.json', 'gpu_smoke.log', 'gpu_smoke_samples.csv'}
    unexpected = {path.name for path in Path(output).iterdir()} - allowed
    if unexpected:
        raise RuntimeError(f'Unexpected gpu-smoke artifacts: {sorted(unexpected)}')
    forbidden = {name for name in GPU_SMOKE_FORBIDDEN_FILES if (Path(output) / name).exists()}
    if forbidden:
        raise RuntimeError(f'GPU smoke created scientific artifacts: {sorted(forbidden)}')


def run_gpu_smoke(*, dataset_root, output_dir, train_batches=GPU_SMOKE_DEFAULT_TRAIN_BATCHES,
                  validation_patches=GPU_SMOKE_DEFAULT_VALIDATION_PATCHES):
    """Run a disposable real-data CUDA integration check; never select/save checkpoints."""
    verified = verify_startup()
    output = verify_output_dir(output_dir, gpu_smoke=True)
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {root}')
    train_ids = verified['train_ids']
    for case in train_ids[:min(len(train_ids), train_batches * GPU_SMOKE_BATCH_SIZE)]:
        if not (root / case).is_dir():
            raise FileNotFoundError(f'Missing locked training case directory: {root / case}')
    val_rows, _ = load_validation_manifest()
    selected_val_rows = gpu_smoke_validation_rows(val_rows, validation_patches)
    for case in {row['case_id'] for row in selected_val_rows}:
        if not (root / case).is_dir():
            raise FileNotFoundError(f'Missing locked validation case directory: {root / case}')
    cases = gpu_smoke_train_cases(train_ids, train_batches)
    output.mkdir(parents=True, exist_ok=False)
    logger = configure_gpu_smoke_logging(output)
    base = dict(**gpu_smoke_summary_template(verified, root, train_batches, validation_patches),
        **gpu_smoke_runtime_identity(),
        device='cuda', cuda_version=torch.version.cuda, pytorch_version=str(torch.__version__),
        )
    summary_path = output / 'gpu_smoke_summary.json'
    summary_path.write_text(json.dumps(base, indent=2, sort_keys=True) + '\n')
    try:
        model = LatentDiffusionUNet3D()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != 22825280:
            raise ValueError(f'Unexpected model parameter count: {parameter_count}')
        base['model_parameter_count'] = parameter_count
        if not torch.cuda.is_available():
            raise RuntimeError('GPU smoke requires CUDA')
        device = torch.device('cuda')
        properties = torch.cuda.get_device_properties(device)
        base['device_name'] = torch.cuda.get_device_name(device)
        base['total_cuda_memory'] = int(properties.total_memory)
        source, source_info = load_source(DEFAULT_E1C, source_experiment='e1c',
            expected_sha256=E1C_SOURCE_SHA256, expected_epoch=11)
        source.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
        source_before = frozen_source_fingerprint(source)
        d0a = json.loads(DEFAULT_D0A.read_text())
        means = torch.tensor([c['mean'] for c in d0a['channels']], dtype=torch.float32)
        stds = torch.tensor([c['population_std'] for c in d0a['channels']], dtype=torch.float32)
        model = model.to(device=device, dtype=torch.float32)
        ema = make_ema(model).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999),
                                     eps=1e-8, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        schedule = cosine_schedule()
        train_losses, grad_norms = [], []
        peak_allocated = peak_reserved = 0
        for batch_index in range(train_batches):
            batch_cases = cases[batch_index * GPU_SMOKE_BATCH_SIZE:(batch_index + 1) * GPU_SMOKE_BATCH_SIZE]
            if len(batch_cases) != GPU_SMOKE_BATCH_SIZE:
                raise RuntimeError('GPU smoke must retain physical batch size 8; no fallback is allowed')
            items = _training_batch(root, batch_cases, 0)
            z0, mask = _extract_batch(items, source, means, stds, device)
            if z0.shape != (GPU_SMOKE_BATCH_SIZE, *LATENT_SHAPE) or not torch.isfinite(z0).all():
                raise ValueError('Invalid real training z0 shape or values')
            random_values = [sample_randomness(item['row']) for item in items]
            timesteps = torch.tensor([value[0] for value in random_values], device=device, dtype=torch.long)
            epsilon = torch.stack([value[1] for value in random_values]).to(device=device, dtype=torch.float32)
            zt, target = forward_diffuse(z0, timesteps, epsilon, mask, schedule)
            if zt.shape != (GPU_SMOKE_BATCH_SIZE, *LATENT_SHAPE) or not torch.isfinite(zt).all():
                raise ValueError('Invalid real training z_t shape or values')
            optimizer.zero_grad(set_to_none=True)
            # Source tensors are ready; measure allocations of the actual diffusion step.
            torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                prediction = model(zt, timesteps)
            if prediction.shape != z0.shape or not torch.isfinite(prediction).all():
                raise ValueError('Invalid diffusion prediction shape or values')
            loss = weighted_batch_loss(prediction, target, mask)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite GPU smoke training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
            if not torch.isfinite(norm):
                raise ValueError('Non-finite GPU smoke gradient norm')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not optimizer_step_succeeded(old_scale, scaler.get_scale()):
                raise RuntimeError('GradScaler unexpectedly skipped a GPU smoke optimizer step')
            before_ema = next(ema.parameters()).detach().clone()
            current_parameter = next(model.parameters()).detach()
            ema_update(ema, model, 0.999)
            expected_ema = before_ema.mul(0.999).add(current_parameter, alpha=0.001)
            if not torch.equal(next(ema.parameters()), expected_ema):
                raise RuntimeError('EMA did not update after successful GPU smoke optimizer step')
            torch.cuda.synchronize(device)
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
            peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))
            train_losses.append(float(loss.detach().float().item()))
            grad_norms.append(float(norm.detach().float().item()))
            base['train_batches_completed'] += 1
            base['samples_processed'] += GPU_SMOKE_BATCH_SIZE
            base['successful_optimizer_steps'] += 1
            base['ema_updates'] += 1
            logger.info('gpu-smoke train batch=%d/%d loss=%.8g grad_norm=%.8g',
                batch_index + 1, train_batches, train_losses[-1], grad_norms[-1])
        ema.eval()
        patch_losses = []
        means_device, stds_device = means.to(device), stds.to(device)
        cached_case_id, cached_case = None, None
        with torch.no_grad():
            for row in selected_val_rows:
                if cached_case_id != row['case_id']:
                    image, raw_mask = load_case(root, row['case_id'])
                    cached_case = (image, raw_mask, normalize_nonzero(image.float()))
                    cached_case_id = row['case_id']
                item = _load_item(root, row, row['validation_patch_index'], validation=True, loaded=cached_case)
                z0 = extract_z0(source, item['corrupted'][None], means_device, stds_device, device)
                if z0.shape != (1, *LATENT_SHAPE) or not torch.isfinite(z0).all():
                    raise ValueError('Invalid validation z0 shape or values')
                identity = [row[key] for key in ('case_id','validation_patch_index','crop_start_d','crop_start_h',
                    'crop_start_w','crop_stop_d','crop_stop_h','crop_stop_w')]
                expected_t_seed = generator_seed([20260923, identity, 'validation_timestep'])
                expected_e_seed = generator_seed([20260923, identity, 'validation_epsilon'])
                if row['timestep_seed'] != expected_t_seed or row['epsilon_seed'] != expected_e_seed:
                    raise ValueError('Frozen validation seed identity mismatch')
                regenerated_t = int(torch.randint(0, 1000, (1,), generator=torch.Generator(device='cpu').manual_seed(
                    row['timestep_seed'])).item())
                if regenerated_t != row['timestep']:
                    raise ValueError('Frozen validation timestep does not match its stored seed')
                epsilon = torch.randn(LATENT_SHAPE, dtype=torch.float32, device='cpu',
                    generator=torch.Generator(device='cpu').manual_seed(row['epsilon_seed']))[None].to(device)
                mask = item['support'][None].to(device=device, dtype=torch.float32)
                timesteps = torch.tensor([row['timestep']], device=device, dtype=torch.long)
                zt, target = forward_diffuse(z0, timesteps, epsilon, mask, schedule)
                if not torch.isfinite(zt).all():
                    raise ValueError('Non-finite validation z_t')
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    prediction = ema(zt * mask, timesteps)
                if prediction.shape != z0.shape or not torch.isfinite(prediction).all():
                    raise ValueError('Invalid validation prediction shape or values')
                patch_loss = weighted_batch_loss(prediction, target, mask)
                if not torch.isfinite(patch_loss):
                    raise ValueError('Non-finite validation patch loss')
                patch_losses.append(float(patch_loss.item()))
                logger.info('gpu-smoke validation patch=%d/%d loss=%.8g',
                    len(patch_losses), validation_patches, patch_losses[-1])
        base.update(device_name=torch.cuda.get_device_name(device), train_batches_requested=train_batches,
            train_batches_completed=len(train_losses), samples_processed=len(train_losses) * GPU_SMOKE_BATCH_SIZE,
            physical_batch_size=GPU_SMOKE_BATCH_SIZE, loss_values=train_losses, gradient_norms=grad_norms,
            successful_optimizer_steps=len(train_losses), ema_updates=len(train_losses),
            validation_patches_completed=len(patch_losses), validation_patch_losses=patch_losses,
            peak_memory_allocated=peak_allocated, peak_memory_reserved=peak_reserved,
            peak_reserved_fraction=peak_reserved / int(properties.total_memory),
            source_model_unchanged=source_fingerprint_unchanged(source_before, source),
            model_parameter_count=parameter_count, source_checkpoint_sha256=source_info['sha256'])
        if not base['source_model_unchanged']:
            raise RuntimeError('Frozen E1-C Encoder/P_a parameters changed during GPU smoke')
        base['status'] = 'PASS'
        logger.info('GPU smoke passed; no scientific training or checkpoint selection occurred.')
        summary_path.write_text(json.dumps(base, indent=2, sort_keys=True) + '\n')
        validate_gpu_smoke_artifact_inventory(output)
        return base
    except Exception as exc:
        base['status'] = 'FAIL'
        base['failure'] = f'{type(exc).__name__}: {exc}'
        summary_path.write_text(json.dumps(base, indent=2, sort_keys=True) + '\n')
        logger.exception('GPU smoke failed; no batch-size fallback was attempted.')
        raise


def _load_item(case_root, row, epoch_index, *, validation=False, loaded=None):
    image, raw_mask, normalized = loaded if loaded is not None else load_case(case_root, row['case_id']) + (None,)
    if normalized is None:
        normalized = normalize_nonzero(image.float())
    coords = row
    slices = tuple(slice(row[f'crop_start_{axis}'], row[f'crop_stop_{axis}']) for axis in 'dhw')
    cropped, support, fallback = corrupt_loaded_patch(
        image[(slice(None), *slices)].float(), raw_mask[slices],
        normalized[(slice(None), *slices)], case_id=row['case_id'],
        patch_index=row['corruption_patch_index'], seed=row['corruption_global_seed'])
    item = dict(row=row, corrupted=cropped, support=support, fallback_count=fallback)
    return item


def _training_batch(case_root, cases, epoch_index, crop_rows_by_case=None):
    result = []
    for case in cases:
        image, raw_mask = load_case(case_root, case)
        if crop_rows_by_case is None:
            coords = deterministic_crop(case, epoch_index, image.shape[1:])
        else:
            coords = crop_rows_by_case[case]
            if int(coords['epoch_index']) != int(epoch_index):
                raise ValueError('Training crop manifest epoch mismatch')
        slices = tuple(slice(coords[f'crop_start_{axis}'], coords[f'crop_stop_{axis}']) for axis in 'dhw')
        raw_patch = image[(slice(None), *slices)].float()
        normalized_full = normalize_nonzero(image.float())
        try:
            corrupted, support, fallback = corrupt_loaded_patch(
                raw_patch, raw_mask[slices], normalized_full[(slice(None), *slices)],
                case_id=case, patch_index=epoch_index)
        except ValueError as exc:
            if 'Zero valid latent support' in str(exc):
                raise ValueError(f'Zero valid latent support: case_id={case}, epoch_index={epoch_index}, '
                    f'crop_coordinates={{{", ".join(f"{k}: {v}" for k,v in coords.items() if k.startswith("crop_") )}}}') from exc
            raise
        row = dict(**coords, corruption_patch_index=epoch_index, corruption_global_seed=20260923)
        result.append(dict(row=row, corrupted=corrupted, support=support, fallback_count=fallback))
    return result


def _extract_batch(items, source, means, stds, device):
    return extract_z0(source, torch.stack([item['corrupted'] for item in items]), means, stds, device), \
           torch.stack([item['support'] for item in items]).to(device=device, dtype=torch.float32)


def _validate(ema, source, means, stds, rows, data_root, schedule, device, logger, human_epoch):
    ema.eval()
    ema.to(device)
    patch_rows, per_case = [], {}
    total_sq, total_valid = 0.0, 0
    loaded_case = None
    loaded_id = None
    for offset in range(0, len(rows), 8):
        batch_rows = rows[offset:offset + 8]
        items = []
        for row in batch_rows:
            if loaded_id != row['case_id']:
                image, raw_mask = load_case(data_root, row['case_id'])
                loaded_case = (image, raw_mask, normalize_nonzero(image.float()))
                loaded_id = row['case_id']
            items.append(_load_item(data_root, row, row['validation_patch_index'],
                                    validation=True, loaded=loaded_case))
        z0, mask = _extract_batch(items, source, means, stds, device)
        timesteps, noises = [], []
        for item in items:
            row = item['row']
            identity = [row[k] for k in ('case_id','validation_patch_index','crop_start_d','crop_start_h',
                'crop_start_w','crop_stop_d','crop_stop_h','crop_stop_w')]
            t_seed = row['timestep_seed']
            expected_e = generator_seed([20260923, identity, 'validation_epsilon'])
            if t_seed != generator_seed([20260923, identity, 'validation_timestep']) or row['epsilon_seed'] != expected_e:
                raise ValueError('Validation manifest randomness identity mismatch')
            timesteps.append(row['timestep'])
            noises.append(torch.randn(LATENT_SHAPE, generator=torch.Generator(device='cpu').manual_seed(row['epsilon_seed'])))
        t = torch.tensor(timesteps, device=device, dtype=torch.long)
        epsilon = torch.stack(noises).to(device=device, dtype=torch.float32)
        ab = schedule['alpha_bar'].to(device=device, dtype=torch.float32)[t][:,None,None,None,None]
        epsilon_supported = epsilon * mask
        zt = (ab.sqrt() * z0.float() + (1-ab).sqrt() * epsilon_supported) * mask
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            prediction = ema(zt * mask, t)
        patch_losses = []
        for j, item in enumerate(items):
            m = mask[j:j+1]
            n = int(m.sum().item()) * 256
            if n == 0:
                raise ValueError(f'Zero validation support: {item["row"]["case_id"]}')
            sq = ((prediction[j:j+1].float() - epsilon_supported[j:j+1].float()).square() * m).sum()
            value = float((sq / n).item())
            if not math.isfinite(value):
                raise ValueError('Non-finite validation patch metric')
            patch_losses.append(value)
            total_sq += float(sq.item())
            total_valid += n
            row = item['row']
            patch_rows.append(dict(human_epoch=human_epoch, epoch_index=human_epoch-1,
                case_id=row['case_id'], patch_count=4, validation_patch_index=row['validation_patch_index'],
                case_mean_epsilon_mse=''))
            per_case.setdefault(row['case_id'], []).append((row['validation_patch_index'], value))
    case_metrics = {case: sum(loss for _,loss in sorted(values))/4 for case,values in per_case.items()}
    if len(case_metrics) != 270 or any(len(v) != 4 for v in per_case.values()):
        raise ValueError('Validation must evaluate exactly 4 patches for each of 270 cases')
    primary = primary_validation_metric(case_metrics)
    global_mse = total_sq / total_valid
    case_csv_rows = [dict(human_epoch=human_epoch, epoch_index=human_epoch-1, case_id=case,
                          patch_count=4, case_mean_epsilon_mse=value) for case,value in sorted(case_metrics.items())]
    logger.info('validation epoch=%d primary=%.12g secondary_global=%.12g',human_epoch,primary,global_mse)
    return primary, global_mse, case_csv_rows


def _checkpoint_contract_sha():
    return hashlib.sha256(DEFAULT_CONTRACT.read_bytes()).hexdigest()


def validate_resume_provenance(existing, current):
    keys = ('dataset_root', 'training_split_sha256', 'validation_split_sha256',
            'd0b4_final_freeze_sha256', 'd0b4c_contract_sha256',
            'd0b4b_contract_sha256', 'validation_manifest_sha256')
    for key in keys:
        if existing.get(key) != current[key]:
            raise ValueError(f'Resume provenance mismatch: {key}')


def train(config_path=DEFAULT_CONFIG, *, dataset_root=None, output_dir=DEFAULT_OUTPUT,
          resume=False, smoke=False, device=None):
    if smoke:
        return run_smoke(output_dir)
    # Startup contract audit is deliberately before CUDA and any output creation.
    verified = verify_startup()
    active_device = device or 'cuda'
    if active_device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Scientific D0-C0 training requires CUDA; use --smoke for CPU utility checks')
    output = verify_output_dir(output_dir, resume=resume)
    if resume:
        for name in ('latest.pt', 'training_provenance.json', 'validation_manifest.csv'):
            if not (output / name).is_file():
                raise FileNotFoundError(f'Resume output is missing required {name}')
        if (output / 'validation_manifest.csv').read_bytes() != DEFAULT_VAL_MANIFEST.read_bytes():
            raise ValueError('Resume validation manifest differs from frozen bytes')
    config = read_config(config_path)
    root = Path(dataset_root or config['root']).expanduser().resolve()
    train_ids, val_ids = verified['train_ids'], verified['validation_ids']
    for case in train_ids + val_ids:
        if not (root / case).is_dir():
            raise FileNotFoundError(f'Missing locked case directory: {root / case}')
    val_rows, val_data = load_validation_manifest()
    # Dimensions come from the existing loader. GT is never passed to the crop
    # sampler; only image.shape is retained for deterministic coordinate math.
    train_dimensions = {}
    for case in train_ids:
        image, _raw_mask = load_case(root, case)
        train_dimensions[case] = tuple(int(size) for size in image.shape[1:])
    output.mkdir(parents=True, exist_ok=resume)
    manifest_sha = validation_manifest_copy(output, val_data)
    logger = configure_logging(output)
    provenance = runtime_provenance(config, verified, root, active_device, manifest_sha)
    prov_path = output / 'training_provenance.json'
    if resume:
        existing = json.loads(prov_path.read_text())
        validate_resume_provenance(existing, provenance)
    else:
        if any((output / name).exists() for name in ('best.pt','latest.pt','training_history.csv',
                                                      'validation_per_case.csv','training_provenance.json')):
            raise FileExistsError('Output already contains scientific run artifacts')
        prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    random.seed(20260923); np.random.seed(20260923); torch.manual_seed(20260923); torch.cuda.manual_seed_all(20260923)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source, source_info = load_source(DEFAULT_E1C, source_experiment='e1c', expected_sha256=E1C_SOURCE_SHA256, expected_epoch=11)
    source.eval().requires_grad_(False).to(active_device)
    d0a = json.loads(DEFAULT_D0A.read_text())
    means = torch.tensor([c['mean'] for c in d0a['channels']], dtype=torch.float32)
    stds = torch.tensor([c['population_std'] for c in d0a['channels']], dtype=torch.float32)
    model = LatentDiffusionUNet3D().to(active_device)
    if sum(p.numel() for p in model.parameters()) != 22825280:
        raise ValueError('Diffusion architecture parameter count differs from D0-B4')
    ema = make_ema(model).to(active_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    schedule = cosine_schedule()
    successful_steps, best_metric, best_epoch, start_epoch = 0, None, None, 0
    last_primary = last_secondary = None
    latest_path = output / 'latest.pt'
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError('Resume requires latest.pt')
        payload = torch.load(latest_path, map_location='cpu', weights_only=False)
        validate_resume_payload(payload, model=model, provenance_sha=_checkpoint_contract_sha(), manifest_sha=manifest_sha)
        restore_resume(payload, model, ema, optimizer, scaler)
        successful_steps = payload['successful_optimizer_step']
        best_metric, best_epoch = payload['best_primary_validation_metric'], payload['best_epoch']
        last_primary = payload.get('latest_validation_primary_metric')
        last_secondary = payload.get('latest_secondary_global_support_weighted_mse')
        truncate_csv_after_epoch(output/'training_history.csv', HISTORY_FIELDS, payload['epoch_index'])
        truncate_csv_after_epoch(output/'validation_per_case.csv', CASE_FIELDS, payload['epoch_index'])
        start_epoch = payload['epoch_index'] + 1
        if start_epoch >= 100:
            raise ValueError('Training already completed 100 epochs')
        expected_next_rows = training_epoch_crop_rows(train_ids, start_epoch, train_dimensions)
        verify_training_crop_resume(output/'training_crop_manifest.csv',
            completed_epoch_index=payload['epoch_index'],
            checkpoint_sha256=payload['training_crop_manifest_sha256'],
            checkpoint_row_count=payload['training_crop_manifest_row_count'],
            expected_next_rows=expected_next_rows)
    best_path = output / 'best.pt'
    for epoch in range(start_epoch, 100):
        epoch_started = time.perf_counter()
        expected_crop_rows = training_epoch_crop_rows(train_ids, epoch, train_dimensions)
        ordered = [row['case_id'] for row in expected_crop_rows]
        crop_manifest_path = output / 'training_crop_manifest.csv'
        crop_manifest_sha, crop_manifest_row_count = ensure_training_crop_epoch(
            crop_manifest_path, expected_crop_rows)
        crop_rows_by_case = {row['case_id']: row for row in expected_crop_rows}
        losses, grad_norms, fallback_total = [], [], 0
        skipped = 0
        model.train(); optimizer.zero_grad(set_to_none=True)
        for batch_index in range(0, 1080, 8):
            cases = ordered[batch_index:batch_index+8]
            items = _training_batch(root, cases, epoch, crop_rows_by_case)
            fallback_total += sum(x['fallback_count'] for x in items)
            z0, mask = _extract_batch(items, source, means, stds, active_device)
            random_values = [sample_randomness(item['row']) for item in items]
            t = torch.tensor([x[0] for x in random_values], dtype=torch.long, device=active_device)
            noise = torch.stack([x[1] for x in random_values]).to(active_device)
            zt, target = forward_diffuse(z0, t, noise, mask, schedule)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                prediction = model(zt, t)
            loss = weighted_batch_loss(prediction, target, mask)
            if not torch.isfinite(loss):
                raise ValueError(f'Non-finite loss at epoch_index={epoch}, batch={batch_index//8}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
            if not torch.isfinite(norm):
                raise ValueError('Non-finite global gradient norm')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            succeeded = optimizer_step_succeeded(old_scale, scaler.get_scale())
            if succeeded:
                successful_steps += 1
                ema_update(ema, model, 0.999)
            else:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().float().item())); grad_norms.append(float(norm.detach().float().item()))
            if (batch_index // 8 + 1) % 5 == 0:
                logger.info('human_epoch=%d batch=%d/135 loss=%.7g successful_steps=%d scaler=%.4g elapsed=%.1fs',
                    epoch+1,batch_index//8+1,losses[-1],successful_steps,scaler.get_scale(),time.perf_counter()-epoch_started)
        human = epoch + 1
        primary = secondary = None
        is_best = False
        case_rows = None
        if human % 5 == 0:
            primary, secondary, case_rows = _validate(ema,source,means,stds,val_rows,root,schedule,active_device,logger,human)
            last_primary, last_secondary = primary, secondary
            is_best = checkpoint_improves(primary,best_metric)
            if is_best:
                best_metric, best_epoch = primary, epoch
                best_payload = state_payload(model,ema,optimizer,scaler,epoch_index=epoch,
                    successful_steps=successful_steps,best_metric=best_metric,best_epoch=best_epoch,
                    manifest_sha=manifest_sha,provenance_sha=_checkpoint_contract_sha(),
                    crop_manifest_sha=crop_manifest_sha,crop_manifest_row_count=crop_manifest_row_count)
                best_payload['primary_validation_metric'] = primary
                torch.save(best_payload,best_path.with_suffix('.pt.tmp')); os.replace(best_path.with_suffix('.pt.tmp'),best_path)
            upsert_csv_many(output/'validation_per_case.csv',CASE_FIELDS,case_rows,('epoch_index','case_id'))
        epochrow = dict(epoch_index=epoch,human_epoch=human,samples_presented=1080,
            successful_optimizer_steps=successful_steps,mean_training_loss=sum(losses)/len(losses),
            global_grad_norm_mean=sum(grad_norms)/len(grad_norms),global_grad_norm_max=max(grad_norms),
            gradscaler_skipped_steps=skipped,learning_rate=1e-4,elapsed_seconds=time.perf_counter()-epoch_started,
            donor_fallback_modality_count=fallback_total,
            primary_validation_metric=primary if primary is not None else '',
            secondary_global_support_weighted_mse=secondary if secondary is not None else '',is_best=is_best)
        upsert_csv(output/'training_history.csv',HISTORY_FIELDS,epochrow,('epoch_index',))
        payload = state_payload(model,ema,optimizer,scaler,epoch_index=epoch,
            successful_steps=successful_steps,best_metric=best_metric,best_epoch=best_epoch,
            manifest_sha=manifest_sha,provenance_sha=_checkpoint_contract_sha(),
            crop_manifest_sha=crop_manifest_sha,crop_manifest_row_count=crop_manifest_row_count)
        payload['latest_validation_primary_metric'] = last_primary
        payload['latest_secondary_global_support_weighted_mse'] = last_secondary
        save_checkpoint(latest_path,payload)
        summary = dict(scheduled_validation_epochs_completed=list(range(5,(human//5)*5+1,5)),
            best_human_epoch=None if best_epoch is None else best_epoch+1,best_primary_metric=best_metric,
            tie_tolerance=1e-12,checkpoint_rule='EMA lowest primary mean case-level validation support-masked epsilon MSE; improvement < best - 1e-12',
            latest_validation_primary_metric=last_primary,latest_secondary_global_support_weighted_mse=last_secondary,
            validation_role='DEVELOPMENT_VALIDATION',independent_final_evaluation='separate/TBD',
            successful_optimizer_steps=successful_steps,donor_fallback_modalities_seen=fallback_total)
        (output/'validation_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    logger.info('Reached frozen maximum of 100 human-facing epochs; stopping.')
    crop_manifest_path = output / 'training_crop_manifest.csv'
    crop_rows = _read_crop_rows(crop_manifest_path)
    epoch_counts = {}
    for row in crop_rows:
        epoch_counts[int(row['epoch_index'])] = epoch_counts.get(int(row['epoch_index']), 0) + 1
    if len(crop_rows) != 108000 or epoch_counts != dict.fromkeys(range(100), 1080):
        raise ValueError('Completed training crop manifest is incomplete')
    final_crop_sha = file_sha256(crop_manifest_path)
    provenance['training_crop_manifest_path'] = str(crop_manifest_path.resolve())
    provenance['training_crop_manifest_sha256'] = final_crop_sha
    provenance['training_crop_manifest_row_count'] = len(crop_rows)
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    summary_path = output / 'validation_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary['training_crop_manifest_sha256'] = final_crop_sha
        summary['training_crop_manifest_row_count'] = len(crop_rows)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return dict(output_dir=str(output),best_epoch=None if best_epoch is None else best_epoch+1,
                best_metric=best_metric,successful_optimizer_steps=successful_steps)


def main():
    args = build_arg_parser()
    if args.gpu_smoke:
        if args.resume:
            raise SystemExit('--resume is not available for --gpu-smoke')
        run_gpu_smoke(dataset_root=args.dataset_root, output_dir=args.output_dir,
            train_batches=args.gpu_smoke_train_batches,
            validation_patches=args.gpu_smoke_validation_patches)
    elif args.smoke:
        output = args.output_dir
        if output == DEFAULT_OUTPUT:
            output = DEFAULT_SMOKE_OUTPUT
        run_smoke(output)
    else:
        train(args.config, dataset_root=args.dataset_root, output_dir=args.output_dir, resume=args.resume)


if __name__ == '__main__':
    main()
