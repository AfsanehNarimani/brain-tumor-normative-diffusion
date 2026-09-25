#!/usr/bin/env python3
"""Run frozen D0-C3a DDIM/DeltaZ inference (explicit scientific or smoke mode).

The generation path reads original MRI modalities only. Segmentation files are
never opened here. A smoke run is non-scientific and processes only a bounded
prefix of frozen manifest windows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.transforms import normalize_nonzero
from models.latent_diffusion_unet3d import LatentDiffusionUNet3D, cosine_schedule
from models.latent_probe import E1C_SOURCE_SHA256, file_sha256, load_source
from scripts.audit_d0b1_abnormal_occupancy import block_occupancy, support_mask

B4B_PATH = ROOT / 'configs/d0b4b_inference_contract.json'
SPATIAL_CONTRACT_PATH = ROOT / 'configs/d0c3_spatial_inference_contract.json'
SPATIAL_DIR = ROOT / 'outputs/frozen/d0c3_spatial_inference_design'
SPATIAL_MANIFEST_PATH = SPATIAL_DIR / 'inference_window_manifest.csv'
D0A_PATH = ROOT / 'outputs/frozen/d0_latent_distribution_audit/d0_za_normalization_contract.json'
C2_FREEZE_DIR = ROOT / 'outputs/frozen/d0c2_scientific_diffusion_training'
DEFAULT_E1C = ROOT / 'outputs/frozen/e1c_anatomy_constraint/training/best.pt'
DEFAULT_SCIENTIFIC_OUTPUT = ROOT / 'outputs/scientific/d0c3_deltaz'
DEFAULT_SMOKE_OUTPUT = ROOT / 'outputs/smoke/d0c3_deltaz'

B4B_SHA256 = '29cf659528f6257cd9f8f218e9f4eaf8fabdc6d8a0e4905505aa8548ae68bde4'
SPATIAL_MANIFEST_SHA256 = '58fac2ebef778d4d19d6da9f21451c25175c61bb4100c03387e8d3c859c05d53'
D0A_SHA256 = 'bca5c0f8f1617d73fc4d0850fe5b6f45e4b2bcbacbb4018f1d5d825de49ad792'
E1C_SHA256 = E1C_SOURCE_SHA256
D0C2_SHA256 = 'aaa6258f26a56c852dc751b8028bfb046da5752974d4cbaf446ad2699a0c6574'
VALIDATION_SPLIT_SHA256 = 'ec5fe6b0bbcf190fe8b7c032f8da4ff1d495969c2d868561e7a3351c572cc5ca'
VALIDATION_SPLIT_PATH = ROOT / 'splits/val_cases.txt'
EXPECTED_PARAMETER_COUNT = 22_825_280
WINDOW_SHAPE = (96, 96, 96)
LATENT_SHAPE = (256, 12, 12, 12)
GLOBAL_SHAPE = (23, 28, 23)
GLOBAL_SEED = 20260923
REPLICATE_INDICES = (0, 1, 2, 3)
MANIFEST_FIELDS = (
    'case_id', 'split', 'window_index', 'original_d', 'original_h', 'original_w',
    'padded_d', 'padded_h', 'padded_w', 'start_d', 'start_h', 'start_w',
    'stop_d', 'stop_h', 'stop_w', 'latent_start_d', 'latent_start_h', 'latent_start_w',
    'latent_stop_d', 'latent_stop_h', 'latent_stop_w',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    return file_sha256(path)


def canonical_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def verify_directory_manifest(directory):
    directory = Path(directory)
    manifest = directory / 'MANIFEST.sha256'
    require(manifest.is_file(), f'Missing frozen MANIFEST.sha256: {manifest}')
    entries = {}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest, rel = line.split('  ', 1)
        require(rel not in entries, f'Duplicate MANIFEST path: {rel}')
        path = directory / rel
        require(path.is_file() and not path.is_symlink(), f'Missing/unsafe frozen artifact: {path}')
        actual = sha256_file(path)
        require(actual == digest, f'MANIFEST hash mismatch: {rel}')
        entries[rel] = digest
    actual_files = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
                    if p.is_file() and p.name != 'MANIFEST.sha256'}
    require(actual_files == set(entries), 'Frozen directory file inventory differs from MANIFEST')
    return sha256_file(manifest), entries


def verify_frozen_inputs():
    require(sha256_file(B4B_PATH) == B4B_SHA256, 'D0-B4b inference contract SHA256 mismatch')
    require(sha256_file(D0A_PATH) == D0A_SHA256, 'D0-A normalization contract SHA256 mismatch')
    require(sha256_file(VALIDATION_SPLIT_PATH) == VALIDATION_SPLIT_SHA256,
            'DEVELOPMENT_VALIDATION split SHA256 mismatch')
    spatial_manifest_sha, spatial_entries = verify_directory_manifest(SPATIAL_DIR)
    require(spatial_entries.get('inference_window_manifest.csv') == SPATIAL_MANIFEST_SHA256,
            'Spatial manifest SHA does not match its frozen MANIFEST')
    require(sha256_file(SPATIAL_MANIFEST_PATH) == SPATIAL_MANIFEST_SHA256,
            'Spatial inference manifest SHA256 mismatch')
    b4b = json.loads(B4B_PATH.read_text(encoding='utf-8'))
    spatial = json.loads(SPATIAL_CONTRACT_PATH.read_text(encoding='utf-8'))
    spatial_provenance = json.loads((SPATIAL_DIR / 'provenance.json').read_text(encoding='utf-8'))
    require(spatial_provenance.get('contract_sha256') == sha256_file(SPATIAL_CONTRACT_PATH),
            'Spatial contract SHA differs from frozen spatial provenance')
    require(spatial_provenance.get('manifest_sha256') == SPATIAL_MANIFEST_SHA256,
            'Spatial manifest SHA differs from frozen spatial provenance')
    require(spatial['locked_dependencies']['d0b4b_inference_contract_sha256'] == B4B_SHA256,
            'Spatial design references a different D0-B4b contract')
    require(b4b['randomness']['seed_hash'] == 'SHA256'
            and b4b['randomness']['seed_payload'] == '[inference_global_seed, sample_identity, replicate_index, purpose]'
            and b4b['randomness']['diffusion_timestep_in_seed_payload'] is False,
            'D0-B4b randomness contract mismatch')
    require(b4b['randomness']['seed_integer'] == 'int(sha256(payload_bytes).hexdigest()[:16], 16) & ((1 << 63) - 1)',
            'Unsupported D0-B4b deterministic seed derivation')
    require(b4b['partial_noise']['primary_timestep'] == 495
            and b4b['ddim']['eta'] == 0.0
            and b4b['ddim']['model_evaluations'] == 50
            and b4b['randomness']['inference_global_seed'] == GLOBAL_SEED
            and b4b['randomness']['replicate_indices'] == list(REPLICATE_INDICES),
            'D0-B4b primary inference parameters mismatch')
    schedule = cosine_schedule(timesteps=1000, offset=0.008, max_beta=0.999)['alpha_bar']
    for timestep, expected in ((327, 0.7501165866851807), (495, 0.5000776052474976),
                               (663, 0.25003883242607117)):
        point = next(item for item in b4b['partial_noise']['points'] if item['timestep'] == timestep)
        require(abs(float(schedule[timestep]) - expected) <= 1e-7
                and abs(float(point['alpha_bar']) - expected) <= 1e-12,
                f'Frozen cosine alpha_bar mismatch at timestep {timestep}')
    require(spatial['window_geometry']['expected_manifest_rows'] == 9720
            and spatial['window_geometry']['windows_per_case'] == 36
            and spatial['tensor_geometry']['global_latent_shape_for_known_padded_shape'] == list(GLOBAL_SHAPE),
            'D0-C3 spatial geometry contract mismatch')
    d0a = json.loads(D0A_PATH.read_text(encoding='utf-8'))
    require(d0a['channel_count'] == 256 and len(d0a['channels']) == 256
            and d0a['formula'] == '(Za[c] - mean[c]) / population_std[c]'
            and d0a['clipping'] is False and d0a['per_case_normalization'] is False
            and d0a['per_patch_normalization'] is False,
            'D0-A normalization contract mismatch')
    c2_manifest_sha, c2_entries = verify_directory_manifest(C2_FREEZE_DIR)
    c2_summary = json.loads((C2_FREEZE_DIR / 'd0c2_training_summary.json').read_text(encoding='utf-8'))
    c2_artifacts = json.loads((C2_FREEZE_DIR / 'd0c2_artifact_manifest.json').read_text(encoding='utf-8'))
    best_ref = c2_artifacts['artifacts']['best.pt']
    require(c2_summary['best_human_epoch'] == 100
            and c2_summary['best_checkpoint_sha256'] == D0C2_SHA256
            and best_ref['sha256'] == D0C2_SHA256,
            'D0-C2 best checkpoint freeze metadata mismatch')
    return dict(b4b_contract_sha256=B4B_SHA256,
        spatial_contract_sha256=sha256_file(SPATIAL_CONTRACT_PATH),
        spatial_manifest_sha256=SPATIAL_MANIFEST_SHA256,
        spatial_freeze_manifest_sha256=spatial_manifest_sha,
        d0a_contract_sha256=D0A_SHA256, d0c2_freeze_manifest_sha256=c2_manifest_sha,
        d0c2_best_checkpoint_expected_sha256=D0C2_SHA256,
        e1c_checkpoint_expected_sha256=E1C_SHA256,
        validation_split_sha256=VALIDATION_SPLIT_SHA256,
        b4b=b4b, spatial=spatial, d0a=d0a,
        c2_summary=c2_summary, c2_artifacts=c2_artifacts)


def verify_checkpoint(path, expected_sha, label):
    path = Path(path).expanduser().resolve()
    require(path.is_file(), f'{label} checkpoint does not exist: {path}')
    actual = sha256_file(path)
    require(actual == expected_sha, f'{label} checkpoint SHA256 mismatch: {actual} != {expected_sha}')
    return path, actual


def load_inference_models(e1c_checkpoint, diffusion_checkpoint, device='cpu'):
    """Hash-check source files before deserialization; use EMA only for diffusion."""
    e1c_path, e1c_sha = verify_checkpoint(e1c_checkpoint, E1C_SHA256, 'E1-C')
    diffusion_path, diffusion_sha = verify_checkpoint(diffusion_checkpoint, D0C2_SHA256, 'D0-C2 best')
    source, source_info = load_source(e1c_path, source_experiment='e1c', expected_sha256=E1C_SHA256,
                                      expected_epoch=11)
    source.eval().requires_grad_(False)
    source.to(device=device, dtype=torch.float32)
    checkpoint = torch.load(diffusion_path, map_location='cpu', weights_only=False)
    required_metadata = dict(epoch_index=99, human_epoch=100, successful_optimizer_step=13499,
        primary_validation_metric=0.7276276773324719,
        best_primary_validation_metric=0.7276276773324719, best_epoch=99, next_sample_cursor=1080)
    for key, value in required_metadata.items():
        require(checkpoint.get(key) == value, f'D0-C2 checkpoint metadata mismatch for {key}')
    arch = checkpoint.get('architecture_provenance')
    require(isinstance(arch, dict) and arch.get('model') == 'LatentDiffusionUNet3D'
            and arch.get('parameter_count') == EXPECTED_PARAMETER_COUNT,
            'D0-C2 checkpoint architecture provenance mismatch')
    require(isinstance(checkpoint.get('ema_state'), dict), 'D0-C2 checkpoint is missing ema_state')
    require(isinstance(checkpoint.get('raw_model_state'), dict), 'D0-C2 checkpoint is missing raw_model_state')
    require(len(checkpoint['ema_state']) == 142 and len(checkpoint['raw_model_state']) == 142,
            'D0-C2 checkpoint model state key count mismatch')
    model = LatentDiffusionUNet3D()
    parameter_count = sum(p.numel() for p in model.parameters())
    require(parameter_count == EXPECTED_PARAMETER_COUNT, f'Diffusion architecture parameter count mismatch: {parameter_count}')
    model.load_state_dict(checkpoint['ema_state'], strict=True)
    model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    # The raw state is required to be present, but is never loaded into the inference model.
    info = dict(e1c_checkpoint_path=str(e1c_path), e1c_checkpoint_sha256=e1c_sha,
        d0c2_checkpoint_path=str(diffusion_path), d0c2_checkpoint_sha256=diffusion_sha,
        diffusion_state_key='ema_state', raw_state_key_loaded=False,
        ema_state_key_count=len(checkpoint['ema_state']), raw_state_key_count=len(checkpoint['raw_model_state']),
        checkpoint_metadata=required_metadata, architecture_parameter_count=parameter_count,
        e1c_source_info=source_info)
    return source, model, info


def load_normalization_stats(contract=None):
    contract = json.loads(D0A_PATH.read_text(encoding='utf-8')) if contract is None else contract
    require(contract.get('channel_count') == 256 and len(contract.get('channels', [])) == 256,
            'D0-A requires exactly 256 ordered channel statistics')
    require([item.get('channel_index') for item in contract['channels']] == list(range(256)),
            'D0-A channels are not in canonical channel order')
    means = torch.tensor([item['mean'] for item in contract['channels']], dtype=torch.float32).view(1, 256, 1, 1, 1)
    stds = torch.tensor([item['population_std'] for item in contract['channels']], dtype=torch.float32).view(1, 256, 1, 1, 1)
    require(torch.isfinite(means).all() and torch.isfinite(stds).all() and (stds > 0).all(),
            'D0-A statistics must be finite with positive population standard deviations')
    return means, stds


def standardize_za(za, means, stds):
    require(za.ndim == 5 and tuple(za.shape[1:]) == LATENT_SHAPE, 'Expected Za [B,256,12,12,12]')
    require(means.shape == stds.shape == (1, 256, 1, 1, 1), 'Expected fixed D0-A [1,256,1,1,1] stats')
    result = (za.float() - means.to(za.device)) / stds.to(za.device)
    require(torch.isfinite(result).all(), 'D0-A standardized latent contains non-finite values')
    return result


def support_from_raw_window(raw_window):
    if raw_window.ndim == 5:
        require(raw_window.shape[0] == 1, 'Support helper accepts one inference window at a time')
        raw_window = raw_window[0]
    raw = support_mask(raw_window)
    support = block_occupancy(raw).ge(0.50)[None, None]
    require(support.shape == (1, 1, 12, 12, 12), 'Unexpected latent support mask shape')
    require(support.any(), 'Inference window has zero valid latent support')
    return support


def sample_identity(case_id, row):
    starts = [int(row[f'start_{axis}']) for axis in 'dhw']
    stops = [int(row[f'stop_{axis}']) for axis in 'dhw']
    return [str(case_id), *starts, *stops]


def deterministic_noise_seed(case_id, row, replicate_index, *, contract=None):
    identity = sample_identity(case_id, row)
    seed_contract = (json.loads(B4B_PATH.read_text(encoding='utf-8'))['randomness']
                     if contract is None else contract['randomness'])
    require(replicate_index in seed_contract['replicate_indices'], 'Invalid partial-noise replicate index')
    require(seed_contract['purpose'] == 'partial_noise' and seed_contract['diffusion_timestep_in_seed_payload'] is False,
            'Unsupported partial-noise seed contract')
    payload = [int(seed_contract['inference_global_seed']), identity, int(replicate_index), seed_contract['purpose']]
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return int(digest[:16], 16) & ((1 << 63) - 1)


def make_partial_noise(case_id, row, replicate_index, *, contract=None, device='cpu'):
    seed = deterministic_noise_seed(case_id, row, replicate_index, contract=contract)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(LATENT_SHAPE, dtype=torch.float32, device='cpu', generator=generator)[None]
    return seed, noise.to(device)


def ddim_reconstruct(model, z0, support, epsilon, start_timestep, *, b4b_contract=None,
                     schedule=None):
    """Run the contract-resolved deterministic DDIM sequence; return prior and call count."""
    contract = json.loads(B4B_PATH.read_text(encoding='utf-8')) if b4b_contract is None else b4b_contract
    ddim = contract['ddim'] if start_timestep == 495 else next(
        item for item in contract['sensitivity_ddim'] if item['start_timestep'] == start_timestep)
    sequence = [int(t) for t in ddim['resolved_timesteps']]
    require(len(sequence) == 50 and sequence[0] == start_timestep and sequence[-1] == 0
            and len(set(sequence)) == 50 and all(a > b for a, b in zip(sequence, sequence[1:])),
            'Frozen DDIM timestep sequence is invalid')
    require(ddim['eta'] == 0.0 and ddim['model_evaluations'] == 50 and ddim['end_timestep'] == 0,
            'Frozen DDIM settings mismatch')
    require(z0.shape == (1, *LATENT_SHAPE) and support.shape == (1, 1, 12, 12, 12)
            and epsilon.shape == z0.shape, 'DDIM input shape mismatch')
    device = z0.device
    schedule = cosine_schedule(timesteps=1000, offset=0.008, max_beta=0.999) if schedule is None else schedule
    alpha_bar = schedule['alpha_bar'].to(device=device, dtype=torch.float32)
    if start_timestep == 495:
        require(abs(float(alpha_bar[495]) - 0.5000776052474976) <= 1e-7,
                'Existing cosine schedule alpha_bar[495] mismatch')
    mask = support.to(dtype=torch.float32)
    z0_supported = z0.float() * mask
    epsilon_supported = epsilon.float() * mask
    t0 = torch.tensor([start_timestep], dtype=torch.long, device=device)
    ab0 = alpha_bar[t0].view(1, 1, 1, 1, 1)
    current = (torch.sqrt(ab0) * z0_supported + torch.sqrt(1.0 - ab0) * epsilon_supported) * mask
    evaluations = 0
    model.eval()
    with torch.no_grad():
        for index, timestep in enumerate(sequence):
            current = current * mask
            t = torch.tensor([timestep], dtype=torch.long, device=device)
            epsilon_hat = model(current, t)
            evaluations += 1
            require(epsilon_hat.shape == z0.shape and torch.isfinite(epsilon_hat).all(),
                    'DDIM model produced invalid epsilon prediction')
            ab_t = alpha_bar[timestep]
            z0_hat = (current - torch.sqrt(1.0 - ab_t) * epsilon_hat) / torch.sqrt(ab_t)
            if timestep == 0:
                prior = z0_hat * mask
                break
            t_prev = sequence[index + 1]
            ab_prev = alpha_bar[t_prev]
            current = (torch.sqrt(ab_prev) * z0_hat
                       + torch.sqrt(1.0 - ab_prev) * epsilon_hat) * mask
    require(evaluations == 50 and prior.shape == z0.shape and torch.isfinite(prior).all(),
            'DDIM did not complete exactly 50 finite model evaluations')
    return prior, evaluations


def local_deltaz(z0, priors, support):
    require(len(priors) == 4, 'Primary DeltaZ requires exactly four noise replicates')
    require(z0.shape == (1, *LATENT_SHAPE) and support.shape == (1, 1, 12, 12, 12),
            'DeltaZ input shapes mismatch')
    mask = support.to(dtype=torch.bool)
    per_replicate = [((z0 - prior).abs() * mask).mean(dim=1, keepdim=True) for prior in priors]
    result = torch.stack(per_replicate, dim=0).mean(dim=0) * mask
    require(result.shape == (1, 1, 12, 12, 12), 'Unexpected local DeltaZ shape')
    return result, per_replicate


def place_and_aggregate(local_maps, supports, rows):
    require(len(local_maps) == len(supports) == len(rows), 'Local result/manifest window count mismatch')
    total = torch.zeros((1, *GLOBAL_SHAPE), dtype=torch.float32)
    counts = torch.zeros((1, *GLOBAL_SHAPE), dtype=torch.int32)
    for local, support, row in zip(local_maps, supports, rows):
        start = tuple(int(row[f'start_{axis}']) for axis in 'dhw')
        region = tuple(slice(int(row[f'latent_start_{axis}']), int(row[f'latent_stop_{axis}'])) for axis in 'dhw')
        require(tuple(s.stop - s.start for s in region) == (12, 12, 12), 'Manifest latent window extent mismatch')
        require(tuple(start_i // 8 for start_i in start) == tuple(s.start for s in region),
                'Manifest local-to-global coordinate mismatch')
        mask = support.to(torch.bool)
        total[(slice(None), *region)] += local[0].float() * mask[0].float()
        counts[(slice(None), *region)] += mask[0].to(torch.int32)
    valid = counts > 0
    mean = torch.where(valid, total / counts.clamp_min(1), torch.zeros_like(total))
    require(mean.shape == valid.shape == counts.shape == (1, *GLOBAL_SHAPE), 'Global map shape mismatch')
    return mean, valid, counts


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode('utf-8'))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(canonical_json_bytes(list(value.shape)))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _load_original_mri(dataset_root, case_id):
    """Read four original MRI modalities only; never open segmentation paths."""
    folder = Path(dataset_root).expanduser().resolve() / case_id
    paths = [folder / f'{case_id}-{name}.nii.gz' for name in ('t1n', 't1c', 't2w', 't2f')]
    volumes = [nib.load(str(path)) for path in paths]
    reference = volumes[0]
    arrays = []
    for path, volume in zip(paths, volumes):
        require(len(volume.shape) == 3 and volume.shape == reference.shape,
                f'{case_id}: MRI modality spatial shape mismatch: {path}')
        require(np.allclose(volume.affine, reference.affine, atol=1e-5, rtol=0),
                f'{case_id}: MRI modality affine mismatch: {path}')
        require(np.allclose(volume.header.get_zooms()[:3], reference.header.get_zooms()[:3], atol=1e-5, rtol=0),
                f'{case_id}: MRI modality spacing mismatch: {path}')
        data = volume.get_fdata(dtype=np.float32)
        require(np.isfinite(data).all(), f'{case_id}: MRI contains non-finite values')
        arrays.append(torch.from_numpy(np.asarray(data)))
    return torch.stack(arrays)


def prepare_case_mri(dataset_root, case_id, rows):
    raw = _load_original_mri(dataset_root, case_id)
    expected_original = tuple(int(rows[0][f'original_{axis}']) for axis in 'dhw')
    expected_padded = tuple(int(rows[0][f'padded_{axis}']) for axis in 'dhw')
    require(tuple(raw.shape[1:]) == expected_original, f'{case_id}: original MRI shape differs from frozen manifest')
    normalized = normalize_nonzero(raw)
    pad = tuple(v for axis in reversed(range(3)) for v in (0, expected_padded[axis] - expected_original[axis]))
    padded = F.pad(normalized, pad, mode='constant', value=0)
    raw_padded = F.pad(raw, pad, mode='constant', value=0)
    require(tuple(padded.shape[1:]) == expected_padded, f'{case_id}: padded MRI shape mismatch')
    require(tuple(raw_padded.shape[1:]) == expected_padded, f'{case_id}: raw support pad shape mismatch')
    return raw, raw_padded, padded


def extract_za(source, normalized_window, device):
    require(tuple(normalized_window.shape) == (1, 4, *WINDOW_SHAPE), 'Expected E1-C input [1,4,96,96,96]')
    source.eval()
    source.requires_grad_(False)
    with torch.no_grad():
        bottleneck = source.encoder(normalized_window.to(device=device, dtype=torch.float32))[-1]
        za = source.projection.anatomy_projector(bottleneck)
    require(za.shape == (1, *LATENT_SHAPE) and za.dtype == torch.float32 and torch.isfinite(za).all(),
            'E1-C Za shape/dtype/finiteness mismatch')
    require(not za.requires_grad, 'Frozen source unexpectedly tracked gradients')
    return za


def parse_manifest(path=SPATIAL_MANIFEST_PATH):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(MANIFEST_FIELDS), 'Spatial inference manifest column order mismatch')
        rows = []
        for item in reader:
            row = dict(item)
            for key in MANIFEST_FIELDS:
                if key not in ('case_id', 'split'):
                    row[key] = int(row[key])
            rows.append(row)
    require(len(rows) == 9720, f'Expected 9720 frozen spatial manifest rows; got {len(rows)}')
    require(all(row['split'] == 'validation' for row in rows), 'Spatial manifest includes a non-validation split')
    require([(r['case_id'], r['window_index']) for r in rows]
            == sorted((r['case_id'], r['window_index']) for r in rows), 'Spatial manifest ordering is not canonical')
    by_case = {}
    for row in rows:
        by_case.setdefault(row['case_id'], []).append(row)
        for axis in 'dhw':
            require(row[f'stop_{axis}'] - row[f'start_{axis}'] == 96, 'Manifest window extent mismatch')
            require(row[f'latent_start_{axis}'] * 8 == row[f'start_{axis}']
                    and row[f'latent_stop_{axis}'] * 8 == row[f'stop_{axis}'], 'Manifest latent coordinate mismatch')
    require(len(by_case) == 270 and all(len(case_rows) == 36 for case_rows in by_case.values()),
            'Spatial manifest must contain 270 cases x 36 windows')
    return rows, by_case


def path_is_within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def paths_overlap(first, second):
    return path_is_within(first, second) or path_is_within(second, first)


def validate_output_mode_path(output_dir, *, scientific):
    requested = Path(output_dir).expanduser()
    require(not requested.is_symlink(), 'Inference output path cannot be a symlink')
    output = requested.resolve()
    scientific_root, smoke_root = DEFAULT_SCIENTIFIC_OUTPUT.resolve(), DEFAULT_SMOKE_OUTPUT.resolve()
    if scientific:
        require(path_is_within(output, scientific_root), 'Scientific output must remain under outputs/scientific/d0c3_deltaz')
        require(not path_is_within(output, smoke_root) and not path_is_within(smoke_root, output),
                'Scientific output overlaps smoke output namespace')
    else:
        require(path_is_within(output, smoke_root), 'Smoke output must remain under outputs/smoke/d0c3_deltaz')
        require(not path_is_within(output, scientific_root) and not path_is_within(scientific_root, output),
                'Smoke output overlaps scientific output namespace')
    require(not path_is_within(output, (ROOT / 'outputs/frozen').resolve()),
            'Inference output cannot be inside frozen artifacts')
    return output


def _file_sha_map(directory):
    return {p.name: sha256_file(p) for p in sorted(Path(directory).iterdir()) if p.is_file()}


def _atomic_json(path, payload):
    path = Path(path)
    fd, temp_name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def mark_run_in_progress(provenance, *, runtime, started_at=None):
    """Return resume provenance that cannot retain a stale completed status."""
    current = dict(provenance)
    previous_completed_at = current.pop('completed_at', None)
    current.update(status='RUNNING', last_attempt_started_at=started_at or datetime.now(timezone.utc).isoformat(),
                   runtime_last_attempt=runtime)
    if previous_completed_at is not None:
        current['previous_completed_at'] = previous_completed_at
    return current


def finalize_run_provenance(provenance, *, completed_at, cases_processed,
                            windows_processed, source_model_unchanged, runtime_final):
    """Publish PASS only after all inference and final integrity checks succeeded."""
    final = dict(provenance)
    final.update(status='PASS', completed_at=completed_at,
        cases_processed=cases_processed, windows_processed=windows_processed,
        source_model_unchanged=source_model_unchanged, runtime_final=runtime_final)
    return final


def atomic_save_case(case_dir, tensors, provenance, replicate_rows):
    """Publish a complete case directory atomically; partial staging is never complete."""
    case_dir = Path(case_dir)
    if case_dir.exists():
        raise FileExistsError(f'Case output already exists: {case_dir}')
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{case_dir.name}.', dir=case_dir.parent) as temp:
        stage = Path(temp)
        tensor_path = stage / 'case_deltaz.pt'
        torch.save(tensors, tensor_path)
        (stage / 'case_provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        csv_path = stage / 'replicate_provenance.csv'
        fields = ('case_id', 'window_index', 'start_d', 'start_h', 'start_w', 'stop_d', 'stop_h', 'stop_w',
                  'replicate_index', 'seed', 'timestep', 'noise_identity', 'support_fraction',
                  'local_deltaz_mean', 'local_deltaz_std', 'local_deltaz_min', 'local_deltaz_max')
        with csv_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows({field: row[field] for field in fields} for row in replicate_rows)
        checksums = _file_sha_map(stage)
        (stage / 'CHECKSUMS.sha256').write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(checksums.items())), encoding='utf-8')
        os.replace(stage, case_dir)


def verify_completed_case(case_dir, expected_identity):
    case_dir = Path(case_dir)
    required = {'case_deltaz.pt', 'case_provenance.json', 'replicate_provenance.csv', 'CHECKSUMS.sha256'}
    require(case_dir.is_dir() and {p.name for p in case_dir.iterdir()} == required,
            f'Incomplete or unexpected case artifact directory: {case_dir}')
    recorded = {}
    for line in (case_dir / 'CHECKSUMS.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        require(name not in recorded, 'Duplicate case checksum entry')
        recorded[name] = digest
    require(set(recorded) == required - {'CHECKSUMS.sha256'}, 'Case checksum inventory mismatch')
    for name, digest in recorded.items():
        require(sha256_file(case_dir / name) == digest, f'Completed case artifact checksum mismatch: {name}')
    provenance = json.loads((case_dir / 'case_provenance.json').read_text())
    require(provenance.get('run_identity') == expected_identity,
            'Existing case provenance does not match this inference run')
    return provenance


def _runtime_provenance(device=None):
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    if device is not None and torch.device(device).type == 'cuda' and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.device(device))
    else:
        device_name = 'CPU' if device is None or torch.device(device).type == 'cpu' else 'CUDA'
    return dict(runtime_git_branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
        runtime_git_commit=git('rev-parse', 'HEAD'), git_status_short=git('status', '--short').splitlines(),
        pytorch_version=torch.__version__, cuda_version=torch.version.cuda,
        device_name=device_name)


def _run_identity(*, mode, deps, checkpoints, dataset_root, runtime):
    return dict(mode=mode, scientific_analysis=(mode == 'scientific'), scientific_result=(mode == 'scientific'),
        dataset_root=str(Path(dataset_root).expanduser().resolve()),
        b4b_sha256=deps['b4b_contract_sha256'], spatial_contract_sha256=deps['spatial_contract_sha256'],
        spatial_manifest_sha256=deps['spatial_manifest_sha256'], d0a_sha256=deps['d0a_contract_sha256'],
        validation_split_sha256=VALIDATION_SPLIT_SHA256,
        e1c_checkpoint_sha256=checkpoints['e1c_checkpoint_sha256'],
        d0c2_checkpoint_sha256=checkpoints['d0c2_checkpoint_sha256'],
        runtime_git_commit=runtime['runtime_git_commit'],
        runtime_git_status_short=runtime['git_status_short'],
        timestep=495, alpha_bar=0.5000776052474976, ddim_evaluations=50, eta=0.0,
        replicates=4, global_seed=GLOBAL_SEED)


def _window_delta(source, diffusion, padded, raw, row, means, stds, device,
                  b4b, *, timesteps=(495,)):
    slices = tuple(slice(row[f'start_{axis}'], row[f'stop_{axis}']) for axis in 'dhw')
    norm_window = padded[(slice(None), *slices)].unsqueeze(0)
    raw_window = raw[(slice(None), *slices)]
    support = support_from_raw_window(raw_window).to(device)
    za = extract_za(source, norm_window, device)
    z0 = standardize_za(za, means, stds)
    require(z0.shape == (1, *LATENT_SHAPE), 'D0-A z0 shape mismatch')
    priors, noises, seeds, replicate_maps = {}, {}, {}, {}
    for timestep in timesteps:
        timestep_priors = []
        for replicate in REPLICATE_INDICES:
            seed, epsilon = make_partial_noise(row['case_id'], row, replicate, contract=b4b)
            epsilon = epsilon.to(device)
            prior, evaluations = ddim_reconstruct(diffusion, z0, support, epsilon, timestep, b4b_contract=b4b)
            require(evaluations == 50, 'Unexpected DDIM model evaluation count')
            timestep_priors.append(prior)
            noises[(timestep, replicate)] = epsilon
            seeds[(timestep, replicate)] = seed
        priors[timestep] = timestep_priors
    maps = {}
    for timestep, values in priors.items():
        maps[timestep], per_replicate = local_deltaz(z0, values, support)
        for replicate, replicate_map in enumerate(per_replicate):
            replicate_maps[(timestep, replicate)] = replicate_map
    return z0, support, maps, seeds, noises, replicate_maps


def run(*, dataset_root, e1c_checkpoint, diffusion_checkpoint, output_dir=None,
        scientific=False, smoke=False, resume=False, max_cases=1, max_windows=2,
        device=None, sensitivity=False):
    require(scientific != smoke, 'Select exactly one of --scientific or --smoke')
    require(not (smoke and resume), 'Smoke outputs are disposable and do not support resume')
    require(not (scientific and resume is False and output_dir is not None and Path(output_dir).exists()
                 and any(Path(output_dir).iterdir())),
            'Scientific output already exists; use --resume for verified completed cases')
    deps = verify_frozen_inputs()
    manifest_rows, by_case = parse_manifest()
    dataset_root = Path(dataset_root).expanduser().resolve()
    require(dataset_root.is_dir(), f'Dataset root does not exist: {dataset_root}')
    if output_dir is None:
        output_dir = DEFAULT_SCIENTIFIC_OUTPUT if scientific else DEFAULT_SMOKE_OUTPUT
    output = validate_output_mode_path(output_dir, scientific=scientific)
    require(not paths_overlap(output, dataset_root), 'Inference output must be separate from dataset root')
    if output.exists() and not resume:
        require(not any(output.iterdir()), f'Output directory is non-empty: {output}')
    provenance_filename = 'smoke_run_provenance.json' if smoke else 'run_provenance.json'
    allowed_output_names = {provenance_filename, 'smoke_cases' if smoke else 'cases',
                            'smoke_case_index.csv' if smoke else 'case_summary.csv'}
    if output.exists() and resume:
        require({p.name for p in output.iterdir()} <= allowed_output_names,
                'Resume output contains unexpected files')
        if any(output.iterdir()):
            require((output / provenance_filename).is_file(),
                    'Cannot resume existing inference output without its run provenance')
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA device requested but CUDA is unavailable')
    source, diffusion, checkpoint_info = load_inference_models(e1c_checkpoint, diffusion_checkpoint, device)
    before_source = model_fingerprint(source)
    means, stds = load_normalization_stats(deps['d0a'])
    b4b = deps['b4b']
    selected_cases = sorted(by_case)[:max_cases] if smoke else sorted(by_case)
    require(not smoke or max_cases > 0 and max_windows > 0, 'Smoke limits must be positive')
    runtime = _runtime_provenance(device)
    mode = 'smoke' if smoke else 'scientific'
    identity = _run_identity(mode=mode, deps=deps, checkpoints=checkpoint_info,
                             dataset_root=dataset_root, runtime=runtime)
    run_provenance = dict(stage='D0-C3a', status='RUNNING', run_identity=identity,
        scientific_analysis=scientific, scientific_result=scientific,
        smoke=(dict(scientific_analysis=False, scientific_result=False) if smoke else None),
        selected_source='D0-C2 EMA state_dict key ema_state',
        ground_truth_opened=False, pathology_corruption_invoked=False,
        checkpoint_info=checkpoint_info, frozen_inputs={k: v for k, v in deps.items() if not isinstance(v, (dict, list))},
        runtime=runtime, started_at=datetime.now(timezone.utc).isoformat(),
        resume_semantics='Completed case directories are checksum/provenance-verified before skipping; publication is atomic.')
    provenance_path = output / provenance_filename
    if provenance_path.exists():
        require(resume, 'Run provenance already exists; use --resume only for verified idempotent continuation')
        existing = json.loads(provenance_path.read_text())
        require(existing.get('run_identity') == identity, 'Run provenance mismatch on resume')
        run_provenance = mark_run_in_progress(existing, runtime=runtime)
        _atomic_json(provenance_path, run_provenance)
    else:
        _atomic_json(provenance_path, run_provenance)
    output_cases = output / ('smoke_cases' if smoke else 'cases')
    require(not output_cases.is_symlink(), 'Inference case-output directory cannot be a symlink')
    output_cases.mkdir(exist_ok=True)
    scheduled_timesteps = (495, 327, 663) if sensitivity else (495,)
    all_summary_rows = []
    for case_id in selected_cases:
        case_rows = by_case[case_id]
        if smoke:
            case_rows = case_rows[:max_windows]
        case_output = output_cases / case_id
        if case_output.exists():
            require(resume, f'Case already exists; use --resume to verify and skip: {case_output}')
            previous = verify_completed_case(case_output, identity)
            all_summary_rows.append(dict(case_id=case_id, status='already_verified',
                window_count=previous.get('window_count'),
                supported_deltaz_mean=previous.get('deltaz_supported_mean')))
            continue
        raw_original, raw_padded, normalized_padded = prepare_case_mri(dataset_root, case_id, case_rows)
        local_maps, local_supports, replicate_rows = [], [], []
        smoke_reproducible = True
        for row in case_rows:
            z0, support, maps, seeds, noise_cache, replicate_maps = _window_delta(
                source, diffusion, normalized_padded, raw_padded, row, means, stds, device, b4b,
                timesteps=scheduled_timesteps)
            local_maps.append(maps[495].cpu())
            local_supports.append(support.cpu())
            for timestep in scheduled_timesteps:
                for replicate in REPLICATE_INDICES:
                    replicate_map = replicate_maps[(timestep, replicate)]
                    support_values = replicate_map.masked_select(support)
                    replicate_rows.append(dict(case_id=case_id, window_index=row['window_index'],
                        **{key: row[key] for key in ('start_d', 'start_h', 'start_w', 'stop_d', 'stop_h', 'stop_w')},
                        replicate_index=replicate, seed=seeds[(timestep, replicate)], timestep=timestep,
                        noise_identity=json.dumps(sample_identity(case_id, row), ensure_ascii=False, separators=(',', ':')),
                        support_fraction=float(support.float().mean()),
                        local_deltaz_mean=float(support_values.mean()),
                        local_deltaz_std=float(support_values.std(unbiased=False)),
                        local_deltaz_min=float(support_values.min()), local_deltaz_max=float(support_values.max())))
            if smoke and row['window_index'] == case_rows[0]['window_index']:
                rerun_maps = {}
                for timestep in scheduled_timesteps:
                    rerun_priors = [ddim_reconstruct(diffusion, z0, support,
                        noise_cache[(timestep, replicate)], timestep, b4b_contract=b4b)[0]
                        for replicate in REPLICATE_INDICES]
                    rerun_maps[timestep] = local_deltaz(z0, rerun_priors, support)[0]
                smoke_reproducible = all(torch.equal(maps[t], rerun_maps[t]) for t in maps)
                require(smoke_reproducible, 'Smoke deterministic rerun equality failed')
        if not smoke or len(case_rows) == 36:
            global_map, global_valid, counts = place_and_aggregate(local_maps, local_supports, case_rows)
        else:
            # A bounded smoke subset is never represented as a complete case-level map.
            global_map = global_valid = counts = None
        if smoke:
            tensor_payload = dict(local_DeltaZ=torch.stack([m[0] for m in local_maps]),
                local_support=torch.stack([m[0] for m in local_supports]),
                processed_window_indices=torch.tensor([r['window_index'] for r in case_rows]),
                complete_case_aggregation=False)
        else:
            tensor_payload = dict(DeltaZ_global=global_map, valid_support_global=global_valid,
                contribution_count_global=counts, original_shape=torch.tensor(raw_original.shape[1:]),
                padded_shape=torch.tensor(normalized_padded.shape[1:]), global_latent_shape=torch.tensor(GLOBAL_SHAPE))
            if sensitivity:
                # Secondary timestep maps retain the same valid-only overlap rule.
                for timestep in (327, 663):
                    maps_for_t = []
                    for row in case_rows:
                        # Re-run is intentionally deterministic, using exactly the same per-sample seeds.
                        z0, support, maps, _, _, _ = _window_delta(source, diffusion, normalized_padded,
                            raw_padded, row, means, stds, device, b4b, timesteps=(timestep,))
                        maps_for_t.append(maps[timestep].cpu())
                    tensor_payload[f'DeltaZ_t{timestep}_global'] = place_and_aggregate(
                        maps_for_t, local_supports, case_rows)[0]
        if global_map is not None:
            stats_values = global_map.masked_select(global_valid)
            case_support_fraction = float(global_valid.float().mean())
        else:
            stats_values = torch.cat([m.masked_select(s) for m, s in zip(local_maps, local_supports)])
            case_support_fraction = float(np.mean([float(s.float().mean()) for s in local_supports]))
        support_cells = sum(int(s.sum()) for s in local_supports)
        case_provenance = dict(run_identity=identity, case_id=case_id, split='validation',
            split_sha256=VALIDATION_SPLIT_SHA256,
            window_count=len(case_rows), complete_case=(len(case_rows) == 36),
            tensor_shapes={key: list(value.shape) for key, value in tensor_payload.items() if isinstance(value, torch.Tensor)},
            original_shape=list(raw_original.shape[1:]), padded_shape=list(normalized_padded.shape[1:]),
            global_latent_shape=list(GLOBAL_SHAPE), local_window_size=list(WINDOW_SHAPE),
            support_valid_cells_local_sum=support_cells,
            support_fraction_global=case_support_fraction,
            deltaz_supported_value_count=int(stats_values.numel()),
            deltaz_supported_mean=float(stats_values.mean()) if stats_values.numel() else None,
            deltaz_supported_std=float(stats_values.std(unbiased=False)) if stats_values.numel() else None,
            deltaz_supported_min=float(stats_values.min()) if stats_values.numel() else None,
            deltaz_supported_max=float(stats_values.max()) if stats_values.numel() else None,
            timesteps=list(scheduled_timesteps), alpha_bar={str(t): float(cosine_schedule()['alpha_bar'][t]) for t in scheduled_timesteps},
            replicates=4, ddim_evaluations=50, eta=0.0, global_seed=GLOBAL_SEED,
            source_checkpoint_sha256=checkpoint_info['e1c_checkpoint_sha256'],
            diffusion_checkpoint_sha256=checkpoint_info['d0c2_checkpoint_sha256'],
            b4b_contract_sha256=deps['b4b_contract_sha256'], spatial_contract_sha256=deps['spatial_contract_sha256'],
            spatial_manifest_sha256=deps['spatial_manifest_sha256'], d0a_contract_sha256=deps['d0a_contract_sha256'],
            runtime=runtime, execution_device=str(device), ground_truth_used=False,
            pathology_corruption_invoked=False, source_model_fingerprint_before=before_source,
            source_model_fingerprint_after=model_fingerprint(source), source_model_unchanged=(before_source == model_fingerprint(source)),
            smoke_deterministic_rerun_equal=(smoke_reproducible if smoke else None))
        require(case_provenance['source_model_unchanged'], 'Frozen E1-C source parameters changed during inference')
        atomic_save_case(case_output, tensor_payload, case_provenance, replicate_rows)
        all_summary_rows.append(dict(case_id=case_id, status='completed', **{
            'window_count': len(case_rows), 'supported_deltaz_mean': case_provenance['deltaz_supported_mean']}))
    # Aggregate index is generation metadata only; no pathology fields are read or computed.
    index_name = 'smoke_case_index.csv' if smoke else 'case_summary.csv'
    temp_index = output / (index_name + '.tmp')
    with temp_index.open('w', newline='', encoding='utf-8') as stream:
        fields = sorted({key for row in all_summary_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(all_summary_rows)
    os.replace(temp_index, output / index_name)
    final_runtime = _runtime_provenance(device)
    final = finalize_run_provenance(run_provenance,
        completed_at=datetime.now(timezone.utc).isoformat(),
        cases_processed=len(all_summary_rows),
        windows_processed=sum(int(row.get('window_count', 0)) for row in all_summary_rows),
        source_model_unchanged=(before_source == model_fingerprint(source)), runtime_final=final_runtime)
    _atomic_json(provenance_path, final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--scientific', action='store_true')
    modes.add_argument('--smoke', action='store_true')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--e1c-checkpoint', type=Path, default=DEFAULT_E1C)
    parser.add_argument('--diffusion-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-cases', type=int)
    parser.add_argument('--max-windows', type=int)
    parser.add_argument('--device', default=None)
    parser.add_argument('--sensitivity', action='store_true')
    args = parser.parse_args(argv)
    if not args.smoke and (args.max_cases is not None or args.max_windows is not None):
        parser.error('--max-cases and --max-windows are smoke-only')
    if args.smoke:
        args.max_cases = 1 if args.max_cases is None else args.max_cases
        args.max_windows = 2 if args.max_windows is None else args.max_windows
    if args.smoke and args.resume:
        parser.error('Smoke outputs are disposable and do not support resume')
    if args.scientific and args.sensitivity:
        # Sensitivity is a separate predefined secondary output path; kept opt-in to avoid
        # silently doubling work in the primary fixed protocol.
        pass
    result = run(dataset_root=args.dataset_root, e1c_checkpoint=args.e1c_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint, output_dir=args.output_dir,
        scientific=args.scientific, smoke=args.smoke, resume=args.resume,
        max_cases=args.max_cases or 1, max_windows=args.max_windows or 2, device=args.device,
        sensitivity=args.sensitivity)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
