#!/usr/bin/env python3
"""Generate and freeze the pre-inference D0-C3a spatial design manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from datasets.transforms import normalize_nonzero
CONTRACT_PATH = ROOT / 'configs/d0c3_spatial_inference_contract.json'
B4B_PATH = ROOT / 'configs/d0b4b_inference_contract.json'
B4_FINAL_PATH = ROOT / 'configs/d0b4_final_design_freeze.json'
D0A_PATH = ROOT / 'outputs/frozen/d0_latent_distribution_audit/d0_za_normalization_contract.json'
E1C_PATH = ROOT / 'outputs/frozen/e1c_anatomy_constraint/training/best.pt'
C2_DIR = ROOT / 'outputs/frozen/d0c2_scientific_diffusion_training'
C2_SUMMARY_PATH = C2_DIR / 'd0c2_training_summary.json'
C2_ARTIFACTS_PATH = C2_DIR / 'd0c2_artifact_manifest.json'
VAL_SPLIT_PATH = ROOT / 'splits/val_cases.txt'
DEFAULT_OUTPUT = ROOT / 'outputs/frozen/d0c3_spatial_inference_design'

EXPECTED = {
    'd0b4b': '29cf659528f6257cd9f8f218e9f4eaf8fabdc6d8a0e4905505aa8548ae68bde4',
    'd0b4_final': '05d13292994b59f3c45d70c98591c306970269673e475797620c644446aa86d8',
    'd0a': 'bca5c0f8f1617d73fc4d0850fe5b6f45e4b2bcbacbb4018f1d5d825de49ad792',
    'e1c': 'cf5a04a2d845cecb45a0ea87d617abad0afab9ad4264abce8342a6045563ee53',
    'c2_best': 'aaa6258f26a56c852dc751b8028bfb046da5752974d4cbaf446ad2699a0c6574',
    'validation_split': 'ec5fe6b0bbcf190fe8b7c032f8da4ff1d495969c2d868561e7a3351c572cc5ca',
}
KNOWN_ORIGINAL_SHAPE = (182, 218, 182)
WINDOW_SHAPE = (96, 96, 96)
WINDOW_STRIDE = (48, 48, 48)
ENCODER_STRIDE = 8
GLOBAL_SEED = 20260923
REPLICATES = (0, 1, 2, 3)
MANIFEST_FIELDS = (
    'case_id', 'split', 'window_index',
    'original_d', 'original_h', 'original_w',
    'padded_d', 'padded_h', 'padded_w',
    'start_d', 'start_h', 'start_w', 'stop_d', 'stop_h', 'stop_w',
    'latent_start_d', 'latent_start_h', 'latent_start_w',
    'latent_stop_d', 'latent_stop_h', 'latent_stop_w',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sha(path, expected, label):
    path = Path(path)
    require(path.is_file(), f'Missing {label}: {path}')
    actual = sha256_file(path)
    require(actual == expected, f'{label} SHA256 mismatch: {actual} != {expected}')
    return actual


def verify_sha_manifest(directory):
    directory = Path(directory)
    manifest = directory / 'MANIFEST.sha256'
    require(manifest.is_file(), f'Missing frozen dependency manifest: {manifest}')
    checked = {}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest, relative = line.split('  ', 1)
        require(relative not in checked, f'Duplicate hash-manifest path: {relative}')
        path = directory / relative
        require(path.is_file(), f'Missing hash-manifest entry: {relative}')
        actual = sha256_file(path)
        require(actual == digest, f'Frozen dependency manifest hash mismatch: {relative}')
        checked[relative] = actual
    files = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
             if p.is_file() and p.name != 'MANIFEST.sha256'}
    require(files == set(checked), 'Frozen dependency manifest file set mismatch')
    return sha256_file(manifest), checked


def verify_dependencies():
    _validate_design_contract()
    b4b_sha = verify_sha(B4B_PATH, EXPECTED['d0b4b'], 'D0-B4b inference contract')
    b4_final_sha = verify_sha(B4_FINAL_PATH, EXPECTED['d0b4_final'], 'D0-B4 final design freeze')
    d0a_sha = verify_sha(D0A_PATH, EXPECTED['d0a'], 'D0-A normalization contract')
    e1c_sha = verify_sha(E1C_PATH, EXPECTED['e1c'], 'E1-C checkpoint')
    split_sha = verify_sha(VAL_SPLIT_PATH, EXPECTED['validation_split'], 'validation split')
    b4b = json.loads(B4B_PATH.read_text(encoding='utf-8'))
    require(b4b['randomness']['seed_hash'] == 'SHA256'
            and b4b['randomness']['seed_integer'] == 'int(sha256(payload_bytes).hexdigest()[:16], 16) & ((1 << 63) - 1)'
            and b4b['randomness']['seed_payload'] == '[inference_global_seed, sample_identity, replicate_index, purpose]'
            and b4b['randomness']['diffusion_timestep_in_seed_payload'] is False,
            'D0-B4b deterministic noise-seed derivation is missing or changed')
    require(b4b['randomness']['inference_global_seed'] == GLOBAL_SEED
            and b4b['randomness']['primary_noise_replicates'] == 4
            and b4b['randomness']['purpose'] == 'partial_noise',
            'D0-B4b noise identity values mismatch')
    c2_manifest_sha, c2_entries = verify_sha_manifest(C2_DIR)
    summary = json.loads(C2_SUMMARY_PATH.read_text(encoding='utf-8'))
    artifacts = json.loads(C2_ARTIFACTS_PATH.read_text(encoding='utf-8'))
    require(summary.get('stage') == 'D0-C2' and summary.get('status') == 'CLOSED_FROZEN',
            'D0-C2 freeze summary is not closed')
    require(summary.get('best_human_epoch') == 100
            and summary.get('best_checkpoint_sha256') == EXPECTED['c2_best'],
            'D0-C2 selected checkpoint reference mismatch')
    checkpoint_ref = artifacts.get('artifacts', {}).get('best.pt', {})
    require(checkpoint_ref.get('sha256') == EXPECTED['c2_best'],
            'D0-C2 artifact manifest selected checkpoint SHA mismatch')
    require(c2_entries.get('d0c2_training_summary.json') == sha256_file(C2_SUMMARY_PATH)
            and c2_entries.get('d0c2_artifact_manifest.json') == sha256_file(C2_ARTIFACTS_PATH),
            'D0-C2 metadata is not covered by its MANIFEST.sha256')
    checkpoint_path = Path(checkpoint_ref['path'])
    return dict(
        d0b4b_contract_path=str(B4B_PATH.relative_to(ROOT)), d0b4b_contract_sha256=b4b_sha,
        d0b4_final_freeze_path=str(B4_FINAL_PATH.relative_to(ROOT)), d0b4_final_freeze_sha256=b4_final_sha,
        d0a_normalization_contract_path=str(D0A_PATH.relative_to(ROOT)), d0a_normalization_contract_sha256=d0a_sha,
        e1c_checkpoint_path=str(E1C_PATH.relative_to(ROOT)), e1c_checkpoint_sha256=e1c_sha,
        d0c2_best_checkpoint_path=str(checkpoint_path), d0c2_best_checkpoint_sha256=EXPECTED['c2_best'],
        d0c2_best_checkpoint_bytes_verified_by_design_freeze=False,
        d0c2_best_checkpoint_verification_note=(
            'Checkpoint bytes and EMA state are deliberately not required or verified by this design freeze; verify on the real server before inference.'),
        d0c2_freeze_manifest_path=str((C2_DIR / 'MANIFEST.sha256').relative_to(ROOT)),
        d0c2_freeze_manifest_sha256=c2_manifest_sha,
        validation_split_path=str(VAL_SPLIT_PATH.relative_to(ROOT)), validation_split_sha256=split_sha)


def _validate_design_contract():
    contract = json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
    require(contract.get('stage') == 'D0-C3a-spatial-design'
            and contract.get('status') == 'CLOSED_FROZEN_PRE_INFERENCE',
            'Spatial design contract stage/status mismatch')
    require(contract.get('locked_dependencies', {}).get('d0b4b_inference_contract_sha256') == EXPECTED['d0b4b']
            and contract['locked_dependencies'].get('d0b4_final_design_freeze_sha256') == EXPECTED['d0b4_final']
            and contract['locked_dependencies'].get('d0a_normalization_contract_sha256') == EXPECTED['d0a']
            and contract['locked_dependencies'].get('e1c_checkpoint_sha256') == EXPECTED['e1c']
            and contract['locked_dependencies'].get('d0c2_best_checkpoint_expected_sha256') == EXPECTED['c2_best'],
            'Spatial design contract dependency references mismatch')
    geometry = contract.get('window_geometry', {})
    require(geometry.get('window_size') == list(WINDOW_SHAPE)
            and geometry.get('stride') == list(WINDOW_STRIDE)
            and geometry.get('axis_starts_for_known_shape') == {'d': [0, 48, 88], 'h': [0, 48, 96, 128], 'w': [0, 48, 88]}
            and geometry.get('windows_per_case') == 36
            and geometry.get('expected_manifest_rows') == 9720,
            'Spatial design geometry contract mismatch')
    require(contract.get('input_and_preprocessing', {}).get('normalization_precedes_padding') is True
            and contract['input_and_preprocessing'].get('known_padded_shape') == [184, 224, 184],
            'Spatial design preprocessing contract mismatch')
    require(contract.get('case_level_overlap_aggregation', {}).get('primary_rule') == 'uniform_mean_valid_support_only'
            and contract.get('support', {}).get('occupancy_mapping') == 'exact non-overlapping 8x8x8 voxel-block occupancy'
            and contract['support'].get('valid_rule') == 'occupancy >= 0.50',
            'Spatial design support/aggregation contract mismatch')
    require(contract.get('noise_identity', {}).get('seed_derivation') == 'int(sha256(payload_bytes).hexdigest()[:16], 16) & ((1 << 63) - 1)'
            and contract['noise_identity'].get('seed_payload') == '[inference_global_seed, sample_identity, replicate_index, purpose]'
            and contract['noise_identity'].get('diffusion_timestep_in_seed_payload') is False,
            'Spatial design randomness contract mismatch')
    return contract


def padded_shape(original_shape, window_shape=WINDOW_SHAPE, encoder_stride=ENCODER_STRIDE):
    if len(original_shape) != 3 or len(window_shape) != 3:
        raise ValueError('Spatial shapes must have three dimensions')
    result = []
    for original, window in zip(original_shape, window_shape):
        original, window = int(original), int(window)
        if original <= 0 or window <= 0:
            raise ValueError('Spatial dimensions must be positive')
        target_min = max(original, window)
        result.append(((target_min + encoder_stride - 1) // encoder_stride) * encoder_stride)
    return tuple(result)


def axis_window_starts(padded_size, window_size=96, stride=48, alignment=8):
    if padded_size < window_size:
        raise ValueError('Padded dimension is smaller than the window')
    starts = list(range(0, padded_size - window_size + 1, stride))
    high_start = padded_size - window_size
    if not starts or starts[-1] != high_start:
        starts.append(high_start)
    if any(start % alignment for start in starts):
        raise ValueError(f'Window start is not aligned to {alignment}: {starts}')
    if any(start < 0 or start + window_size > padded_size for start in starts):
        raise ValueError('Window coordinate is out of bounds')
    return starts


def window_geometry(original_shape=KNOWN_ORIGINAL_SHAPE):
    padded = padded_shape(original_shape)
    starts = [axis_window_starts(size, window, stride)
              for size, window, stride in zip(padded, WINDOW_SHAPE, WINDOW_STRIDE)]
    return padded, starts


def build_manifest_rows(case_ids, *, split='validation', original_shape=KNOWN_ORIGINAL_SHAPE):
    if split != 'validation':
        raise ValueError('D0-C3a primary case set is the locked validation split only')
    if len(case_ids) != len(set(case_ids)):
        raise ValueError('Validation case IDs must be unique')
    ordered_cases = sorted(str(case_id) for case_id in case_ids)
    padded, axes = window_geometry(original_shape)
    rows = []
    for case_id in ordered_cases:
        for window_index, starts in enumerate(itertools.product(*axes)):
            stops = tuple(start + size for start, size in zip(starts, WINDOW_SHAPE))
            values = dict(case_id=case_id, split='validation', window_index=window_index)
            values.update({f'original_{axis}': int(size) for axis, size in zip('dhw', original_shape)})
            values.update({f'padded_{axis}': int(size) for axis, size in zip('dhw', padded)})
            values.update({f'start_{axis}': int(value) for axis, value in zip('dhw', starts)})
            values.update({f'stop_{axis}': int(value) for axis, value in zip('dhw', stops)})
            values.update({f'latent_start_{axis}': int(value // 8) for axis, value in zip('dhw', starts)})
            values.update({f'latent_stop_{axis}': int(value // 8) for axis, value in zip('dhw', stops)})
            rows.append(values)
    return rows


def manifest_bytes(rows):
    import io
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows({field: row[field] for field in MANIFEST_FIELDS} for row in rows)
    return output.getvalue().encode('utf-8')


def normalize_full_then_pad(image, *, normalizer=normalize_nonzero, window_shape=WINDOW_SHAPE,
                            encoder_stride=ENCODER_STRIDE):
    """Normalize the original volume first, then add only high-end zero padding."""
    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError('Expected original MRI [4,D,H,W]')
    original_shape = tuple(int(size) for size in image.shape[1:])
    normalized = normalizer(image)
    target_shape = padded_shape(original_shape, window_shape, encoder_stride)
    pad_d, pad_h, pad_w = (target - source for source, target in zip(original_shape, target_shape))
    padded = F.pad(normalized, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
    return normalized, padded, original_shape, target_shape


def local_to_global_slices(window_start, local_shape=(12, 12, 12)):
    if len(window_start) != 3 or any(int(value) % 8 for value in window_start):
        raise ValueError('Raw window starts must be divisible by 8')
    starts = tuple(int(value) // 8 for value in window_start)
    return tuple(slice(start, start + int(length)) for start, length in zip(starts, local_shape))


def aggregate_valid_windows(local_maps, local_masks, window_starts, global_shape=(23, 28, 23)):
    """Uniform valid-support-only overlap mean with explicit invalid cells."""
    if not (len(local_maps) == len(local_masks) == len(window_starts)):
        raise ValueError('Maps, support masks, and starts must have equal lengths')
    sums = torch.zeros((1, *global_shape), dtype=torch.float32)
    counts = torch.zeros((1, *global_shape), dtype=torch.int64)
    for local_map, local_mask, start in zip(local_maps, local_masks, window_starts):
        value = torch.as_tensor(local_map, dtype=torch.float32)
        mask = torch.as_tensor(local_mask, dtype=torch.bool)
        if value.shape == (12, 12, 12):
            value = value.unsqueeze(0)
        if mask.shape == (12, 12, 12):
            mask = mask.unsqueeze(0)
        if value.shape != (1, 12, 12, 12) or mask.shape != value.shape:
            raise ValueError('Expected local DeltaZ/support [1,12,12,12]')
        region = local_to_global_slices(start)
        sums[(slice(None), *region)] += value * mask
        counts[(slice(None), *region)] += mask.to(torch.int64)
    valid = counts > 0
    mean = torch.where(valid, sums / counts.clamp_min(1), torch.zeros_like(sums))
    return mean, valid, counts


def partial_noise_seed(case_id, starts, replicate_index, *, global_seed=GLOBAL_SEED):
    stops = tuple(int(start) + size for start, size in zip(starts, WINDOW_SHAPE))
    identity = [str(case_id), *map(int, starts), *stops]
    payload = [int(global_seed), identity, int(replicate_index), 'partial_noise']
    payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int(hashlib.sha256(payload_bytes).hexdigest()[:16], 16) & ((1 << 63) - 1)


def partial_noise(case_id, starts, replicate_index, *, global_seed=GLOBAL_SEED):
    seed = partial_noise_seed(case_id, starts, replicate_index, global_seed=global_seed)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return seed, torch.randn((256, 12, 12, 12), dtype=torch.float32, device='cpu', generator=generator)


def _manifest_file_bytes(directory):
    lines = []
    for path in sorted(Path(directory).rglob('*')):
        if path.is_file() and path.name != 'MANIFEST.sha256':
            lines.append(f'{sha256_file(path)}  {path.relative_to(directory).as_posix()}')
    return ('\n'.join(lines) + '\n').encode('utf-8')


def _load_validation_ids():
    raw = VAL_SPLIT_PATH.read_text(encoding='utf-8').splitlines()
    ids = [line.strip() for line in raw if line.strip()]
    require(len(ids) == 270 and len(set(ids)) == 270, 'Expected exactly 270 unique validation case IDs')
    return ids


def _summary(rows):
    padded, axes = window_geometry()
    return dict(stage='D0-C3a-spatial-design', status='CLOSED_FROZEN_PRE_INFERENCE',
        scientific_delta_z_generated=False, scientific_result_observed=False,
        primary_case_set='DEVELOPMENT_VALIDATION', case_count=270,
        validation_split_sha256=EXPECTED['validation_split'],
        window_size=list(WINDOW_SHAPE), stride=list(WINDOW_STRIDE),
        axis_starts={axis: values for axis, values in zip('dhw', axes)},
        windows_per_case=math.prod(len(values) for values in axes), manifest_rows=len(rows),
        original_shape=list(KNOWN_ORIGINAL_SHAPE), padded_shape=list(padded),
        global_latent_shape=[size // ENCODER_STRIDE for size in padded],
        local_latent_shape=[12, 12, 12], aggregation='uniform_mean_valid_support_only',
        independent_final_evaluation='separate/TBD',
        freeze_statement='This spatial design was frozen before any D0-C3 DeltaZ scientific result was generated.',
        new_design_completion=True,
        note='Spatial sampling and overlap aggregation complete the pre-inference design; these choices were not specified by D0-B4b.')


def _canonical_manifest_manifest(directory):
    return _manifest_file_bytes(directory)


def _verify_output(output, rows, deps):
    output = Path(output)
    manifest_path = output / 'MANIFEST.sha256'
    require(manifest_path.is_file(), 'Spatial design MANIFEST.sha256 is missing')
    recorded = {}
    for line in manifest_path.read_text(encoding='utf-8').splitlines():
        digest, relpath = line.split('  ', 1)
        require(relpath not in recorded, f'Duplicate frozen manifest entry: {relpath}')
        recorded[relpath] = digest
    actual_files = {path.relative_to(output).as_posix() for path in output.rglob('*')
                    if path.is_file() and path.name != 'MANIFEST.sha256'}
    require(actual_files == {'inference_window_manifest.csv', 'spatial_design_summary.json', 'provenance.json'},
            'Unexpected files in spatial design freeze')
    require(set(recorded) == actual_files, 'Spatial design MANIFEST file set mismatch')
    for relpath, digest in recorded.items():
        require(sha256_file(output / relpath) == digest, f'Spatial design MANIFEST hash mismatch: {relpath}')
    expected_bytes = manifest_bytes(rows)
    require((output / 'inference_window_manifest.csv').read_bytes() == expected_bytes,
            'Frozen inference window manifest differs from deterministic geometry')
    summary = json.loads((output / 'spatial_design_summary.json').read_text(encoding='utf-8'))
    require(summary == _summary(rows), 'Spatial design summary mismatch')
    provenance = json.loads((output / 'provenance.json').read_text(encoding='utf-8'))
    require(provenance.get('locked_dependencies') == deps, 'Spatial design dependency provenance mismatch')
    require(provenance.get('contract_sha256') == sha256_file(CONTRACT_PATH),
            'Spatial design contract SHA mismatch')
    require(provenance.get('manifest_row_count') == len(rows)
            and provenance.get('manifest_sha256') == hashlib.sha256(expected_bytes).hexdigest(),
            'Spatial design manifest provenance mismatch')
    return dict(manifest_sha256=sha256_file(output / 'inference_window_manifest.csv'),
                frozen_manifest_sha256=sha256_file(manifest_path), summary=summary)


def freeze(output=DEFAULT_OUTPUT, *, verify_only=False):
    deps = verify_dependencies()
    ids = _load_validation_ids()
    rows = build_manifest_rows(ids)
    require(len(rows) == 9720 and all(row['split'] == 'validation' for row in rows),
            'Expected 9720 validation-only inference windows')
    output = Path(output).expanduser().resolve()
    if verify_only:
        return _verify_output(output, rows, deps)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing spatial design freeze: {output}')
    config_sha = sha256_file(CONTRACT_PATH)
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    summary = _summary(rows)
    provenance = dict(stage='D0-C3a-spatial-design', created_at=datetime.now(timezone.utc).isoformat(),
        runtime_git_branch=git('rev-parse', '--abbrev-ref', 'HEAD'), runtime_git_commit=git('rev-parse', 'HEAD'),
        git_status_short=git('status', '--short').splitlines(), contract_path=str(CONTRACT_PATH.relative_to(ROOT)),
        contract_sha256=config_sha, locked_dependencies=deps,
        e1c_sha_audit_note=('The request-provided E1-C SHA string had 69 hex characters; the actual checkpoint and locked B4b contract agree on the valid 64-character SHA recorded here.'),
        case_set='DEVELOPMENT_VALIDATION', case_count=len(ids),
        manifest_path='outputs/frozen/d0c3_spatial_inference_design/inference_window_manifest.csv',
        manifest_row_count=len(rows), manifest_sha256=hashlib.sha256(manifest_bytes(rows)).hexdigest(),
        ground_truth_used=False, delta_z_generated=False,
        design_status='frozen before any D0-C3 DeltaZ scientific result')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.d0c3_spatial_design.', dir=output.parent) as tmp:
        stage = Path(tmp)
        (stage / 'inference_window_manifest.csv').write_bytes(manifest_bytes(rows))
        (stage / 'spatial_design_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
        (stage / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
        (stage / 'MANIFEST.sha256').write_bytes(_manifest_file_bytes(stage))
        _verify_output(stage, rows, deps)
        os.replace(stage, output)
    return _verify_output(output, rows, deps)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    result = freeze(args.output_dir, verify_only=args.verify_only)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
