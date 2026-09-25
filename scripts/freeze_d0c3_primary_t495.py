#!/usr/bin/env python3
"""Freeze integrity evidence for the completed primary D0-C3 t=495 generation.

This utility reads the scientific run written by ``run_d0c3_deltaz_inference``
and emits compact metadata only. It never copies the case tensor artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_DEFAULT = ROOT / 'outputs/scientific/d0c3_deltaz/primary_t495'
SPATIAL_MANIFEST = ROOT / 'outputs/frozen/d0c3_spatial_inference_design/inference_window_manifest.csv'
OUTPUT_DEFAULT = ROOT / 'outputs/frozen/d0c3_primary_t495'
EXPECTED = {
    'runtime_git_commit': '75e0d3868b73a947d53534631ef2a6f9d83ebb04',
    'runtime_git_branch': 'experiment/d0c3-scientific-generation',
    'b4b_contract_sha256': '29cf659528f6257cd9f8f218e9f4eaf8fabdc6d8a0e4905505aa8548ae68bde4',
    'spatial_contract_sha256': '832b8b04e411802186ddc5ba0fb598fa748d53876c87c282b950863919ef986c',
    'spatial_manifest_sha256': '58fac2ebef778d4d19d6da9f21451c25175c61bb4100c03387e8d3c859c05d53',
    'source_checkpoint_sha256': 'cf5a04a2d845cecb45a0ea87d617abad0afab9ad4264abce8342a6045563ee53',
    'diffusion_checkpoint_sha256': 'aaa6258f26a56c852dc751b8028bfb046da5752974d4cbaf446ad2699a0c6574',
    'd0a_contract_sha256': 'bca5c0f8f1617d73fc4d0850fe5b6f45e4b2bcbacbb4018f1d5d825de49ad792',
    'validation_split_sha256': 'ec5fe6b0bbcf190fe8b7c032f8da4ff1d495969c2d868561e7a3351c572cc5ca',
}
CASE_FILES = ('case_deltaz.pt', 'case_provenance.json', 'replicate_provenance.csv')
CASE_MANIFEST_FIELDS = ('case_id', 'window_count', 'timestep', 'support_fraction_global',
                        'case_deltaz_sha256', 'case_provenance_sha256',
                        'replicate_provenance_sha256', 'checksums_sha256')
FREEZE_FILES = ('d0c3_primary_summary.json', 'd0c3_primary_case_manifest.csv',
                'd0c3_primary_integrity.json', 'provenance.json')


def require(test, message):
    if not test:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def expected_cases_from_manifest(path=SPATIAL_MANIFEST, expected_sha=EXPECTED['spatial_manifest_sha256']):
    path = Path(path)
    require(path.is_file(), f'Missing frozen spatial manifest: {path}')
    require(sha256_file(path) == expected_sha, 'Spatial manifest SHA256 mismatch')
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        require('case_id' in (reader.fieldnames or []), 'Spatial manifest lacks case_id')
        rows = list(reader)
    ids = [row['case_id'] for row in rows]
    require(len(rows) == 9720 and len(set(ids)) == 270, 'Frozen spatial manifest case/window inventory mismatch')
    counts = {case_id: ids.count(case_id) for case_id in set(ids)}
    require(all(count == 36 for count in counts.values()), 'Spatial manifest does not have 36 windows per case')
    return sorted(counts)


def _verify_run_provenance(source, runtime_status=None, *, verify_checkpoint_bytes=True):
    run = read_json(source / 'run_provenance.json')
    require(run.get('status') == 'PASS', 'Scientific run status is not PASS')
    require(run.get('scientific_analysis') is True and run.get('scientific_result') is True,
            'Scientific run flags are not true')
    require(run.get('stage') == 'D0-C3a', 'Unexpected source run stage')
    identity = run.get('run_identity')
    require(isinstance(identity, dict) and identity.get('mode') == 'scientific', 'Run identity is not scientific')
    for key in ('runtime_git_commit', 'b4b_contract_sha256', 'spatial_contract_sha256',
                'spatial_manifest_sha256', 'd0a_contract_sha256', 'validation_split_sha256',
                'e1c_checkpoint_sha256', 'd0c2_checkpoint_sha256'):
        actual = identity.get(key)
        identity_key = {'b4b_contract_sha256': 'b4b_sha256', 'd0a_contract_sha256': 'd0a_sha256'}.get(key, key)
        expected_key = {'e1c_checkpoint_sha256': 'source_checkpoint_sha256',
                        'd0c2_checkpoint_sha256': 'diffusion_checkpoint_sha256'}.get(key, key)
        require(identity.get(identity_key) == EXPECTED[expected_key],
                f'Run identity mismatch for {identity_key}: {identity.get(identity_key)!r}')
    require(identity.get('runtime_git_commit') == EXPECTED['runtime_git_commit'], 'Runtime commit mismatch')
    require(identity.get('timestep') == 495 and identity.get('alpha_bar') == 0.5000776052474976
            and identity.get('ddim_evaluations') == 50 and identity.get('eta') == 0.0
            and identity.get('replicates') == 4 and identity.get('global_seed') == 20260923,
            'Primary inference settings mismatch')
    require(run.get('selected_source') == 'D0-C2 EMA state_dict key ema_state', 'Diffusion source is not frozen EMA')
    runtime = run.get('runtime') or {}
    require(runtime.get('runtime_git_branch') == EXPECTED['runtime_git_branch'], 'Runtime branch mismatch')
    require(runtime.get('runtime_git_commit') == EXPECTED['runtime_git_commit'], 'Run runtime commit mismatch')
    require(runtime.get('git_status_short') == [], 'Run runtime tree was not clean')
    require(runtime.get('device_name') == 'NVIDIA GeForce RTX 3090', 'GPU identity mismatch')
    require(runtime.get('pytorch_version') == '2.14.0+cu130' and runtime.get('cuda_version') == '13.0',
            'Run software environment mismatch')
    final_runtime = run.get('runtime_final') or {}
    require(final_runtime.get('runtime_git_branch') == EXPECTED['runtime_git_branch']
            and final_runtime.get('runtime_git_commit') == EXPECTED['runtime_git_commit']
            and final_runtime.get('git_status_short') == [], 'Final runtime provenance mismatch')
    require(run.get('cases_processed') == run.get('run_identity', {}).get('case_count', 270) and run.get('windows_processed') == run.get('cases_processed', 270) * 36
            and run.get('source_model_unchanged') is True, 'Run completion counts/source integrity mismatch')
    checkpoints = run.get('checkpoint_info') or {}
    for path_key, hash_key, expected_key in (
            ('e1c_checkpoint_path', 'e1c_checkpoint_sha256', 'source_checkpoint_sha256'),
            ('d0c2_checkpoint_path', 'd0c2_checkpoint_sha256', 'diffusion_checkpoint_sha256')):
        checkpoint_path = Path(checkpoints.get(path_key, ''))
        require(checkpoints.get(hash_key) == EXPECTED[expected_key], f'Run checkpoint provenance mismatch for {hash_key}')
        if verify_checkpoint_bytes:
            require(checkpoint_path.is_file() and sha256_file(checkpoint_path) == EXPECTED[expected_key],
                    f'Checkpoint bytes unavailable or mismatched for {path_key}: {checkpoint_path}')
    require(checkpoints.get('diffusion_state_key') == 'ema_state' and checkpoints.get('raw_state_key_loaded') is False,
            'Run did not record EMA-only diffusion weight selection')
    return run


def verify_locked_input_bytes(spatial_manifest):
    paths = {
        'b4b_contract': ROOT / 'configs/d0b4b_inference_contract.json',
        'spatial_contract': ROOT / 'configs/d0c3_spatial_inference_contract.json',
        'd0a_contract': ROOT / 'outputs/frozen/d0_latent_distribution_audit/d0_za_normalization_contract.json',
        'validation_split': ROOT / 'splits/val_cases.txt',
    }
    for name, path in paths.items():
        key = {'b4b_contract': 'b4b_contract_sha256', 'spatial_contract': 'spatial_contract_sha256',
               'd0a_contract': 'd0a_contract_sha256', 'validation_split': 'validation_split_sha256'}[name]
        require(path.is_file() and sha256_file(path) == EXPECTED[key], f'Frozen input bytes mismatch: {path}')
    spatial_dir_manifest = ROOT / 'outputs/frozen/d0c3_spatial_inference_design/MANIFEST.sha256'
    require(spatial_dir_manifest.is_file(), 'Missing frozen spatial design MANIFEST.sha256')
    spatial_entries = {}
    for line in spatial_dir_manifest.read_text(encoding='utf-8').splitlines():
        digest, rel = line.split('  ', 1)
        spatial_entries[rel] = digest
    require(spatial_entries.get('inference_window_manifest.csv') == EXPECTED['spatial_manifest_sha256']
            and sha256_file(spatial_manifest) == EXPECTED['spatial_manifest_sha256'],
            'Frozen spatial manifest bytes/inventory mismatch')
    return {name: {'path': str(path), 'sha256': sha256_file(path)} for name, path in paths.items()} | {
        'spatial_manifest': {'path': str(Path(spatial_manifest)), 'sha256': sha256_file(spatial_manifest)},
        'spatial_design_manifest': {'path': str(spatial_dir_manifest), 'sha256': sha256_file(spatial_dir_manifest)}}


def _parse_checksums(case_dir):
    path = case_dir / 'CHECKSUMS.sha256'
    require(path.is_file() and not path.is_symlink(), f'Missing/unsafe case checksum file: {path}')
    checks = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.split('  ', 1)
        require(len(parts) == 2, f'Malformed checksum line in {path}')
        digest, name = parts
        require(name not in checks, f'Duplicate checksum entry {name}')
        checks[name] = digest
    require(set(checks) == set(CASE_FILES), f'Unexpected case checksum inventory: {case_dir}')
    for name, digest in checks.items():
        artifact = case_dir / name
        require(artifact.is_file() and not artifact.is_symlink(), f'Missing/unsafe case artifact: {artifact}')
        require(sha256_file(artifact) == digest, f'Case checksum mismatch: {artifact}')
    return checks, sha256_file(path)


def verify_case(case_dir, case_id, expected_case_manifest_sha=EXPECTED['spatial_manifest_sha256']):
    case_dir = Path(case_dir)
    require(case_dir.is_dir() and not case_dir.is_symlink(), f'Missing/unsafe case directory: {case_dir}')
    require({p.name for p in case_dir.iterdir()} == set(CASE_FILES) | {'CHECKSUMS.sha256'},
            f'Unexpected or incomplete per-case artifact schema: {case_id}')
    checks, checks_sha = _parse_checksums(case_dir)
    provenance = read_json(case_dir / 'case_provenance.json')
    require(provenance.get('case_id') == case_id, f'Case provenance ID mismatch for {case_id}')
    require(provenance.get('split') == 'validation' and provenance.get('split_sha256') == EXPECTED['validation_split_sha256'],
            f'Case split identity mismatch for {case_id}')
    require(provenance.get('window_count') == 36 and provenance.get('timesteps') == [495]
            and provenance.get('alpha_bar', {}).get('495') == 0.5000776052474976
            and provenance.get('replicates') == 4 and provenance.get('ddim_evaluations') == 50
            and provenance.get('eta') == 0.0 and provenance.get('global_seed') == 20260923,
            f'Case window/timestep mismatch for {case_id}')
    for key, value in (('spatial_manifest_sha256', expected_case_manifest_sha),
                       ('spatial_contract_sha256', EXPECTED['spatial_contract_sha256']),
                       ('source_checkpoint_sha256', EXPECTED['source_checkpoint_sha256']),
                       ('diffusion_checkpoint_sha256', EXPECTED['diffusion_checkpoint_sha256']),
                       ('b4b_contract_sha256', EXPECTED['b4b_contract_sha256']),
                       ('d0a_contract_sha256', EXPECTED['d0a_contract_sha256'])):
        require(provenance.get(key) == value, f'Case provenance mismatch {key} for {case_id}')
    require(provenance.get('source_model_unchanged') is True
            and provenance.get('source_model_fingerprint_before') == provenance.get('source_model_fingerprint_after'),
            f'Source model fingerprint changed for {case_id}')
    case_runtime = provenance.get('runtime') or {}
    require(case_runtime.get('runtime_git_branch') == EXPECTED['runtime_git_branch']
            and case_runtime.get('runtime_git_commit') == EXPECTED['runtime_git_commit']
            and case_runtime.get('git_status_short') == []
            and case_runtime.get('device_name') == 'NVIDIA GeForce RTX 3090'
            and case_runtime.get('pytorch_version') == '2.14.0+cu130'
            and case_runtime.get('cuda_version') == '13.0',
            f'Case runtime provenance mismatch for {case_id}')
    require(provenance.get('ground_truth_used') is False
            and provenance.get('pathology_corruption_invoked') is False, f'Leakage/corruption flag in {case_id}')
    require(provenance.get('support_fraction_global') is not None, f'Missing support fraction for {case_id}')
    tensor_path = case_dir / 'case_deltaz.pt'
    try:
        payload = torch.load(tensor_path, map_location='cpu', weights_only=True)
    except TypeError:  # compatibility with older torch builds
        payload = torch.load(tensor_path, map_location='cpu')
    for key in ('DeltaZ_global', 'valid_support_global', 'contribution_count_global',
                'global_latent_shape', 'original_shape', 'padded_shape'):
        require(key in payload, f'Missing tensor payload key {key} for {case_id}')
    shape = (1, 23, 28, 23)
    delta, valid, count = payload['DeltaZ_global'], payload['valid_support_global'], payload['contribution_count_global']
    require(tuple(delta.shape) == shape and tuple(valid.shape) == shape and tuple(count.shape) == shape,
            f'Global tensor shape mismatch for {case_id}')
    require(tuple(int(x) for x in payload['global_latent_shape'].reshape(-1).tolist()) == (23, 28, 23),
            f'Global latent metadata mismatch for {case_id}')
    require(tuple(int(x) for x in payload['original_shape'].reshape(-1).tolist()) == (182, 218, 182)
            and tuple(int(x) for x in payload['padded_shape'].reshape(-1).tolist()) == (184, 224, 184),
            f'Original/padded tensor metadata mismatch for {case_id}')
    require(not torch.any(count < 0), f'Negative contribution count for {case_id}')
    require(torch.equal(valid.bool(), count > 0), f'Support/count mismatch for {case_id}')
    require(torch.isfinite(delta[valid.bool()]).all().item(), f'Non-finite supported DeltaZ for {case_id}')
    return dict(case_id=case_id, window_count=36, timestep=495,
        support_fraction_global=float(provenance['support_fraction_global']),
        case_deltaz_sha256=checks['case_deltaz.pt'],
        case_provenance_sha256=checks['case_provenance.json'],
        replicate_provenance_sha256=checks['replicate_provenance.csv'],
        checksums_sha256=checks_sha, tensor_shapes={k: list(payload[k].shape) for k in ('DeltaZ_global', 'valid_support_global', 'contribution_count_global')})


def _read_case_summary(path, expected_ids):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        require({'case_id', 'window_count'} <= set(reader.fieldnames or []), 'case_summary.csv schema mismatch')
        rows = list(reader)
    ids = [row['case_id'] for row in rows]
    require(len(ids) == len(expected_ids) and set(ids) == set(expected_ids), 'case_summary.csv case set mismatch')
    require(all(int(row['window_count']) == 36 for row in rows), 'case_summary.csv window count mismatch')
    return rows


def verify_source_run(source, expected_ids, spatial_manifest_sha=EXPECTED['spatial_manifest_sha256'], *,
                      verify_checkpoint_bytes=True):
    source = Path(source)
    require(source.is_dir() and not source.is_symlink(), f'Scientific source run is unavailable: {source}')
    locked_input_bytes = verify_locked_input_bytes(SPATIAL_MANIFEST)
    run = _verify_run_provenance(source, verify_checkpoint_bytes=verify_checkpoint_bytes)
    top = ('run_provenance.json', 'case_summary.csv')
    top_hashes = {}
    for name in top:
        path = source / name
        require(path.is_file() and not path.is_symlink(), f'Missing source run metadata: {path}')
        top_hashes[name] = sha256_file(path)
    _read_case_summary(source / 'case_summary.csv', expected_ids)
    cases_root = source / 'cases'
    require(cases_root.is_dir() and not cases_root.is_symlink(), 'Missing scientific cases directory')
    found = {item.name for item in cases_root.iterdir() if item.is_dir() and not item.is_symlink()}
    require(found == set(expected_ids), f'Generated case set mismatch: missing={sorted(set(expected_ids)-found)}, unexpected={sorted(found-set(expected_ids))}')
    records = [verify_case(cases_root / case_id, case_id, spatial_manifest_sha) for case_id in sorted(expected_ids)]
    return dict(run=run, top_level_hashes=top_hashes, locked_input_bytes=locked_input_bytes, case_records=records,
                checksum_manifests_verified=len(records), checksum_failures=0)


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _manifest_entries(directory):
    return {p.relative_to(directory).as_posix(): sha256_file(p)
            for p in sorted(Path(directory).rglob('*')) if p.is_file() and p.name != 'MANIFEST.sha256'}


def verify_freeze_manifest(directory):
    directory = Path(directory)
    manifest = directory / 'MANIFEST.sha256'
    require(manifest.is_file(), 'Missing freeze MANIFEST.sha256')
    entries = {}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest, rel = line.split('  ', 1)
        require(rel not in entries, f'Duplicate freeze manifest entry {rel}')
        path = directory / rel
        require(path.is_file() and not path.is_symlink(), f'Missing/unsafe freeze file {rel}')
        require(sha256_file(path) == digest, f'Freeze MANIFEST hash mismatch for {rel}')
        entries[rel] = digest
    require(entries == _manifest_entries(directory), 'Freeze file inventory differs from MANIFEST')
    require(set(entries) == set(FREEZE_FILES), 'Frozen output contains an unexpected/missing metadata file')
    require(not any(p.suffix == '.pt' for p in directory.rglob('*') if p.is_file()), 'Freeze must not contain copied tensors')
    return entries


def _make_metadata(source, output, verified, spatial_manifest_sha):
    records = verified['case_records']
    case_manifest_rows = [{key: row[key] for key in CASE_MANIFEST_FIELDS} for row in records]
    case_count = len(records)
    summary = dict(stage='D0-C3a', substage='PRIMARY_T495_SCIENTIFIC_GENERATION', status='CLOSED_FROZEN_PASS',
        scientific_generation=True, dataset_role='DEVELOPMENT_VALIDATION', independent_test=False,
        case_count=case_count, windows_per_case=36, total_windows=case_count * 36, timestep=495, replicates=4,
        ddim_evaluations=50, eta=0.0, exact_frozen_case_set_match=True,
        case_checksum_manifests_verified=case_count, case_checksum_failures=0, tensor_integrity_failures=0,
        provenance_failures=0, source_model_unchanged_all_cases=True,
        selected_diffusion_weights='D0-C2 EMA state_dict key ema_state',
        statement=('This freeze establishes completeness, integrity, identity, and reproducibility provenance of the primary DeltaZ generation. '
                   'It does not establish pathology association, tumor localization performance, diagnostic performance, or complementary pathology signal.'))
    integrity = dict(source_run=str(Path(source).resolve()), source_top_level_hashes=verified['top_level_hashes'],
        case_count=len(records), checksum_manifests_verified=verified['checksum_manifests_verified'],
        checksum_failures=verified['checksum_failures'], tensor_integrity_failures=0,
        provenance_failures=0, case_tensor_hashes={row['case_id']: row['case_deltaz_sha256'] for row in records})
    provenance = dict(source_run=str(Path(source).resolve()), source_runtime=verified['run']['runtime'],
        runtime_git_commit=EXPECTED['runtime_git_commit'], runtime_git_branch=EXPECTED['runtime_git_branch'],
        runtime_git_status='clean', identities=EXPECTED.copy(), locked_input_bytes=verified['locked_input_bytes'],
        spatial_manifest_sha256=spatial_manifest_sha,
        no_artifact_copies=True, frozen_metadata_only=True)
    (output / FREEZE_FILES[0]).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    _write_csv(output / FREEZE_FILES[1], case_manifest_rows, CASE_MANIFEST_FIELDS)
    (output / FREEZE_FILES[2]).write_text(json.dumps(integrity, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (output / FREEZE_FILES[3]).write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    lines = ''.join(f'{digest}  {rel}\n' for rel, digest in sorted(_manifest_entries(output).items()))
    (output / 'MANIFEST.sha256').write_text(lines, encoding='utf-8')


def freeze(source=SOURCE_DEFAULT, output=OUTPUT_DEFAULT, spatial_manifest=SPATIAL_MANIFEST, *, verify_only=False,
           expected_case_ids=None, verify_checkpoint_bytes=True):
    source, output = Path(source), Path(output)
    if not verify_only:
        require(not output.exists(), f'Refusing to overwrite frozen output: {output}')
    expected_ids = sorted(expected_case_ids) if expected_case_ids is not None else expected_cases_from_manifest(spatial_manifest)
    verified = verify_source_run(source, expected_ids, verify_checkpoint_bytes=verify_checkpoint_bytes)
    if verify_only:
        verify_freeze_manifest(output)
        frozen_summary = read_json(output / FREEZE_FILES[0])
        required_summary = dict(stage='D0-C3a', substage='PRIMARY_T495_SCIENTIFIC_GENERATION',
            status='CLOSED_FROZEN_PASS', scientific_generation=True,
            dataset_role='DEVELOPMENT_VALIDATION', independent_test=False,
            case_count=len(expected_ids), windows_per_case=36, total_windows=len(expected_ids) * 36,
            timestep=495, replicates=4, ddim_evaluations=50, eta=0.0,
            exact_frozen_case_set_match=True, case_checksum_manifests_verified=len(expected_ids),
            case_checksum_failures=0, tensor_integrity_failures=0, provenance_failures=0,
            source_model_unchanged_all_cases=True,
            selected_diffusion_weights='D0-C2 EMA state_dict key ema_state')
        require(all(frozen_summary.get(key) == value for key, value in required_summary.items()),
                'Frozen summary consistency failure')
        integrity = read_json(output / FREEZE_FILES[2])
        require(integrity.get('source_top_level_hashes') == verified['top_level_hashes'], 'Source metadata changed since freeze')
        require(integrity.get('case_count') == len(expected_ids)
                and integrity.get('checksum_manifests_verified') == len(expected_ids)
                and integrity.get('checksum_failures') == 0
                and integrity.get('tensor_integrity_failures') == 0
                and integrity.get('provenance_failures') == 0
                and integrity.get('case_tensor_hashes') == {row['case_id']: row['case_deltaz_sha256'] for row in verified['case_records']},
                'Frozen integrity metadata differs from source verification')
        frozen_provenance = read_json(output / FREEZE_FILES[3])
        require(frozen_provenance.get('identities') == EXPECTED
                and frozen_provenance.get('runtime_git_commit') == EXPECTED['runtime_git_commit']
                and frozen_provenance.get('runtime_git_branch') == EXPECTED['runtime_git_branch']
                and frozen_provenance.get('runtime_git_status') == 'clean'
                and frozen_provenance.get('locked_input_bytes') == verified['locked_input_bytes'],
                'Frozen provenance consistency failure')
        with (output / FREEZE_FILES[1]).open(newline='', encoding='utf-8') as stream:
            frozen_rows = list(csv.DictReader(stream))
        expected_rows = [{key: str(row[key]) for key in CASE_MANIFEST_FIELDS} for row in verified['case_records']]
        require(frozen_rows == expected_rows, 'Frozen case manifest differs from source artifacts')
        return verified
    require(not output.exists(), f'Refusing to overwrite frozen output: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.', dir=output.parent) as tmp:
        stage = Path(tmp) / output.name
        stage.mkdir()
        _make_metadata(source, stage, verified, EXPECTED['spatial_manifest_sha256'])
        os.replace(stage, output)
    return verified


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=SOURCE_DEFAULT)
    parser.add_argument('--spatial-manifest', type=Path, default=SPATIAL_MANIFEST)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    freeze(args.source_run, args.output_dir, args.spatial_manifest, verify_only=args.verify_only)


if __name__ == '__main__':
    main()
