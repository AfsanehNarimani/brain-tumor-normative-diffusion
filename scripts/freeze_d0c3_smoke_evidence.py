#!/usr/bin/env python3
"""Freeze checksummed evidence from two completed non-scientific D0-C3 runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SMOKE_ROOT = ROOT / 'outputs/smoke/d0c3_deltaz'
DEFAULT_OUTPUT = ROOT / 'outputs/frozen/d0c3_real_data_smoke'
SPATIAL_CONTRACT_PATH = ROOT / 'configs/d0c3_spatial_inference_contract.json'
RUN_NAMES = ('smoke_02', 'smoke_03')
CASE_ID = 'BraTS-GLI-00008-101'
RUN_COMMIT = '4e7ca572d6238e3ed1f50e74c4a8f0d9f78c7679'
EXPECTED = {
    'd0c2_checkpoint_sha256': 'aaa6258f26a56c852dc751b8028bfb046da5752974d4cbaf446ad2699a0c6574',
    'e1c_checkpoint_sha256': 'cf5a04a2d845cecb45a0ea87d617abad0afab9ad4264abce8342a6045563ee53',
    'd0b4b_contract_sha256': '29cf659528f6257cd9f8f218e9f4eaf8fabdc6d8a0e4905505aa8548ae68bde4',
    'd0a_normalization_contract_sha256': 'bca5c0f8f1617d73fc4d0850fe5b6f45e4b2bcbacbb4018f1d5d825de49ad792',
    'spatial_manifest_sha256': '58fac2ebef778d4d19d6da9f21451c25175c61bb4100c03387e8d3c859c05d53',
    'validation_split_sha256': 'ec5fe6b0bbcf190fe8b7c032f8da4ff1d495969c2d868561e7a3351c572cc5ca',
    'spatial_contract_sha256': '832b8b04e411802186ddc5ba0fb598fa748d53876c87c282b950863919ef986c',
}
EXPECTED_GPU = 'NVIDIA GeForce RTX 3090'
EXPECTED_PYTORCH = '2.14.0+cu130'
EXPECTED_CUDA = '13.0'
CASE_FILES = ('case_deltaz.pt', 'case_provenance.json', 'replicate_provenance.csv')
FREEZE_FILES = ('d0c3_smoke_summary.json', 'd0c3_smoke_artifact_manifest.json',
                'd0c3_reproducibility.json', 'provenance.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_case_checksums(case_dir):
    case_dir = Path(case_dir)
    checksum_path = case_dir / 'CHECKSUMS.sha256'
    require(checksum_path.is_file(), f'Missing smoke case CHECKSUMS.sha256: {checksum_path}')
    entries = {}
    for line in checksum_path.read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        require(name not in entries, f'Duplicate smoke case checksum entry: {name}')
        require(name in CASE_FILES, f'Unexpected smoke case checksum target: {name}')
        path = case_dir / name
        require(path.is_file() and not path.is_symlink(), f'Missing/unsafe smoke case artifact: {path}')
        actual = sha256_file(path)
        require(actual == digest, f'Smoke case checksum mismatch: {name}')
        entries[name] = actual
    require(set(entries) == set(CASE_FILES), 'Smoke case checksum inventory mismatch')
    actual_names = {p.name for p in case_dir.iterdir() if p.is_file()}
    require(actual_names == set(CASE_FILES) | {'CHECKSUMS.sha256'}, 'Unexpected file in smoke case artifact directory')
    return entries, sha256_file(checksum_path)


def _tensor_equal(left, right, path='root'):
    details = {}
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        require(isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor),
                f'Tensor structure differs at {path}')
        require(left.shape == right.shape and left.dtype == right.dtype,
                f'Tensor shape/dtype differs at {path}')
        equal = torch.equal(left, right)
        max_diff = 0.0
        if left.numel() and left.is_floating_point():
            max_diff = float((left.to(torch.float64) - right.to(torch.float64)).abs().max())
        details[path] = dict(shape=list(left.shape), dtype=str(left.dtype).removeprefix('torch.'),
                             exact_equal=bool(equal), max_abs_diff=max_diff)
        require(equal, f'Tensor values differ at {path}; max_abs_diff={max_diff}')
        return details
    if isinstance(left, dict) or isinstance(right, dict):
        require(isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right),
                f'Dictionary structure differs at {path}')
        for key in sorted(left):
            details.update(_tensor_equal(left[key], right[key], f'{path}.{key}'))
        return details
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        require(type(left) is type(right) and len(left) == len(right), f'Sequence structure differs at {path}')
        for index, (a, b) in enumerate(zip(left, right)):
            details.update(_tensor_equal(a, b, f'{path}[{index}]'))
        return details
    require(type(left) is type(right) and left == right, f'Content differs at {path}')
    details[path] = dict(value=left, exact_equal=True)
    return details


def load_case_tensor(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(payload, dict), 'Smoke case tensor artifact must be a dictionary')
    required = {'local_DeltaZ', 'local_support', 'processed_window_indices', 'complete_case_aggregation'}
    require(required <= set(payload), f'Smoke tensor artifact missing fields: {sorted(required - set(payload))}')
    require(payload['local_DeltaZ'].shape == (2, 1, 12, 12, 12)
            and payload['local_DeltaZ'].dtype == torch.float32,
            'Observed local_DeltaZ must be float32 [2,1,12,12,12]')
    require(payload['local_support'].shape == (2, 1, 12, 12, 12),
            'Observed local_support shape mismatch')
    require(payload['local_support'].dtype == torch.bool,
            'Observed local_support dtype mismatch')
    require(payload['processed_window_indices'].shape == (2,),
            'Observed processed_window_indices shape mismatch')
    require(payload['processed_window_indices'].dtype == torch.int64,
            'Observed processed_window_indices dtype mismatch')
    require(torch.isfinite(payload['local_DeltaZ']).all(), 'Smoke local_DeltaZ contains non-finite values')
    require(isinstance(payload['complete_case_aggregation'], bool),
            'complete_case_aggregation must be the actual stored boolean value')
    return payload


def _check_run_identity(run_dir):
    run_dir = Path(run_dir)
    index_path = run_dir / 'smoke_case_index.csv'
    provenance_path = run_dir / 'smoke_run_provenance.json'
    case_dir = run_dir / 'smoke_cases' / CASE_ID
    require(index_path.is_file() and provenance_path.is_file(), f'Incomplete smoke run: {run_dir}')
    run_prov = json_read(provenance_path)
    require(run_prov.get('status') == 'PASS', f'{run_dir.name} did not PASS')
    require(run_prov.get('scientific_analysis') is False and run_prov.get('scientific_result') is False,
            f'{run_dir.name} is not marked non-scientific')
    require(run_prov.get('ground_truth_opened') is False
            and run_prov.get('pathology_corruption_invoked') is False
            and run_prov.get('source_model_unchanged') is True,
            f'{run_dir.name} failed GT/corruption/source immutability guards')
    runtime = run_prov.get('runtime', {})
    require(runtime.get('runtime_git_commit') == RUN_COMMIT
            and runtime.get('git_status_short') == [], f'{run_dir.name} runtime git state is not clean/locked')
    require(runtime.get('device_name') == EXPECTED_GPU
            and runtime.get('pytorch_version') == EXPECTED_PYTORCH
            and runtime.get('cuda_version') == EXPECTED_CUDA,
            f'{run_dir.name} hardware/software identity mismatch')
    identity = run_prov.get('run_identity', {})
    require(identity.get('mode') == 'smoke' and identity.get('scientific_analysis') is False
            and identity.get('scientific_result') is False,
            f'{run_dir.name} run identity is not explicitly non-scientific')
    actual_refs = dict(d0c2_checkpoint_sha256=identity.get('d0c2_checkpoint_sha256'),
        e1c_checkpoint_sha256=identity.get('e1c_checkpoint_sha256'),
        d0b4b_contract_sha256=identity.get('b4b_sha256'),
        d0a_normalization_contract_sha256=identity.get('d0a_sha256'),
        spatial_manifest_sha256=identity.get('spatial_manifest_sha256'),
        validation_split_sha256=identity.get('validation_split_sha256'),
        spatial_contract_sha256=identity.get('spatial_contract_sha256'))
    require(actual_refs == EXPECTED, f'{run_dir.name} frozen run identity hashes mismatch')
    require(sha256_file(SPATIAL_CONTRACT_PATH) == EXPECTED['spatial_contract_sha256']
            and identity.get('spatial_contract_sha256') == EXPECTED['spatial_contract_sha256'],
            f'{run_dir.name} D0-C3 spatial contract SHA mismatch')
    checkpoint_info = run_prov.get('checkpoint_info', {})
    require(checkpoint_info.get('d0c2_checkpoint_sha256') == EXPECTED['d0c2_checkpoint_sha256']
            and checkpoint_info.get('e1c_checkpoint_sha256') == EXPECTED['e1c_checkpoint_sha256']
            and checkpoint_info.get('diffusion_state_key') == 'ema_state'
            and checkpoint_info.get('raw_state_key_loaded') is False,
            f'{run_dir.name} checkpoint/EMA provenance mismatch')
    with index_path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 1 and rows[0].get('case_id') == CASE_ID
            and rows[0].get('status') in ('completed', 'already_verified')
            and int(rows[0].get('window_count', -1)) == 2,
            f'{run_dir.name} smoke case index mismatch')
    require(case_dir.is_dir(), f'{run_dir.name} case artifact directory missing')
    case_prov_path = case_dir / 'case_provenance.json'
    require(case_prov_path.is_file(), f'{run_dir.name} case provenance missing')
    case_prov = json_read(case_prov_path)
    require(case_prov.get('case_id') == CASE_ID and case_prov.get('window_count') == 2,
            f'{run_dir.name} case/window provenance mismatch')
    require(case_prov.get('run_identity') == identity, f'{run_dir.name} case/run identity mismatch')
    require(case_prov.get('ground_truth_used') is False
            and case_prov.get('pathology_corruption_invoked') is False
            and case_prov.get('source_model_unchanged') is True,
            f'{run_dir.name} case leakage/immutability flags mismatch')
    case_refs = dict(d0c2_checkpoint_sha256=case_prov.get('diffusion_checkpoint_sha256'),
        e1c_checkpoint_sha256=case_prov.get('source_checkpoint_sha256'),
        d0b4b_contract_sha256=case_prov.get('b4b_contract_sha256'),
        d0a_normalization_contract_sha256=case_prov.get('d0a_contract_sha256'),
        spatial_manifest_sha256=case_prov.get('spatial_manifest_sha256'),
        validation_split_sha256=case_prov.get('split_sha256'),
        spatial_contract_sha256=case_prov.get('spatial_contract_sha256'))
    require(case_refs == EXPECTED, f'{run_dir.name} case artifact frozen identities mismatch')
    hashes, checksums_sha = verify_case_checksums(case_dir)
    tensor_payload = load_case_tensor(case_dir / 'case_deltaz.pt')
    require(len(tensor_payload['processed_window_indices']) == 2,
            f'{run_dir.name} did not process exactly two windows')
    # Two windows cannot support a complete 36-window case map; require the saved
    # flag to truthfully reflect that rather than deriving it from rerun equality.
    require(case_prov.get('complete_case') is tensor_payload['complete_case_aggregation'],
            f'{run_dir.name} stored complete-case flag disagrees with case provenance')
    run_files = {'smoke_case_index.csv': dict(path='smoke_case_index.csv', sha256=sha256_file(index_path)),
        'smoke_run_provenance.json': dict(path='smoke_run_provenance.json', sha256=sha256_file(provenance_path)),
        f'smoke_cases/{CASE_ID}/CHECKSUMS.sha256': dict(
            path=f'smoke_cases/{CASE_ID}/CHECKSUMS.sha256', sha256=checksums_sha)}
    run_files.update({(Path('smoke_cases') / CASE_ID / name).as_posix():
                      dict(path=(Path('smoke_cases') / CASE_ID / name).as_posix(), sha256=digest)
                      for name, digest in hashes.items()})
    return dict(run_name=run_dir.name, run_dir=run_dir, run_provenance=run_prov,
        case_provenance=case_prov, tensor_payload=tensor_payload, artifacts=run_files)


def audit_smoke_runs(smoke_root=DEFAULT_SMOKE_ROOT):
    smoke_root = Path(smoke_root).expanduser().resolve()
    audited = [_check_run_identity(smoke_root / name) for name in RUN_NAMES]
    first, second = audited
    comparison = _tensor_equal(first['tensor_payload'], second['tensor_payload'])
    rep0 = first['run_dir'] / 'smoke_cases' / CASE_ID / 'replicate_provenance.csv'
    rep1 = second['run_dir'] / 'smoke_cases' / CASE_ID / 'replicate_provenance.csv'
    rep_equal = rep0.read_bytes() == rep1.read_bytes()
    require(rep_equal, 'replicate_provenance.csv differs between smoke reruns')
    require(first['tensor_payload']['complete_case_aggregation']
            == second['tensor_payload']['complete_case_aggregation'],
            'complete_case_aggregation differs between smoke reruns')
    require(first['tensor_payload']['processed_window_indices'].tolist()
            == second['tensor_payload']['processed_window_indices'].tolist(),
            'Smoke runs processed different frozen window indices')
    run_artifacts = {entry['run_name']: dict(source_run_path=str(entry['run_dir']),
        files=entry['artifacts']) for entry in audited}
    summary = dict(stage='D0-C3a', substage='REAL_DATA_GPU_SMOKE', status='CLOSED_FROZEN_PASS',
        scientific_result=False, scientific_analysis=False, smoke_runs=list(RUN_NAMES),
        case_id=CASE_ID, windows_per_run=2, full_case_windows=36,
        exact_rerun_reproducibility=True,
        local_deltaz_max_abs_diff=comparison['root.local_DeltaZ']['max_abs_diff'],
        replicate_provenance_byte_identical=rep_equal,
        ground_truth_opened=False, pathology_corruption_invoked=False, source_model_unchanged=True,
        complete_case_aggregation=first['tensor_payload']['complete_case_aggregation'],
        complete_case_aggregation_equal_across_reruns=True,
        interpretation='This smoke validates execution and deterministic reproducibility only. It does not constitute a scientific DeltaZ/pathology result.')
    reproducibility = dict(all_exact=True, tensor_comparison=comparison,
        replicate_provenance_byte_identical=True,
        complete_case_aggregation=dict(run_values={entry['run_name']: entry['tensor_payload']['complete_case_aggregation']
                                                   for entry in audited},
            equal_across_reruns=True),
        scope='Two frozen windows from one case across smoke_02 and smoke_03 only.')
    return summary, dict(smoke_runs=run_artifacts), reproducibility, audited


def _manifest_bytes(directory):
    entries = []
    for path in sorted(Path(directory).rglob('*')):
        if path.is_file() and path.name != 'MANIFEST.sha256':
            entries.append(f'{sha256_file(path)}  {path.relative_to(directory).as_posix()}')
    return ('\n'.join(entries) + '\n').encode('utf-8')


def verify_freeze(output_dir, smoke_root=DEFAULT_SMOKE_ROOT):
    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / 'MANIFEST.sha256'
    require(manifest_path.is_file(), 'Frozen smoke evidence MANIFEST.sha256 missing')
    expected_entries = {}
    for line in manifest_path.read_text(encoding='utf-8').splitlines():
        digest, rel = line.split('  ', 1)
        require(rel not in expected_entries, f'Duplicate frozen MANIFEST entry: {rel}')
        expected_entries[rel] = digest
    actual_files = {path.relative_to(output_dir).as_posix() for path in output_dir.rglob('*')
                    if path.is_file() and path.name != 'MANIFEST.sha256'}
    require(actual_files == set(FREEZE_FILES) and set(expected_entries) == set(FREEZE_FILES),
            'Frozen smoke evidence file inventory mismatch (tensor copies are forbidden)')
    for rel, digest in expected_entries.items():
        require(sha256_file(output_dir / rel) == digest, f'Frozen smoke evidence MANIFEST mismatch: {rel}')
    summary, artifact_manifest, reproducibility, audited = audit_smoke_runs(smoke_root)
    require(json_read(output_dir / 'd0c3_smoke_summary.json') == summary,
            'Frozen smoke summary differs from re-audited source runs')
    require(json_read(output_dir / 'd0c3_smoke_artifact_manifest.json') == artifact_manifest,
            'Frozen smoke artifact references differ from source runs')
    require(json_read(output_dir / 'd0c3_reproducibility.json') == reproducibility,
            'Frozen reproducibility record differs from source runs')
    provenance = json_read(output_dir / 'provenance.json')
    require(provenance.get('source_runtime_commits') == {name: RUN_COMMIT for name in RUN_NAMES}
            and provenance.get('source_smoke_runs') == list(RUN_NAMES),
            'Frozen smoke provenance identity mismatch')
    require(provenance.get('expected_frozen_identities') == EXPECTED
            and provenance.get('source_artifact_manifest') == artifact_manifest,
            'Frozen smoke provenance artifact/contract references mismatch')
    return dict(status=summary['status'], run_count=len(audited),
        freeze_manifest_sha256=sha256_file(manifest_path), artifact_manifest=artifact_manifest)


def freeze(smoke_root=DEFAULT_SMOKE_ROOT, output_dir=DEFAULT_OUTPUT, *, verify_only=False):
    smoke_root = Path(smoke_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if verify_only:
        return verify_freeze(output_dir, smoke_root)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite frozen smoke evidence: {output_dir}')
    # Hash all referenced inputs before and after audit so this read-only tool
    # proves it did not change any smoke source file.
    before = {name: _check_run_identity(smoke_root / name)['artifacts'] for name in RUN_NAMES}
    summary, artifact_manifest, reproducibility, audited = audit_smoke_runs(smoke_root)
    after = {name: _check_run_identity(smoke_root / name)['artifacts'] for name in RUN_NAMES}
    require(before == after, 'Source smoke artifacts changed during freeze audit')
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    provenance = dict(stage='D0-C3a', substage='REAL_DATA_GPU_SMOKE', source_smoke_runs=list(RUN_NAMES),
        source_runtime_commits={name: RUN_COMMIT for name in RUN_NAMES},
        source_runtime_git_status={entry['run_name']: entry['run_provenance']['runtime']['git_status_short']
                                   for entry in audited},
        expected_frozen_identities=EXPECTED, source_artifact_manifest=artifact_manifest,
        runtime_git_branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
        runtime_git_commit=git('rev-parse', 'HEAD'), git_status_short=git('status', '--short').splitlines(),
        source_artifacts_changed_by_freeze=False,
        complete_case_aggregation_observed={entry['run_name']: entry['tensor_payload']['complete_case_aggregation']
                                             for entry in audited})
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.d0c3_smoke_freeze.', dir=output_dir.parent) as temp:
        stage = Path(temp)
        for name, payload in (('d0c3_smoke_summary.json', summary),
                              ('d0c3_smoke_artifact_manifest.json', artifact_manifest),
                              ('d0c3_reproducibility.json', reproducibility),
                              ('provenance.json', provenance)):
            (stage / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        (stage / 'MANIFEST.sha256').write_bytes(_manifest_bytes(stage))
        os.replace(stage, output_dir)
    return verify_freeze(output_dir, smoke_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-root', type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(freeze(args.smoke_root, args.output_dir, verify_only=args.verify_only), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
