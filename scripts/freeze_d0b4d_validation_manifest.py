"""CPU-only D0-B4d manifest/audit freeze. Real generation belongs on the data server.

Run with python -m scripts.freeze_d0b4d_validation_manifest [--verify-only].
Verification uses recorded loader dimensions and locked source files; it does not
reload MRI/GT or repeat the corruption audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess

import nibabel
import numpy as np
import torch

from datasets.transforms import normalize_nonzero
from scripts.audit_d0b1_abnormal_occupancy import abnormal_mask, block_occupancy, load_case, support_mask
from scripts.audit_d0b3_corruption_operator import corrupt_input, donor_pools
from scripts.train_baseline import read_case_ids, read_config

ROOT = Path(__file__).resolve().parents[1]
BRANCH = 'experiment/d0b4-diffusion-design'
LOCKED_COMMIT = 'cb4d830a6ce777fab7935ae77eeb75ca321db5de'
CONTRACT_SHA256 = '861ea070f7bfa2677b9345403905e70400190012e56efac722ccbb892af21f75'
VAL_SHA256 = 'ec5fe6b0bbcf190fe8b7c032f8da4ff1d495969c2d868561e7a3351c572cc5ca'
TRAIN_SHA256 = '581dc6326a22cb0b9582fa249b8e13fd948597a60c5f454cb5ade0493fa7c18e'
DECISION_SHA256 = '095e1f9671411eca0d12a2f834833d78477a14664504bde3e6e1b46766c6cc2f'
CONTRACT = ROOT / 'configs/d0b4c_training_validation_contract.json'
DECISION = ROOT / 'outputs/frozen/d0b3_corruption_operator_audit/decision/corruption_operator_decision.json'
OUTPUT = ROOT / 'outputs/frozen/d0b4d_validation_manifest'
COORDS = tuple(f'crop_{bound}_{axis}' for bound in ('start', 'stop') for axis in 'dhw')
COLUMNS = ('case_id', 'validation_patch_index', *COORDS, 'coordinate_generation_version',
           'seed', 'split_sha256', 'timestep', 'timestep_seed', 'epsilon_seed',
           'corruption_operator', 'corruption_global_seed', 'corruption_patch_index')
AUDIT_COLUMNS = ('case_id', 'validation_patch_index', 'raw_patch_shape',
                 'valid_latent_support_cells', 'total_latent_cells', 'valid_support_fraction',
                 'support_nonzero', 'corruption_shape_valid', 'corruption_finite',
                 'donor_fallback_modality_count')
FILES = ('validation_manifest.csv', 'validation_manifest_audit.csv',
         'validation_manifest_summary.json', 'provenance.json')
FLAGS = dict(scientific_training_started=False, validation_results_observed=False,
             model_selection_performed=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def payload_bytes(payload):
    return json.dumps(payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def hash_integer(payload):
    return int.from_bytes(hashlib.sha256(payload_bytes(payload)).digest(), 'big', signed=False)


def generator_seed(payload):
    return int.from_bytes(hashlib.sha256(payload_bytes(payload)).digest()[:8], 'big', signed=False) & ((1 << 63) - 1)


def checked_file(path, expected):
    data = Path(path).read_bytes()
    require(sha256(data) == expected, f'SHA256 mismatch: {path}')
    return data


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def load_sources(contract_path=CONTRACT, val_split=ROOT / 'splits/val_cases.txt',
                 train_split=ROOT / 'splits/train_cases.txt', decision_path=DECISION):
    contract = json.loads(checked_file(contract_path, CONTRACT_SHA256))
    locked = subprocess.check_output(['git', 'show', f'{LOCKED_COMMIT}:configs/d0b4c_training_validation_contract.json'], cwd=ROOT)
    require(sha256(locked) == CONTRACT_SHA256, 'D0-B4c locked commit/contract mismatch')
    v = contract['validation']
    for key, value in dict(case_count=270, patches_per_case=4, total_patches=1080, validation_global_seed=20260923).items():
        require(v[key] == value, f'D0-B4c validation {key} mismatch')
    checked_file(val_split, VAL_SHA256)
    checked_file(train_split, TRAIN_SHA256)
    ids, train_ids = read_case_ids(val_split), read_case_ids(train_split)
    require(len(ids) == 270 and len(set(ids)) == 270, 'Expected 270 unique validation IDs')
    require(len(train_ids) == 1080, 'Expected 1080 training IDs')
    require(not set(ids) & set(train_ids), 'Training/validation case overlap')
    require(all(Path(case).name == case and case not in ('.', '..') for case in ids), 'Invalid case ID')
    decision = json.loads(checked_file(decision_path, DECISION_SHA256))
    require(decision['decision']['selected_operator'] == v['corruption']['operator'], 'D0-B3 operator mismatch')
    require(contract['source_provenance']['validation_split_sha256'] == VAL_SHA256, 'Contract split mismatch')
    require(contract['source_provenance']['d0b3_corruption_decision_sha256'] == DECISION_SHA256, 'Contract decision mismatch')
    return contract, sorted(ids), dict(
        validation_split_path=str(Path(val_split).resolve()), validation_split_sha256=VAL_SHA256,
        train_split_path=str(Path(train_split).resolve()), train_split_sha256=TRAIN_SHA256,
        train_validation_disjoint=True, d0b4c_contract_path=str(Path(contract_path).resolve()),
        d0b4c_contract_sha256=sha256(Path(contract_path).read_bytes()), d0b4c_locked_commit=LOCKED_COMMIT,
        d0b3_decision_path=str(Path(decision_path).resolve()), d0b3_decision_sha256=DECISION_SHA256)


def case_rows(case_id, dimensions, contract):
    v = contract['validation']
    crop = v['manifest']
    require(len(dimensions) == 3 and all(type(n) is int and n >= 96 for n in dimensions),
            f'Volume axes must be >=96: {case_id}: {dimensions}')
    seed = v['validation_global_seed']
    rows = []
    for index in v['validation_patch_indices']:
        row = dict(case_id=case_id, validation_patch_index=index)
        for axis, size, patch_size in zip(crop['axes'], dimensions, crop['patch_shape']):
            start = hash_integer([seed, case_id, index, axis, crop['purpose']]) % (size - patch_size + 1)
            row[f'crop_start_{axis}'] = start
            row[f'crop_stop_{axis}'] = start + patch_size
        row.update(coordinate_generation_version=crop['coordinate_generation_version'], seed=seed,
                   split_sha256=crop['split_sha256'])
        identity = [row[key] for key in v['diffusion_randomness']['sample_identity_fields']]
        t_seed = generator_seed([seed, identity, v['diffusion_randomness']['timestep_purpose']])
        row.update(timestep=int(torch.randint(0, contract['forward_diffusion']['T'], (1,),
                       generator=torch.Generator(device='cpu').manual_seed(t_seed)).item()),
                   timestep_seed=t_seed,
                   epsilon_seed=generator_seed([seed, identity, v['diffusion_randomness']['epsilon_purpose']]),
                   corruption_operator=v['corruption']['operator'], corruption_global_seed=seed,
                   corruption_patch_index=index)
        rows.append(row)
    return rows


def manifest_rows(dimensions, contract):
    require(len(dimensions) == contract['validation']['case_count'], 'Expected 270 case dimensions')
    return [row for case in sorted(dimensions) for row in case_rows(case, dimensions[case], contract)]


def canonical_csv(rows, columns):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    for row in sorted(rows, key=lambda r: (r['case_id'], int(r['validation_patch_index']))):
        writer.writerow({k: str(v).lower() if isinstance(v, bool) else v for k, v in row.items()})
    return stream.getvalue().encode('utf-8')


def audit_patch(image, raw_mask, normalized, row, contract):
    require(image.device.type == normalized.device.type == raw_mask.device.type == 'cpu', 'Audit requires CPU tensors')
    slices = tuple(slice(row[f'crop_start_{axis}'], row[f'crop_stop_{axis}']) for axis in 'dhw')
    require(all(0 <= s.start < s.stop <= size for s, size in zip(slices, image.shape[1:])), 'Crop out of bounds')
    raw_patch = image[(slice(None), *slices)]
    require(tuple(raw_patch.shape) == (4, 96, 96, 96), 'Expected raw patch [4,96,96,96]')
    support = support_mask(raw_patch)
    occupancy = block_occupancy(support)
    require(tuple(occupancy.shape) == (12, 12, 12), 'Invalid latent support shape')
    count = int(occupancy.ge(contract['support']['occupancy_threshold']).sum())
    require(count > 0, f'Zero valid latent support: {row["case_id"]}/{row["validation_patch_index"]}; no resampling permitted')
    abnormal = abnormal_mask(raw_mask[slices])
    pools, fallbacks = donor_pools(raw_patch, abnormal, support)
    corrupted = corrupt_input(normalized[(slice(None), *slices)], abnormal, pools,
                              row['corruption_operator'], global_seed=row['corruption_global_seed'],
                              case_id=row['case_id'], patch_index=row['corruption_patch_index'])
    require(tuple(corrupted.shape) == (4, 96, 96, 96), 'Invalid corrupted patch shape')
    require(bool(torch.isfinite(corrupted).all()), 'Non-finite corrupted patch')
    return dict(case_id=row['case_id'], validation_patch_index=row['validation_patch_index'],
                raw_patch_shape='[4,96,96,96]', valid_latent_support_cells=count,
                total_latent_cells=1728, valid_support_fraction=count / 1728,
                support_nonzero=True, corruption_shape_valid=True, corruption_finite=True,
                donor_fallback_modality_count=sum(bool(x) for x in fallbacks))


def audit_case(dataset_root, case, contract):
    image, mask = load_case(dataset_root, case)
    require(image.device.type == mask.device.type == 'cpu', 'Loader must return CPU tensors')
    require(image.ndim == 4 and image.shape[0] == 4, 'Expected four MRI modalities')
    require(tuple(mask.shape) == tuple(image.shape[1:]), 'MRI/segmentation shape mismatch')
    dimensions = list(image.shape[1:])
    rows = case_rows(case, dimensions, contract)  # No GT or support is consulted.
    image = image.float()
    normalized = normalize_nonzero(image)  # Existing full-volume normalization.
    audits = [audit_patch(image, mask, normalized, row, contract) for row in rows]
    return dimensions, rows, audits


def parse_audits(data, rows):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))
    require(reader.fieldnames == list(AUDIT_COLUMNS), 'Audit columns mismatch')
    audits = list(reader)
    require(len(audits) == len(rows), 'Audit row count mismatch')
    parsed = []
    for audit, row in zip(audits, rows):
        require(audit['case_id'] == row['case_id'] and audit['validation_patch_index'] == str(row['validation_patch_index']), 'Audit row alignment mismatch')
        require(audit['raw_patch_shape'] == '[4,96,96,96]', 'Audit patch shape mismatch')
        require(all(audit[key] == 'true' for key in ('support_nonzero', 'corruption_shape_valid', 'corruption_finite')), 'Audit support/corruption failure')
        count, total = int(audit['valid_latent_support_cells']), int(audit['total_latent_cells'])
        require(total == 1728 and 0 < count <= total, 'Invalid audit support count')
        require(float(audit['valid_support_fraction']) == count / total, 'Audit support fraction mismatch')
        fallback = int(audit['donor_fallback_modality_count'])
        require(0 <= fallback <= 4, 'Invalid donor fallback count')
        parsed.append(dict(case_id=audit['case_id'], validation_patch_index=int(audit['validation_patch_index']),
            raw_patch_shape=audit['raw_patch_shape'], valid_latent_support_cells=count, total_latent_cells=total,
            valid_support_fraction=count / total, support_nonzero=True, corruption_shape_valid=True,
            corruption_finite=True, donor_fallback_modality_count=fallback))
    require(canonical_csv(parsed, AUDIT_COLUMNS) == data, 'Noncanonical audit CSV')
    return parsed


def summary_record(rows, audits, contract, manifest_data, audit_data):
    v = contract['validation']
    values = np.asarray([a['valid_support_fraction'] for a in audits], dtype=np.float64)
    stats = dict(mean=float(values.mean()), std=float(values.std(ddof=0)), min=float(values.min()), max=float(values.max()))
    stats.update({f'p{p:02d}': float(np.percentile(values, p, method='linear')) for p in (1, 5, 25, 50, 75, 95, 99)})
    return dict(stage='D0-B4d', status='CLOSED_FROZEN_PRE_TRAINING', **FLAGS,
        validation_role='DEVELOPMENT_VALIDATION', case_count=v['case_count'], patches_per_case=v['patches_per_case'],
        total_patches=len(rows), patch_shape=v['manifest']['patch_shape'], latent_shape=[12, 12, 12],
        validation_global_seed=v['validation_global_seed'], coordinate_generation_version=v['manifest']['coordinate_generation_version'],
        support_threshold=contract['support']['occupancy_threshold'], support_mapping='exact 8x8x8 occupancy',
        zero_support_patch_count=0, required_zero_support_patch_count=0,
        corruption_operator=v['corruption']['operator'], validation_manifest_sha256=sha256(manifest_data),
        validation_manifest_audit_sha256=sha256(audit_data), support_fraction_statistics=stats,
        statistics_semantics='Descriptive only; population std; linear percentiles; no sampling or selection changes.',
        donor_fallback_modality_count=sum(a['donor_fallback_modality_count'] for a in audits),
        donor_fallback_patch_count=sum(a['donor_fallback_modality_count'] > 0 for a in audits),
        corruption_executability_audited=True, gt_used_for_coordinate_selection=False,
        gt_use='Only to audit the already-frozen corruption operation.',
        independent_final_test=False)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def manifest_bytes(files):
    return ''.join(f'{sha256(files[name])}  {name}\n' for name in sorted(files)).encode('utf-8')


def validate_bundle(files, contract, ids, sources):
    provenance = json.loads(files['provenance.json'])
    for key, expected in sources.items():
        if not key.endswith('_path'):
            require(provenance.get(key) == expected, f'Provenance mismatch: {key}')
    require(provenance.get('runtime_git_branch') == BRANCH, 'Provenance branch mismatch')
    require(provenance.get('gt_used_for_coordinate_selection') is False, 'GT coordinate provenance mismatch')
    require(provenance.get('gt_use') == 'Only to audit the already-frozen corruption operation.', 'GT use mismatch')
    for key, expected in FLAGS.items():
        require(provenance.get(key) is expected, f'Provenance flag mismatch: {key}')
    dimensions = provenance['case_dimensions']
    require(set(dimensions) == set(ids), 'Case dimensions do not match locked validation IDs')
    rows = manifest_rows(dimensions, contract)
    expected_csv = canonical_csv(rows, COLUMNS)
    require(files['validation_manifest.csv'] == expected_csv, 'Manifest rows/columns/order/coordinates/seeds/identities mismatch')
    audits = parse_audits(files['validation_manifest_audit.csv'], rows)
    summary = summary_record(rows, audits, contract, expected_csv, files['validation_manifest_audit.csv'])
    require(json.loads(files['validation_manifest_summary.json']) == summary, 'Summary consistency mismatch')
    for key in ('validation_global_seed', 'coordinate_generation_version', 'validation_manifest_sha256',
                'validation_manifest_audit_sha256', 'donor_fallback_modality_count', 'donor_fallback_patch_count'):
        require(provenance.get(key) == summary[key], f'Provenance mismatch: {key}')
    return summary


def verify(output_dir, *, contract_path=CONTRACT, val_split=ROOT / 'splits/val_cases.txt',
           train_split=ROOT / 'splits/train_cases.txt', decision_path=DECISION):
    root = Path(output_dir)
    require(not root.is_symlink(), 'Frozen directory must not be a symlink')
    paths = list(root.iterdir())
    require({p.name for p in paths} == {*FILES, 'MANIFEST.sha256'}, 'Frozen file inventory mismatch')
    require(all(p.is_file() and not p.is_symlink() for p in paths), 'Frozen entries must be regular files')
    files = {name: (root / name).read_bytes() for name in FILES}
    require((root / 'MANIFEST.sha256').read_bytes() == manifest_bytes(files), 'MANIFEST hash mismatch')
    contract, ids, sources = load_sources(contract_path, val_split, train_split, decision_path)
    return validate_bundle(files, contract, ids, sources)


def run(*, output_dir=OUTPUT, config_path=ROOT / 'configs/data.yaml', dataset_root=None,
        contract_path=CONTRACT, val_split=ROOT / 'splits/val_cases.txt',
        train_split=ROOT / 'splits/train_cases.txt', decision_path=DECISION, verify_only=False):
    destination = Path(output_dir).expanduser()
    if verify_only:
        return verify(destination, contract_path=contract_path, val_split=val_split,
                      train_split=train_split, decision_path=decision_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Refusing to overwrite frozen output: {destination}')
    require(git('branch', '--show-current') == BRANCH, f'Expected branch {BRANCH}')
    subprocess.run(['git', 'merge-base', '--is-ancestor', LOCKED_COMMIT, 'HEAD'], cwd=ROOT, check=True)
    contract, ids, sources = load_sources(contract_path, val_split, train_split, decision_path)
    root = Path(dataset_root if dataset_root is not None else read_config(config_path)['root']).expanduser().resolve()
    require(all((root / case).is_dir() for case in ids), 'Missing validation case directory in dataset root')
    dimensions, rows, audits = {}, [], []
    for case in ids:
        dims, case_manifest, case_audit = audit_case(root, case, contract)
        dimensions[case] = dims
        rows.extend(case_manifest)
        audits.extend(case_audit)
    files = {'validation_manifest.csv': canonical_csv(rows, COLUMNS),
             'validation_manifest_audit.csv': canonical_csv(audits, AUDIT_COLUMNS)}
    summary = summary_record(rows, audits, contract, files['validation_manifest.csv'], files['validation_manifest_audit.csv'])
    provenance = dict(**sources, **FLAGS, runtime_git_branch=git('branch', '--show-current'),
        runtime_git_commit=git('rev-parse', 'HEAD'), git_status_short=git('status', '--short').splitlines(),
        dataset_root=str(root), case_dimensions=dimensions,
        software_versions=dict(python=platform.python_version(), pytorch=str(torch.__version__),
                               numpy=np.__version__, nibabel=nibabel.__version__),
        gt_used_for_coordinate_selection=False, gt_use='Only to audit the already-frozen corruption operation.',
        verification_scope='Manifest formulas checked against recorded loader dimensions; support/corruption audit records checked without reloading MRI/GT.')
    for key in ('validation_global_seed', 'coordinate_generation_version', 'validation_manifest_sha256',
                'validation_manifest_audit_sha256', 'donor_fallback_modality_count', 'donor_fallback_patch_count'):
        provenance[key] = summary[key]
    files['validation_manifest_summary.json'] = json_bytes(summary)
    files['provenance.json'] = json_bytes(provenance)
    validate_bundle(files, contract, ids, sources)  # All audits pass before any output is created.
    files['MANIFEST.sha256'] = manifest_bytes(files)
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        with (destination / name).open('xb') as stream:
            stream.write(data)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--config', dest='config_path', type=Path, default=ROOT / 'configs/data.yaml')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--contract', dest='contract_path', type=Path, default=CONTRACT)
    parser.add_argument('--validation-split', dest='val_split', type=Path, default=ROOT / 'splits/val_cases.txt')
    parser.add_argument('--train-split', type=Path, default=ROOT / 'splits/train_cases.txt')
    parser.add_argument('--decision', dest='decision_path', type=Path, default=DECISION)
    parser.add_argument('--verify-only', action='store_true')
    print(json.dumps(run(**vars(parser.parse_args())), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
