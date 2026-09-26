#!/usr/bin/env python3
"""Post-hoc case-level association analysis of frozen primary D0-C3 DeltaZ maps."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import nibabel as nib
import numpy as np
import scipy
from scipy.stats import rankdata, wilcoxon
import torch

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / 'configs/d0c3b_pathology_association_contract.json'
CONTRACT_SHA256 = '4404e6e32b92271b9e8cdd3bd4d7c249dc13d0fc4b521922beb68b9ff6245fab'
PRIMARY_RUN_DEFAULT = ROOT / 'outputs/scientific/d0c3_deltaz/primary_t495'
PRIMARY_FREEZE_DEFAULT = ROOT / 'outputs/frozen/d0c3_primary_t495'
DATASET_DEFAULT = Path('/home/friend/data/training_data1_v2')
OUTPUT_DEFAULT = ROOT / 'outputs/scientific/d0c3b_pathology_association'
REGIONS = {'WT': (1, 2, 3), 'TC': (1, 3), 'ET': (3,)}
EXPECTED_SHAPE = (23, 28, 23)
ORIGINAL_SHAPE = (182, 218, 182)
PADDED_SHAPE = (184, 224, 184)
LABELS = {0, 1, 2, 3, 4}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_freeze_manifest(directory):
    directory = Path(directory)
    manifest = directory / 'MANIFEST.sha256'
    require(manifest.is_file(), f'Missing primary freeze MANIFEST: {manifest}')
    entries = {}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest, rel = line.split('  ', 1)
        require(rel not in entries, f'Duplicate freeze entry {rel}')
        path = directory / rel
        require(path.is_file() and not path.is_symlink(), f'Missing/unsafe primary freeze file {rel}')
        require(sha256_file(path) == digest, f'Primary freeze hash mismatch for {rel}')
        entries[rel] = digest
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256'}
    require(actual == set(entries), 'Primary freeze inventory differs from MANIFEST')
    require(sha256_file(manifest) == read_json(CONTRACT_PATH)['sources']['primary_freeze_manifest_sha256'],
            'Primary freeze MANIFEST identity mismatch')
    return entries


def _verify_primary_source(primary_run, primary_freeze):
    entries = verify_freeze_manifest(primary_freeze)
    contract = read_json(CONTRACT_PATH)
    summary = read_json(Path(primary_freeze) / 'd0c3_primary_summary.json')
    require(summary.get('status') == 'CLOSED_FROZEN_PASS' and summary.get('case_count') == 270
            and summary.get('windows_per_case') == 36 and summary.get('total_windows') == 9720
            and summary.get('timestep') == 495 and summary.get('dataset_role') == 'DEVELOPMENT_VALIDATION'
            and summary.get('independent_test') is False,
            'Primary generation freeze summary is invalid')
    provenance = read_json(Path(primary_freeze) / 'provenance.json')
    ids = provenance.get('identities', {})
    for target, source in (("validation_split_sha256", "validation_split_sha256"),
                           ("spatial_manifest_sha256", "spatial_manifest_sha256"),
                           ("spatial_contract_sha256", "spatial_contract_sha256"),
                           ("b4b_contract_sha256", "d0b4b_inference_contract_sha256"),
                           ("d0a_contract_sha256", "d0a_normalization_contract_sha256"),
                           ("source_checkpoint_sha256", "source_checkpoint_sha256"),
                           ("diffusion_checkpoint_sha256", "diffusion_checkpoint_sha256")):
        require(ids.get(target) == contract['sources'][source], f'Primary freeze identity mismatch: {target}')
    require(provenance.get('runtime_git_commit') == '75e0d3868b73a947d53534631ef2a6f9d83ebb04'
            and provenance.get('runtime_git_branch') == 'experiment/d0c3-scientific-generation'
            and provenance.get('runtime_git_status') == 'clean', 'Primary generation runtime provenance mismatch')
    locked_files = {
        'configs/d0b4b_inference_contract.json': contract['sources']['d0b4b_inference_contract_sha256'],
        'configs/d0c3_spatial_inference_contract.json': contract['sources']['spatial_contract_sha256'],
        'outputs/frozen/d0_latent_distribution_audit/d0_za_normalization_contract.json': contract['sources']['d0a_normalization_contract_sha256'],
        'splits/val_cases.txt': contract['sources']['validation_split_sha256'],
        'outputs/frozen/d0c3_spatial_inference_design/inference_window_manifest.csv': contract['sources']['spatial_manifest_sha256'],
    }
    for relative, expected_sha in locked_files.items():
        locked_path = ROOT / relative
        require(locked_path.is_file() and sha256_file(locked_path) == expected_sha,
                f'Locked source file missing or changed: {relative}')
    case_manifest_path = Path(primary_freeze) / 'd0c3_primary_case_manifest.csv'
    require(entries.get('d0c3_primary_case_manifest.csv') == sha256_file(case_manifest_path),
            'Primary case manifest hash not covered by freeze manifest')
    with case_manifest_path.open(newline='', encoding='utf-8') as stream:
        case_rows = list(csv.DictReader(stream))
    ids_sorted = [row['case_id'] for row in case_rows]
    require(len(ids_sorted) == 270 and len(set(ids_sorted)) == 270, 'Frozen case manifest must list 270 unique cases')
    require(ids_sorted == sorted(ids_sorted), 'Frozen case manifest order is not canonical')
    with (ROOT / 'outputs/frozen/d0c3_spatial_inference_design/inference_window_manifest.csv').open(newline='', encoding='utf-8') as stream:
        spatial_ids = sorted({row['case_id'] for row in csv.DictReader(stream)})
    require(ids_sorted == spatial_ids, 'Primary frozen case set differs from the frozen spatial manifest')

    run = Path(primary_run)
    require(run.is_dir(), f'Primary scientific run is unavailable: {run}')
    run_prov_path = run / 'run_provenance.json'
    summary_path = run / 'case_summary.csv'
    require(run_prov_path.is_file() and summary_path.is_file(), 'Primary run metadata files are missing')
    integrity = read_json(Path(primary_freeze) / 'd0c3_primary_integrity.json')
    for name, digest in integrity['source_top_level_hashes'].items():
        require((run / name).is_file() and sha256_file(run / name) == digest,
                f'Primary run metadata hash mismatch: {name}')
    run_prov = read_json(run_prov_path)
    identity = run_prov.get('run_identity', {})
    require(run_prov.get('status') == 'PASS' and run_prov.get('scientific_analysis') is True
            and run_prov.get('scientific_result') is True and identity.get('mode') == 'scientific',
            'Primary run did not complete as scientific PASS')
    require(identity.get('timestep') == 495 and identity.get('alpha_bar') == 0.5000776052474976
            and identity.get('replicates') == 4 and identity.get('ddim_evaluations') == 50 and identity.get('eta') == 0.0,
            'Primary run inference settings mismatch')
    for run_key, contract_key in (('validation_split_sha256', 'validation_split_sha256'),
            ('spatial_manifest_sha256', 'spatial_manifest_sha256'),
            ('spatial_contract_sha256', 'spatial_contract_sha256'),
            ('b4b_sha256', 'd0b4b_inference_contract_sha256'), ('d0a_sha256', 'd0a_normalization_contract_sha256'),
            ('e1c_checkpoint_sha256', 'source_checkpoint_sha256'), ('d0c2_checkpoint_sha256', 'diffusion_checkpoint_sha256')):
        require(identity.get(run_key) == contract['sources'][contract_key], f'Primary run identity mismatch: {run_key}')
    require(run_prov.get('selected_source') == 'D0-C2 EMA state_dict key ema_state'
            and run_prov.get('cases_processed') == 270 and run_prov.get('windows_processed') == 9720
            and run_prov.get('source_model_unchanged') is True,
            'Primary run completion/selected-source provenance mismatch')
    source_cases = run / 'cases'
    require(source_cases.is_dir(), 'Primary run cases directory missing')
    found = {p.name for p in source_cases.iterdir() if p.is_dir()}
    require(found == set(ids_sorted), f'Primary case set differs from freeze: missing={len(set(ids_sorted)-found)}, unexpected={len(found-set(ids_sorted))}')
    manifest_by_id = {row['case_id']: row for row in case_rows}
    for case_id in ids_sorted:
        directory = source_cases / case_id
        record = manifest_by_id[case_id]
        checks_path = directory / 'CHECKSUMS.sha256'
        require(checks_path.is_file() and sha256_file(checks_path) == record['checksums_sha256'],
                f'Case checksum manifest mismatch: {case_id}')
        checksum = {}
        for line in checks_path.read_text(encoding='utf-8').splitlines():
            digest, name = line.split('  ', 1); checksum[name] = digest
        require(set(checksum) == {'case_deltaz.pt', 'case_provenance.json', 'replicate_provenance.csv'},
                f'Primary case checksum inventory mismatch: {case_id}')
        for filename, field in (('case_deltaz.pt', 'case_deltaz_sha256'),
                                ('case_provenance.json', 'case_provenance_sha256'),
                                ('replicate_provenance.csv', 'replicate_provenance_sha256')):
            require(record[field] == checksum.get(filename) == sha256_file(directory / filename),
                    f'Frozen case artifact hash mismatch: {case_id}/{filename}')
        cprov = read_json(directory / 'case_provenance.json')
        require(cprov.get('window_count') == 36 and cprov.get('timesteps') == [495]
                and cprov.get('split') == 'validation'
                and cprov.get('split_sha256') == contract['sources']['validation_split_sha256']
                and cprov.get('spatial_manifest_sha256') == contract['sources']['spatial_manifest_sha256']
                and cprov.get('spatial_contract_sha256') == contract['sources']['spatial_contract_sha256']
                and cprov.get('source_checkpoint_sha256') == contract['sources']['source_checkpoint_sha256']
                and cprov.get('diffusion_checkpoint_sha256') == contract['sources']['diffusion_checkpoint_sha256']
                and cprov.get('b4b_contract_sha256') == contract['sources']['d0b4b_inference_contract_sha256']
                and cprov.get('d0a_contract_sha256') == contract['sources']['d0a_normalization_contract_sha256']
                and cprov.get('source_model_unchanged') is True
                and cprov.get('source_model_fingerprint_before') == cprov.get('source_model_fingerprint_after')
                and cprov.get('original_shape') == list(ORIGINAL_SHAPE)
                and cprov.get('padded_shape') == list(PADDED_SHAPE),
                f'Primary case provenance invalid: {case_id}')
        _load_primary_tensor(directory)
    return case_rows, run_prov, sha256_file(case_manifest_path), sha256_file(run_prov_path)


def high_end_pad(array, target_shape, value=0):
    array = np.asarray(array)
    require(array.ndim == 3, 'Expected a 3D segmentation volume')
    target_shape = tuple(int(x) for x in target_shape)
    require(len(target_shape) == 3 and all(t >= s for s, t in zip(array.shape, target_shape)),
            'Padding target must be at least the source dimensions')
    pads = tuple((0, t - s) for s, t in zip(array.shape, target_shape))
    return np.pad(array, pads, mode='constant', constant_values=value)


def exact_block_reduce(mask, block=(8, 8, 8), *, mode='any'):
    mask = np.asarray(mask, dtype=bool)
    require(mask.ndim == 3 and all(dim % size == 0 for dim, size in zip(mask.shape, block)),
            'Volume must be 3D and exactly divisible into blocks')
    d, h, w = mask.shape
    bd, bh, bw = block
    blocks = mask.reshape(d // bd, bd, h // bh, bh, w // bw, bw)
    if mode == 'any':
        return blocks.any(axis=(1, 3, 5))
    if mode == 'all':
        return blocks.all(axis=(1, 3, 5))
    raise ValueError(f'Unsupported block reduction mode: {mode}')


def map_gt_to_latent(segmentation, original_shape, padded_shape):
    gt = np.asarray(segmentation)
    original_shape = tuple(int(x) for x in original_shape)
    padded_shape = tuple(int(x) for x in padded_shape)
    require(gt.shape == original_shape, f'GT shape mismatch: {gt.shape} != {original_shape}')
    require(set(np.unique(gt).tolist()) <= LABELS, 'GT contains unknown BraTS labels')
    padded = high_end_pad(gt, padded_shape, value=0)
    require(all(size % 8 == 0 for size in padded_shape), 'Padded dimensions must be divisible by 8')
    labels = {label: exact_block_reduce(padded == label, mode='any') for label in LABELS - {0}}
    clean_bg = exact_block_reduce(padded == 0, mode='all')
    return labels, clean_bg, padded


def comparison_masks(segmentation, valid_support, original_shape=ORIGINAL_SHAPE, padded_shape=PADDED_SHAPE):
    region_positive, clean_background, padded = map_gt_to_latent(segmentation, original_shape, padded_shape)
    valid = np.asarray(valid_support, dtype=bool)
    require(valid.shape == region_positive[1].shape, f'Valid-support shape mismatch: {valid.shape}')
    masks = {}
    for region, labels in REGIONS.items():
        positive = np.logical_or.reduce([region_positive[label] for label in labels]) & valid
        background = clean_background & valid
        masks[region] = (positive, background)
    return masks, padded


def binary_metrics(scores_positive, scores_background):
    pos = np.asarray(scores_positive, dtype=np.float64).reshape(-1)
    neg = np.asarray(scores_background, dtype=np.float64).reshape(-1)
    require(pos.size > 0 and neg.size > 0, 'AUROC/AUPRC require both classes')
    scores = np.concatenate((pos, neg))
    labels = np.concatenate((np.ones(pos.size, dtype=np.int8), np.zeros(neg.size, dtype=np.int8)))
    require(np.isfinite(scores).all(), 'DeltaZ scores must be finite')
    prevalence = float(pos.size / scores.size)
    ranks = rankdata(scores, method='average')
    auc = float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
    thresholds = np.unique(scores)[::-1]
    ap, previous_recall = 0.0, 0.0
    for threshold in thresholds:
        predicted = scores >= threshold
        tp = int(np.count_nonzero(predicted & (labels == 1)))
        fp = int(np.count_nonzero(predicted & (labels == 0)))
        recall = tp / pos.size
        precision = tp / (tp + fp)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return dict(auroc=auc, auprc=float(ap), prevalence=prevalence, auprc_lift=float(ap - prevalence))


def per_case_region_metrics(case_id, delta, valid_support, segmentation, original_shape, padded_shape):
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape == (1, *EXPECTED_SHAPE):
        delta = delta[0]
    require(delta.shape == EXPECTED_SHAPE, f'DeltaZ shape mismatch: {delta.shape}')
    valid = np.asarray(valid_support, dtype=bool)
    if valid.shape == (1, *EXPECTED_SHAPE):
        valid = valid[0]
    require(valid.shape == EXPECTED_SHAPE, f'Support shape mismatch: {valid.shape}')
    require(np.isfinite(delta[valid]).all(), 'Supported DeltaZ contains non-finite values')
    masks, _padded = comparison_masks(segmentation, valid, original_shape, padded_shape)
    rows = []
    for region in ('WT', 'TC', 'ET'):
        positive_mask, background_mask = masks[region]
        pos = delta[positive_mask]
        bg = delta[background_mask]
        row = dict(case_id=case_id, region=region, n_positive_cells=int(pos.size),
            n_background_cells=int(bg.size), positive_prevalence_in_comparison=(float(pos.size/(pos.size+bg.size)) if pos.size+bg.size else None),
            positive_DeltaZ_mean=(float(pos.mean()) if pos.size else None),
            positive_DeltaZ_median=(float(np.median(pos)) if pos.size else None),
            background_DeltaZ_mean=(float(bg.mean()) if bg.size else None),
            background_DeltaZ_median=(float(np.median(bg)) if bg.size else None),
            mean_difference=(float(pos.mean()-bg.mean()) if pos.size and bg.size else None),
            median_difference=(float(np.median(pos)-np.median(bg)) if pos.size and bg.size else None),
            AUROC=None, AUPRC=None, AUPRC_baseline_prevalence=None, AUPRC_lift=None,
            evaluable=False, evaluability_reason=None)
        reasons = []
        if pos.size == 0: reasons.append('no_positive_supported_cells')
        if bg.size == 0: reasons.append('no_clean_background_supported_cells')
        if reasons:
            row['evaluability_reason'] = ';'.join(reasons)
        else:
            metrics = binary_metrics(pos, bg)
            row.update(AUROC=metrics['auroc'], AUPRC=metrics['auprc'],
                       AUPRC_baseline_prevalence=metrics['prevalence'], AUPRC_lift=metrics['auprc_lift'],
                       evaluable=True, evaluability_reason='')
        rows.append(row)
    return rows


def bootstrap_median_ci(values, *, replicates=10000, seed=20260923, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and values.size > 0 and np.isfinite(values).all(), 'Bootstrap needs finite case-level values')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    medians = np.median(values[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    return [float(np.median(values)), float(np.quantile(medians, tail)), float(np.quantile(medians, 1.0-tail))]


def holm_adjust(pvalues):
    vals = np.asarray(pvalues, dtype=np.float64)
    adjusted = np.full(vals.shape, np.nan)
    valid_indices = np.flatnonzero(np.isfinite(vals))
    order = valid_indices[np.argsort(vals[valid_indices])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m-rank) * vals[idx]))
        adjusted[idx] = running
    return [float(value) if np.isfinite(value) else None for value in adjusted]


def paired_wilcoxon(values, null):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    differences = values - float(null)
    if differences.size == 0:
        return dict(n=0, statistic=None, p_value=None)
    if np.all(differences == 0):
        return dict(n=int(differences.size), statistic=0.0, p_value=1.0)
    result = wilcoxon(differences, zero_method='wilcox', alternative='two-sided', method='auto')
    return dict(n=int(differences.size), statistic=float(result.statistic), p_value=float(result.pvalue))


def aggregate_case_metrics(rows, *, replicates=10000, seed=20260923):
    result = []
    families = {name: [] for name in ('median_difference_vs_zero', 'AUROC_vs_0.5', 'AUPRC_lift_vs_zero')}
    for region in ('WT', 'TC', 'ET'):
        cases = [r for r in rows if r['region'] == region]
        evaluable = [r for r in cases if r['evaluable']]
        row = dict(region=region, total_cases=len(cases), evaluable_cases=len(evaluable),
                   non_evaluable_cases=len(cases)-len(evaluable))
        columns = ('AUROC', 'AUPRC', 'AUPRC_lift', 'mean_difference', 'median_difference')
        for column in columns:
            values = [r[column] for r in evaluable]
            if values:
                median, low, high = bootstrap_median_ci(values, replicates=replicates, seed=seed)
                row[f'median_{column}'] = median
                row[f'{column}_ci95_low'] = low
                row[f'{column}_ci95_high'] = high
            else:
                row[f'median_{column}'] = row[f'{column}_ci95_low'] = row[f'{column}_ci95_high'] = None
        row['proportion_median_difference_gt_zero'] = (
            float(np.mean([r['median_difference'] > 0 for r in evaluable])) if evaluable else None)
        tests = {
            'median_difference_vs_zero': paired_wilcoxon([r['median_difference'] for r in evaluable], 0.0),
            'AUROC_vs_0.5': paired_wilcoxon([r['AUROC'] for r in evaluable], 0.5),
            'AUPRC_lift_vs_zero': paired_wilcoxon([r['AUPRC_lift'] for r in evaluable], 0.0),
        }
        for family, test in tests.items():
            families[family].append(test['p_value'])
            row[f'{family}_n'] = test['n']
            row[f'{family}_statistic'] = test['statistic']
            row[f'{family}_p_raw'] = test['p_value']
        result.append(row)
    for family, pvalues in families.items():
        adjusted = holm_adjust(pvalues)
        for row, pvalue in zip(result, adjusted):
            row[f'{family}_p_holm'] = pvalue
    return result


def _load_primary_tensor(case_dir):
    path = Path(case_dir) / 'case_deltaz.pt'
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location='cpu')
    expected = (1, *EXPECTED_SHAPE)
    for key in ('DeltaZ_global', 'valid_support_global', 'contribution_count_global'):
        require(tuple(payload[key].shape) == expected, f'{key} shape mismatch in {case_dir}')
    delta = payload['DeltaZ_global'].detach().cpu().numpy()
    valid = payload['valid_support_global'].detach().cpu().numpy().astype(bool)
    count = payload['contribution_count_global'].detach().cpu().numpy()
    require(np.array_equal(valid, count > 0) and not np.any(count < 0), f'Support/count integrity failure in {case_dir}')
    require(np.isfinite(delta[valid]).all(), f'Non-finite supported DeltaZ in {case_dir}')
    original = tuple(payload['original_shape'].tolist())
    padded = tuple(payload['padded_shape'].tolist())
    require(original == ORIGINAL_SHAPE and padded == PADDED_SHAPE
            and tuple(payload['global_latent_shape'].tolist()) == EXPECTED_SHAPE,
            f'Frozen original/padded/latent shape metadata mismatch in {case_dir}')
    return delta, valid, original, padded


def runtime_provenance():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    try:
        sklearn_version = importlib.metadata.version('scikit-learn')
    except importlib.metadata.PackageNotFoundError:
        sklearn_version = None
    try:
        nib_version = importlib.metadata.version('nibabel')
    except importlib.metadata.PackageNotFoundError:
        nib_version = None
    return dict(runtime_git_commit=git('rev-parse', 'HEAD'), runtime_git_branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
        runtime_git_status=git('status', '--short').splitlines(), python_version=platform.python_version(),
        pytorch_version=torch.__version__, scikit_learn_version=sklearn_version,
        scipy_version=scipy.__version__, nibabel_version=nib_version)


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def _write_outputs(output, case_rows, aggregate_rows, provenance, integrity):
    output.mkdir()
    case_fields = list(case_rows[0]) if case_rows else []
    agg_fields = list(aggregate_rows[0]) if aggregate_rows else []
    _write_csv(output / 'per_case_metrics.csv', case_rows, case_fields)
    _write_csv(output / 'aggregate_summary.csv', aggregate_rows, agg_fields)
    (output / 'aggregate_summary.json').write_text(json.dumps(dict(regions=aggregate_rows), indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    (output / 'analysis_provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    (output / 'analysis_integrity.json').write_text(json.dumps(integrity, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    entries = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'MANIFEST.sha256').write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(entries.items())), encoding='utf-8')


def run_analysis(*, dataset_root=DATASET_DEFAULT, primary_run=PRIMARY_RUN_DEFAULT,
                 primary_freeze=PRIMARY_FREEZE_DEFAULT, output=OUTPUT_DEFAULT):
    contract_bytes = CONTRACT_PATH.read_bytes()
    contract = json.loads(contract_bytes)
    require(sha256_bytes(contract_bytes) == CONTRACT_SHA256, 'D0-C3b frozen analysis contract SHA256 mismatch')
    require(contract.get('analysis_started') is False and contract.get('results_observed') is False,
            'Analysis contract must be a pre-analysis design lock')
    require(not Path(output).exists(), f'Refusing to overwrite analysis output: {output}')
    case_rows, run_prov, case_manifest_sha, run_prov_sha = _verify_primary_source(primary_run, primary_freeze)
    dataset_root = Path(dataset_root).expanduser().resolve()
    require(dataset_root.is_dir(), f'Dataset root unavailable: {dataset_root}')
    all_metrics = []
    for record in case_rows:
        case_id = record['case_id']
        case_dir = Path(primary_run) / 'cases' / case_id
        delta, valid, original_shape, padded_shape = _load_primary_tensor(case_dir)
        require(tuple(original_shape) == ORIGINAL_SHAPE and tuple(padded_shape) == PADDED_SHAPE,
                f'Frozen shape metadata mismatch for {case_id}')
        seg_path = dataset_root / case_id / f'{case_id}-seg.nii.gz'
        require(seg_path.is_file(), f'Missing evaluation segmentation for {case_id}: {seg_path}')
        segmentation = np.asanyarray(nib.load(str(seg_path)).dataobj)
        all_metrics.extend(per_case_region_metrics(case_id, delta, valid, segmentation,
                                                   original_shape, padded_shape))
    agg = aggregate_case_metrics(all_metrics, replicates=contract['bootstrap']['replicates'], seed=contract['bootstrap']['seed'])
    runtime = runtime_provenance()
    require(runtime['scikit_learn_version'] is not None,
            'scikit-learn must be installed so its runtime version can be recorded in analysis provenance')
    provenance = dict(stage='D0-C3b', dataset_role='DEVELOPMENT_VALIDATION', independent_test=False,
        primary_freeze_manifest_sha256=sha256_file(Path(primary_freeze)/'MANIFEST.sha256'),
        primary_case_manifest_sha256=case_manifest_sha, primary_run_provenance_sha256=run_prov_sha,
        source_run_identity=run_prov['run_identity'], validation_split_sha256=contract['sources']['validation_split_sha256'],
        spatial_manifest_sha256=contract['sources']['spatial_manifest_sha256'],
        spatial_contract_sha256=contract['sources']['spatial_contract_sha256'],
        analysis_contract_sha256=sha256_bytes(contract_bytes), dataset_root=str(dataset_root), case_count=len(case_rows),
        gt_opened=True, gt_usage='evaluation only', runtime=runtime,
        statement=contract['ground_truth_usage'])
    integrity = dict(case_count=len(case_rows), case_region_rows=len(all_metrics),
        expected_case_region_rows=len(case_rows)*3, completed_case_set_match=True,
        unsupported_cells_excluded=True, interpolation_used=False, source_artifacts_modified=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.', dir=output.parent) as tmp:
        stage = Path(tmp)/output.name
        _write_outputs(stage, all_metrics, agg, provenance, integrity)
        os.replace(stage, output)
    return dict(per_case=all_metrics, aggregate=agg, provenance=provenance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DATASET_DEFAULT)
    parser.add_argument('--primary-run', type=Path, default=PRIMARY_RUN_DEFAULT)
    parser.add_argument('--primary-freeze', type=Path, default=PRIMARY_FREEZE_DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args(argv)
    run_analysis(dataset_root=args.dataset_root, primary_run=args.primary_run,
                 primary_freeze=args.primary_freeze, output=args.output_dir)


if __name__ == '__main__':
    main()
