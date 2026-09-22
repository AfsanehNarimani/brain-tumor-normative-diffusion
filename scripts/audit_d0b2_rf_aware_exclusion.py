"""D0-B2 train-only receptive-field-aware abnormal exclusion audit."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np
import torch
import yaml
import nibabel as nib

from analysis.rf_geometry import derive_encoder_geometry
from models.encoder import Encoder
from scripts.audit_d0b1_abnormal_occupancy import (
    BLOCK_SHAPE, DEFAULT_MANIFEST, EXPECTED_CASES,
    MANIFEST_FIELDS, MANIFEST_SHA256, PATCH_SHAPE, PATCHES_PER_CASE,
    RAW_ABNORMAL_LABELS, SUPPORT_THRESHOLD, TRAIN_SHA256, abnormal_mask,
    block_occupancy, load_case, sha256, support_mask, read_manifest,
)

EXPERIMENT_BASE_COMMIT = '7c0fcd74f04cfee82a0c0f24e7d4d350d24e6ce9'
DEFAULT_OUTPUT = 'outputs/d0b2_rf_aware_exclusion_audit'
FROZEN_B1_SUMMARY = 'outputs/frozen/d0b1_abnormal_occupancy_audit/abnormal_occupancy_summary.json'
DIRECT_TOLERANCE = 1e-6
LATENT_SHAPE = (12, 12, 12)
OUTPUTS = ('analysis_provenance.json', 'rf_geometry.json', 'patch_rf_statistics.csv',
           'case_rf_statistics.csv', 'rf_exclusion_summary.json')


def prefix_sum(mask):
    values = mask.to(torch.int64)
    return torch.nn.functional.pad(values.cumsum(0).cumsum(1).cumsum(2), (1, 0, 1, 0, 1, 0))


def box_sum(prefix, interval):
    (d0, d1), (h0, h1), (w0, w1) = interval
    return (prefix[d1, h1, w1] - prefix[d0, h1, w1] - prefix[d1, h0, w1] - prefix[d1, h1, w0]
            + prefix[d0, h0, w1] + prefix[d0, h1, w0] + prefix[d1, h0, w0] - prefix[d0, h0, w0])


def rf_contaminated(abnormal, geometry):
    if tuple(abnormal.shape) != PATCH_SHAPE or abnormal.dtype != torch.bool:
        raise ValueError('Expected boolean abnormal patch [96,96,96]')
    prefix = prefix_sum(abnormal)
    output = torch.zeros(LATENT_SHAPE, dtype=torch.bool)
    intervals = geometry.clipped_intervals
    for d in range(12):
        for h in range(12):
            for w in range(12):
                output[d, h, w] = box_sum(prefix, (intervals[d], intervals[h], intervals[w])) > 0
    return output


def direct_and_rf_metrics(image_patch, segmentation_patch, geometry):
    support = block_occupancy(support_mask(image_patch))
    abnormal = abnormal_mask(segmentation_patch)
    abnormal_occupancy = block_occupancy(abnormal)
    direct = abnormal_occupancy > 0
    contaminated = rf_contaminated(abnormal, geometry)
    valid = support >= SUPPORT_THRESHOLD
    valid_count = int(valid.sum())
    direct_count = int((valid & direct).sum())
    rf_count = int((valid & contaminated).sum())
    eligible = valid & ~contaminated
    ratio = lambda numerator: (float(numerator / valid_count) if valid_count else None)
    return dict(valid_support_count=valid_count, valid_support_fraction=float(valid.double().mean()),
                direct_abnormal_count_gt0=int(direct.sum()),
                direct_valid_support_excluded_count_gt0=direct_count,
                direct_valid_support_excluded_fraction_gt0=ratio(direct_count),
                rf_contaminated_count=int(contaminated.sum()),
                rf_contaminated_fraction_all_latent=float(contaminated.double().mean()),
                rf_valid_support_excluded_count=rf_count,
                rf_valid_support_excluded_fraction=ratio(rf_count),
                eligible_count=int(eligible.sum()), eligible_fraction_all_latent=float(eligible.double().mean()),
                eligible_fraction_of_valid_support=ratio(int(eligible.sum())),
                valid_support_zero_denominator=(valid_count == 0),
                abnormal_positive_patch=bool(abnormal.any()))


def distribution(values):
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(values):
        return {'count': 0}
    return dict(count=int(len(values)), mean=float(values.mean()), std=float(values.std(ddof=0)),
                min=float(values.min()), max=float(values.max()),
                **{f'p{p:02d}': float(np.percentile(values, p, method='linear')) for p in (1,5,25,50,75,95,99)})


def load_manifest(path, train_ids):
    return read_manifest(path, set(train_ids))


def safe_output(path, overwrite, inputs):
    requested = Path(path).expanduser()
    if requested.is_symlink(): raise ValueError('Output must not be a symlink')
    output = requested.resolve()
    frozen = (Path(__file__).resolve().parents[1] / 'outputs/frozen').resolve()
    if output == frozen or frozen in output.parents: raise ValueError('Output must not be under outputs/frozen')
    if any(output == path or output in path.parents for path in inputs): raise ValueError('Output overlaps input')
    if output.exists():
        if any(output.iterdir()) and not overwrite: raise FileExistsError('Output already contains files; use --overwrite')
        if any(p.name not in OUTPUTS or p.is_symlink() for p in output.iterdir()): raise ValueError('Unrelated output exists')
    return output


def run(data_config='configs/data.yaml', patch_manifest=DEFAULT_MANIFEST,
        output_dir=DEFAULT_OUTPUT, overwrite=False):
    config_path = Path(data_config).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text())
    split = Path(data['train_cases']).expanduser().resolve()
    if sha256(split) != TRAIN_SHA256: raise ValueError('Training split SHA256 mismatch')
    train_ids = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    if len(train_ids) != EXPECTED_CASES or len(set(train_ids)) != EXPECTED_CASES: raise ValueError('Expected 1080 unique training cases')
    manifest_path = Path(patch_manifest).expanduser().resolve()
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError('D0-B0 patch manifest SHA256 mismatch')
    rows = load_manifest(manifest_path, train_ids)
    b1_summary_path = Path(FROZEN_B1_SUMMARY).resolve()
    if not b1_summary_path.is_file(): raise FileNotFoundError(f'Missing frozen D0-B1 summary: {b1_summary_path}')
    b1_summary = json.loads(b1_summary_path.read_text())
    try: expected_direct = b1_summary['distributions']['all_patches']['fraction_valid_support_excluded_gt_0']['mean']
    except (KeyError, TypeError): raise ValueError('Frozen D0-B1 summary lacks direct >0 mean')
    output = safe_output(output_dir, overwrite, (config_path, split, manifest_path, b1_summary_path))
    geometry = derive_encoder_geometry(Encoder())
    root = Path(data['root']).expanduser().resolve()
    by_case = {case: [row for row in rows if row['case_id'] == case] for case in train_ids}
    fields = None; case_fields = None; all_metrics = []; case_rows = []
    with tempfile.TemporaryDirectory(prefix='.d0b2-', dir=output.parent) as staging:
        stage = Path(staging)
        with (stage / 'patch_rf_statistics.csv').open('w', newline='') as patch_stream, (stage / 'case_rf_statistics.csv').open('w', newline='') as case_stream:
            for case in train_ids:
                image, raw_mask = load_case(root, case); patch_metrics = []
                for identity in by_case[case]:
                    slices = tuple(slice(identity[f'crop_start_{axis}'], identity[f'crop_stop_{axis}']) for axis in 'dhw')
                    metrics = direct_and_rf_metrics(image[(slice(None), *slices)], raw_mask[slices], geometry)
                    row = {**identity, **metrics}; all_metrics.append(metrics); patch_metrics.append(metrics)
                    if fields is None: fields = list(row); csv.DictWriter(patch_stream, fieldnames=fields).writeheader()
                    csv.DictWriter(patch_stream, fieldnames=fields).writerow(row)
                case_row = {'case_id': case, 'patch_count': len(patch_metrics), 'valid_support_zero_denominator_patch_count': sum(m['valid_support_zero_denominator'] for m in patch_metrics)}
                for field in ('valid_support_fraction', 'direct_valid_support_excluded_fraction_gt0', 'rf_contaminated_fraction_all_latent', 'rf_valid_support_excluded_fraction', 'eligible_fraction_all_latent', 'eligible_fraction_of_valid_support'):
                    values = [m[field] for m in patch_metrics if m[field] is not None]
                    case_row[f'{field}_mean'] = float(np.mean(values)) if values else None
                    case_row[f'{field}_defined_patch_count'] = len(values)
                if case_fields is None: case_fields = list(case_row); csv.DictWriter(case_stream, fieldnames=case_fields).writeheader()
                csv.DictWriter(case_stream, fieldnames=case_fields).writerow(case_row)
                case_rows.append(case_row)
        direct_values = [m['direct_valid_support_excluded_fraction_gt0'] for m in all_metrics]
        rf_values = [m['rf_valid_support_excluded_fraction'] for m in all_metrics]
        eligible_values = [m['eligible_fraction_of_valid_support'] for m in all_metrics]
        positive = [m for m in all_metrics if m['abnormal_positive_patch']]
        case_distribution_fields = ('valid_support_fraction_mean',
                                    'direct_valid_support_excluded_fraction_gt0_mean',
                                    'rf_contaminated_fraction_all_latent_mean',
                                    'rf_valid_support_excluded_fraction_mean',
                                    'eligible_fraction_all_latent_mean',
                                    'eligible_fraction_of_valid_support_mean')
        direct_mean = distribution(direct_values)['mean']
        summary = dict(case_count=EXPECTED_CASES, patch_count=len(all_metrics),
                       abnormal_positive_patch_count=len(positive), abnormal_negative_patch_count=len(all_metrics)-len(positive),
                       zero_valid_support_patch_count=sum(m['valid_support_zero_denominator'] for m in all_metrics),
                       cases_with_zero_defined_eligible_support_ratio=sum(
                           not any(row['eligible_fraction_of_valid_support_defined_patch_count'] for row in case_rows)
                           for row in case_rows),
                       distributions={'all_patches': {key: distribution([m[key] for m in all_metrics]) for key in ('valid_support_fraction','direct_valid_support_excluded_fraction_gt0','rf_contaminated_fraction_all_latent','rf_valid_support_excluded_fraction','eligible_fraction_all_latent','eligible_fraction_of_valid_support')},
                                      'abnormal_positive_patches': {key: distribution([m[key] for m in positive]) for key in ('valid_support_fraction','direct_valid_support_excluded_fraction_gt0','rf_contaminated_fraction_all_latent','rf_valid_support_excluded_fraction','eligible_fraction_all_latent','eligible_fraction_of_valid_support')},
                                      'all_cases': {key: distribution([row[key] for row in case_rows if row[key] is not None]) for key in case_distribution_fields}},
                       direct_b1_consistency={'frozen_summary_path': str(b1_summary_path), 'frozen_summary_sha256': sha256(b1_summary_path), 'expected_mean': expected_direct, 'recomputed_mean': direct_mean, 'absolute_difference': abs(direct_mean-expected_direct), 'tolerance': DIRECT_TOLERANCE, 'status': 'PASS' if abs(direct_mean-expected_direct) <= DIRECT_TOLERANCE else 'FAIL'})
        if summary['direct_b1_consistency']['status'] != 'PASS': raise ValueError('D0-B1 direct exclusion consistency failed')
        write = lambda path, value: path.write_text(json.dumps(value, indent=2) + '\n')
        write(stage / 'rf_exclusion_summary.json', summary)
        write(stage / 'rf_geometry.json', dict(input_size=geometry.input_size, output_size=geometry.output_size, effective_stride=geometry.effective_stride, theoretical_rf_size=geometry.theoretical_rf_size, theoretical_intervals=geometry.theoretical_intervals, clipped_real_input_intervals=geometry.clipped_intervals))
        def git(*args): return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        provenance = dict(schema='d0b2_rf_aware_exclusion_audit', schema_version=1, audit_name='D0-B2 Receptive-Field-Aware Abnormal Exclusion Audit', runtime_git_commit=git('rev-parse','HEAD'), git_dirty=bool(git('status','--porcelain')), experiment_base_commit=EXPERIMENT_BASE_COMMIT, train_split_path=str(split), train_split_sha256=sha256(split), d0b0_patch_manifest_path=str(manifest_path), d0b0_patch_manifest_sha256=sha256(manifest_path), d0b1_frozen_summary_path=str(b1_summary_path), d0b1_frozen_summary_sha256=sha256(b1_summary_path), data_config_path=str(config_path), data_config_sha256=sha256(config_path), dataset_root=str(root), case_count=EXPECTED_CASES, patch_count=len(all_metrics), patch_shape=list(PATCH_SHAPE), latent_shape=list(LATENT_SHAPE), block_shape=list(BLOCK_SHAPE), encoder_geometry=dict(input_size=geometry.input_size, output_size=geometry.output_size, effective_stride=geometry.effective_stride, theoretical_rf_size=geometry.theoretical_rf_size), raw_abnormal_labels=list(RAW_ABNORMAL_LABELS), support_definition='Raw MRI support is ANY modality nonzero before normalization, mapped by exact 8x8x8 occupancy; valid support requires occupancy >= 0.50.', direct_abnormal_rule='Any raw abnormal voxel in the corresponding 8x8x8 input block.', rf_contamination='Any raw abnormal voxel in the clipped real-input theoretical receptive-field box.', support_threshold=SUPPORT_THRESHOLD, eligibility='valid_support AND NOT rf_contaminated', validation_data_used=False, model_training_performed=False, diffusion_model_loaded=False, new_random_sampling_performed=False, rejection_sampling_used=False, augmentation_used=False, final_corruption_strategy_selected=False, final_abnormal_policy_selected=False, python_version=platform.python_version(), numpy_version=np.__version__, pytorch_version=str(torch.__version__), nibabel_version=nib.__version__, timestamp_utc=datetime.now(timezone.utc).isoformat())
        write(stage / 'analysis_provenance.json', provenance)
        safe_output(output, overwrite, (config_path, split, manifest_path, b1_summary_path)); output.parent.mkdir(parents=True, exist_ok=True); output.mkdir(exist_ok=True)
        for name in OUTPUTS: (stage / name).replace(output / name)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--data-config', default='configs/data.yaml'); parser.add_argument('--patch-manifest', default=DEFAULT_MANIFEST); parser.add_argument('--output-dir', default=DEFAULT_OUTPUT); parser.add_argument('--overwrite', action='store_true'); print(run(**vars(parser.parse_args())))


if __name__ == '__main__': main()
