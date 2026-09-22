"""D0-B1 train-only annotated abnormal occupancy audit using frozen D0-B0 patches."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile

import nibabel as nib
import numpy as np
import torch
import yaml

from datasets.brats_dataset import MODALITIES, load_subject

EXPECTED_CASES = 1080
PATCHES_PER_CASE = 20
TOTAL_PATCHES = EXPECTED_CASES * PATCHES_PER_CASE
PATCH_SHAPE = (96, 96, 96)
LATENT_SHAPE = (12, 12, 12)
BLOCK_SHAPE = (8, 8, 8)
RAW_ABNORMAL_LABELS = (1, 2, 3, 4)
SUPPORT_THRESHOLD = 0.50
CANDIDATE_THRESHOLDS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50)
TRAIN_SHA256 = '581dc6326a22cb0b9582fa249b8e13fd948597a60c5f454cb5ade0493fa7c18e'
MANIFEST_SHA256 = '9c4ba67fc195537a5500229722ce55e4772ea5b7d135b351b3a2fc4b857f44b9'
EXPERIMENT_BASE_COMMIT = '12738979614d4268a8f2e42774808758317db4dc'
DEFAULT_MANIFEST = 'outputs/frozen/d0b_support_mask_audit/patch_manifest.csv'
ARTIFACTS = ('patch_abnormal_statistics.csv', 'case_abnormal_statistics.csv',
             'abnormal_occupancy_summary.json', 'analysis_provenance.json')
MANIFEST_FIELDS = ('case_id', 'patch_index', 'seed', 'crop_start_d', 'crop_start_h',
                   'crop_start_w', 'crop_stop_d', 'crop_stop_h', 'crop_stop_w')
SUPPORT_DEFINITION = 'Any raw T1n/T1c/T2w/T2f modality is nonzero at the voxel before normalization.'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'count': 0}
    return dict(count=int(len(values)), mean=float(values.mean()), std=float(values.std(ddof=0)),
                min=float(values.min()), max=float(values.max()),
                **{f'p{p:02d}': float(np.percentile(values, p, method='linear'))
                   for p in (1, 5, 25, 50, 75, 95, 99)})


def block_occupancy(mask):
    mask = torch.as_tensor(mask)
    if tuple(mask.shape) != PATCH_SHAPE:
        raise ValueError(f'Expected patch shape {PATCH_SHAPE}; got {tuple(mask.shape)}')
    if mask.dtype != torch.bool:
        raise ValueError('Occupancy input must be boolean')
    return mask.reshape(12, 8, 12, 8, 12, 8).sum(dim=(1, 3, 5)).to(torch.float64) / 512.0


def support_mask(image):
    image = torch.as_tensor(image)
    if image.ndim != 4 or image.shape[0] != 4 or tuple(image.shape[1:]) != PATCH_SHAPE:
        raise ValueError('Expected raw multimodal patch [4,96,96,96]')
    if not torch.isfinite(image).all():
        raise ValueError('Non-finite raw MRI patch')
    return image.ne(0).any(dim=0)


def abnormal_mask(raw_mask):
    raw_mask = torch.as_tensor(raw_mask)
    if tuple(raw_mask.shape) != PATCH_SHAPE:
        raise ValueError('Expected raw segmentation patch [96,96,96]')
    if not torch.isfinite(raw_mask).all() or not torch.equal(raw_mask, raw_mask.round()):
        raise ValueError('Segmentation patch must be finite integer-valued')
    unknown = set(raw_mask.unique().tolist()) - {0, *RAW_ABNORMAL_LABELS}
    if unknown:
        raise ValueError(f'Unknown raw segmentation labels: {sorted(unknown)}')
    return torch.isin(raw_mask.to(torch.int64), torch.tensor(RAW_ABNORMAL_LABELS))


def patch_metrics(image_patch, segmentation_patch):
    support = block_occupancy(support_mask(image_patch))
    abnormal = block_occupancy(abnormal_mask(segmentation_patch))
    valid = support >= SUPPORT_THRESHOLD
    abnormal_voxels = abnormal_mask(segmentation_patch)
    metrics = dict(input_abnormal_fraction=float(abnormal_voxels.double().mean()),
                   latent_abnormal_occupancy_mean=float(abnormal.mean()),
                   latent_abnormal_occupancy_std=float(abnormal.std(unbiased=False).item()),
                   latent_abnormal_occupancy_min=float(abnormal.min()),
                   latent_abnormal_occupancy_max=float(abnormal.max()),
                   abnormal_positive_patch=bool(abnormal_voxels.any()),
                   valid_support_latent_count=int(valid.sum()),
                   valid_support_zero_denominator=bool(valid.sum() == 0))
    for threshold in CANDIDATE_THRESHOLDS:
        suffix = 'gt_0' if threshold == 0 else f'ge_{threshold:.2f}'.replace('.', '_')
        all_fraction = float((abnormal > threshold).double().mean()) if threshold == 0 else float((abnormal >= threshold).double().mean())
        metrics[f'fraction_all_latent_abnormal_{suffix}'] = all_fraction
        excluded = (abnormal > threshold) if threshold == 0 else (abnormal >= threshold)
        count = int((valid & excluded).sum())
        metrics[f'abnormal_excluded_cell_count_{suffix}'] = count
        metrics[f'fraction_valid_support_excluded_{suffix}'] = (
            float(count / valid.sum()) if valid.any() else None)
    return metrics


def read_manifest(path, train_ids):
    path = Path(path).expanduser().resolve()
    if sha256(path) != MANIFEST_SHA256:
        raise ValueError('D0-B0 patch manifest SHA256 mismatch')
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(MANIFEST_FIELDS):
            raise ValueError('D0-B0 patch manifest schema mismatch')
        rows = []
        for row in reader:
            try:
                values = {key: int(row[key]) for key in MANIFEST_FIELDS[1:]}
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('Malformed patch manifest row') from error
            case = row['case_id']
            if not case or case not in train_ids:
                raise ValueError(f'Manifest case is not in training split: {case}')
            values['case_id'] = case
            if values['patch_index'] not in range(PATCHES_PER_CASE):
                raise ValueError('Manifest patch index outside 0..19')
            starts = tuple(values[f'crop_start_{axis}'] for axis in 'dhw')
            stops = tuple(values[f'crop_stop_{axis}'] for axis in 'dhw')
            if any(start < 0 or stop - start != 96 for start, stop in zip(starts, stops)):
                raise ValueError('Manifest patch coordinates are invalid')
            rows.append(values)
    if len(rows) != TOTAL_PATCHES or len({r['case_id'] for r in rows}) != EXPECTED_CASES:
        raise ValueError('Manifest must contain exactly 21600 rows and 1080 cases')
    keys = {(r['case_id'], r['patch_index']) for r in rows}
    if len(keys) != TOTAL_PATCHES or any(sum(r['case_id'] == case for r in rows) != 20 for case in train_ids):
        raise ValueError('Manifest must contain exactly 20 unique patches per case')
    if {r['case_id'] for r in rows} != set(train_ids):
        raise ValueError('Manifest case IDs do not match training split')
    return rows


def load_case(root, case):
    folder = Path(root).expanduser() / case
    paths = {key: folder / f'{case}-{key.lower()}.nii.gz' for key in MODALITIES}
    paths['mask'] = folder / f'{case}-seg.nii.gz'
    return load_subject(dict(subject_id=case, paths=paths))


def safe_output(path, overwrite, inputs):
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError('Output must not be a symlink')
    output = requested.resolve()
    frozen = (Path(__file__).resolve().parents[1] / 'outputs/frozen').resolve()
    if output == frozen or frozen in output.parents:
        raise ValueError('Output must not be under outputs/frozen')
    if any(output == p or output in p.parents for p in inputs):
        raise ValueError('Output overlaps an input')
    if output.exists():
        if not overwrite and any(output.iterdir()):
            raise FileExistsError('Output already contains files; use --overwrite')
        if any(p.name not in ARTIFACTS or p.is_symlink() or not p.is_file() for p in output.iterdir()):
            raise ValueError('Refusing to overwrite unrelated output files')
    return output


def run(data_config='configs/data.yaml', patch_manifest=DEFAULT_MANIFEST,
        output_dir='outputs/d0b1_abnormal_occupancy_audit', overwrite=False):
    config_path = Path(data_config).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text())
    split = Path(data['train_cases']).expanduser().resolve()
    if sha256(split) != TRAIN_SHA256:
        raise ValueError('Training split SHA256 mismatch')
    train_ids = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    if len(train_ids) != EXPECTED_CASES or len(set(train_ids)) != EXPECTED_CASES:
        raise ValueError('Training split must contain exactly 1080 unique cases')
    manifest_path = Path(patch_manifest).expanduser().resolve()
    rows = read_manifest(manifest_path, set(train_ids))
    output = safe_output(output_dir, overwrite, (config_path, split, manifest_path))
    root = Path(data['root']).expanduser().resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('Output must be separate from dataset root')
    by_case = {case: [r for r in rows if r['case_id'] == case] for case in train_ids}
    summary_fields = ('input_abnormal_fraction', 'latent_abnormal_occupancy_mean',
                      *(f'fraction_all_latent_abnormal_{"gt_0" if t == 0 else f"ge_{t:.2f}".replace(".", "_")}' for t in CANDIDATE_THRESHOLDS),
                      *(f'fraction_valid_support_excluded_{"gt_0" if t == 0 else f"ge_{t:.2f}".replace(".", "_")}' for t in CANDIDATE_THRESHOLDS))
    values = {field: [] for field in summary_fields}
    positive_values = {field: [] for field in summary_fields}
    with tempfile.TemporaryDirectory(prefix='.d0b1-', dir=output.parent) as staging:
        stage = Path(staging)
        patch_stream = (stage / 'patch_abnormal_statistics.csv').open('w', newline='')
        case_stream = (stage / 'case_abnormal_statistics.csv').open('w', newline='')
        with patch_stream, case_stream:
            patch_writer = case_writer = None
            for case in train_ids:
                image, raw_mask = load_case(root, case)
                case_metrics = []
                for identity in by_case[case]:
                    slices = tuple(slice(identity[f'crop_start_{axis}'], identity[f'crop_stop_{axis}']) for axis in 'dhw')
                    metrics = patch_metrics(image[(slice(None), *slices)], raw_mask[slices])
                    row = {**identity, **metrics}
                    if patch_writer is None:
                        patch_writer = csv.DictWriter(patch_stream, fieldnames=list(row)); patch_writer.writeheader()
                    patch_writer.writerow(row); case_metrics.append(metrics)
                    for field in values:
                        if metrics[field] is not None:
                            values[field].append(metrics[field])
                            if metrics['abnormal_positive_patch']:
                                positive_values[field].append(metrics[field])
                case_row = {'case_id': case, 'patch_count': 20,
                            'abnormal_positive_patch_count': sum(m['abnormal_positive_patch'] for m in case_metrics)}
                for field in ('input_abnormal_fraction', 'latent_abnormal_occupancy_mean'):
                    arr = np.array([m[field] for m in case_metrics])
                    for name, fn in (('mean', np.mean), ('median', np.median), ('min', np.min), ('max', np.max)):
                        case_row[f'{field}_{name}'] = float(fn(arr))
                for threshold in CANDIDATE_THRESHOLDS:
                    suffix = 'gt_0' if threshold == 0 else f'ge_{threshold:.2f}'.replace('.', '_')
                    all_field = f'fraction_all_latent_abnormal_{suffix}'
                    valid_field = f'fraction_valid_support_excluded_{suffix}'
                    case_row[f'{all_field}_mean'] = float(np.mean([m[all_field] for m in case_metrics]))
                    defined = [m[valid_field] for m in case_metrics if m[valid_field] is not None]
                    case_row[f'{valid_field}_mean'] = float(np.mean(defined)) if defined else None
                    case_row[f'{valid_field}_defined_patch_count'] = len(defined)
                if case_writer is None:
                    case_writer = csv.DictWriter(case_stream, fieldnames=list(case_row)); case_writer.writeheader()
                case_writer.writerow(case_row)
        distributions = {'all_patches': {k: distribution(v) for k, v in values.items()},
                         'abnormal_positive_patches': {k: distribution(v) for k, v in positive_values.items()}}
        positive_count = sum(1 for value in values['input_abnormal_fraction'] if value > 0)
        summary = dict(case_count=EXPECTED_CASES, total_patch_count=TOTAL_PATCHES,
                       abnormal_positive_patch_count=positive_count,
                       abnormal_negative_patch_count=TOTAL_PATCHES - positive_count,
                       candidate_thresholds=list(CANDIDATE_THRESHOLDS), locked_support_threshold=SUPPORT_THRESHOLD,
                       distributions=distributions, final_abnormal_threshold_selected=False)
        def git(*args): return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        provenance = dict(schema='d0b1_abnormal_occupancy_audit', schema_version=1,
            analysis_identity='D0-B1 latent abnormal occupancy audit', experiment_base_commit=EXPERIMENT_BASE_COMMIT,
            runtime_git_commit=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')),
            train_split_sha256=sha256(split), patch_manifest_path=str(manifest_path), patch_manifest_sha256=sha256(manifest_path),
            case_count=EXPECTED_CASES, patch_count=TOTAL_PATCHES, patch_shape=list(PATCH_SHAPE), latent_shape=list(LATENT_SHAPE), block_shape=list(BLOCK_SHAPE),
            raw_abnormal_labels=list(RAW_ABNORMAL_LABELS), support_definition='D0-B0 raw MRI support: ANY modality nonzero before normalization',
            locked_support_threshold=SUPPORT_THRESHOLD, candidate_abnormal_thresholds=list(CANDIDATE_THRESHOLDS),
            validation_data_used=False, model_training_performed=False, augmentation_used=False,
            new_random_sampling_performed=False, rejection_sampling_used=False, segmentation_used=True,
            segmentation_purpose='Ground-truth labels are used only to audit a future training exclusion policy, never as an inference input.',
            final_abnormal_threshold_selected=False, statistics_convention='population standard deviation, ddof=0',
            no_model_loaded=True, no_diffusion_model_loaded=True, no_rejection_sampling=True,
            python_version=platform.python_version(), numpy_version=np.__version__, pytorch_version=str(torch.__version__), nibabel_version=nib.__version__,
            data_config_path=str(config_path), data_config_sha256=sha256(config_path),
            dataset_root=str(root), timestamp_utc=datetime.now(timezone.utc).isoformat())
        (stage / 'abnormal_occupancy_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (stage / 'analysis_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        safe_output(output, overwrite, (config_path, split, manifest_path)); output.parent.mkdir(parents=True, exist_ok=True); output.mkdir(exist_ok=True)
        for name in ARTIFACTS: (stage / name).replace(output / name)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', default='configs/data.yaml')
    parser.add_argument('--patch-manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--output-dir', default='outputs/d0b1_abnormal_occupancy_audit')
    parser.add_argument('--overwrite', action='store_true')
    print(run(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
