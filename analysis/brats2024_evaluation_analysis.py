"""Read-only summaries of official-package outputs; no metric recomputation.

Finite summaries use population std (ddof=0) and linear quantiles. All cases,
including empty references, retain upstream observations. Nonfinite observations
are counted, never imputed. Extreme-case ranking uses finite observations only.
"""
import csv
import hashlib
import io
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ('NETC', 'SNFH', 'ET', 'RC', 'WT', 'TC')
METRICS = ('global_bin_dsc', 'global_bin_hd95', 'global_bin_nsd', 'rq',
           'sq_dsc', 'pq_dsc', 'sq_hd95', 'sq_nsd', 'prec', 'rec')
BURDEN = ('n_ref_instances', 'n_pred_instances', 'tp', 'fp', 'fn')
COLUMNS = ('case_id', 'Labels', 'n_ref_instances', 'n_pred_instances', 'tp', 'fp', 'fn',
           'rq', 'sq_dsc', 'sq_dsc_std', 'pq_dsc', 'sq_hd95', 'sq_hd95_std',
           'sq_nsd', 'sq_nsd_std', 'instance_voxel_count_ref', 'instance_volume_ref',
           'global_bin_volume_pred', 'global_bin_volume_ref', 'global_bin_dsc',
           'global_bin_hd95', 'global_bin_nsd', 'reference_instances', 'prec', 'rec')
COUNTS = ('total_count', 'finite_count', 'nan_count', 'positive_infinity_count', 'negative_infinity_count')


def load_metrics(content, expected_cases=270):
    """Validate literal IDs/regions, numeric fields and complete case coverage."""
    header = next(csv.reader(io.StringIO(content)), [])
    if len(header) != len(set(header)) or set(header) != set(COLUMNS):
        raise ValueError('official_metrics.csv schema: expected exactly the documented columns')
    frame = pd.read_csv(io.StringIO(content), dtype=str, keep_default_na=False)
    if frame['case_id'].str.strip().eq('').any() or (frame.case_id != frame.case_id.str.strip()).any():
        raise ValueError('Invalid empty or whitespace-padded case ID')
    if frame.duplicated(['case_id', 'Labels']).any():
        raise ValueError('Duplicate case/region pairs')
    if set(frame.Labels) != set(REGIONS):
        raise ValueError('Unexpected or missing regions')
    if frame.case_id.nunique() != expected_cases or len(frame) != expected_cases * len(REGIONS):
        raise ValueError(f'Expected {expected_cases} unique cases and {expected_cases * 6} rows')
    if any(set(group.Labels) != set(REGIONS) for _, group in frame.groupby('case_id')):
        raise ValueError('Every case must contain exactly six expected regions')
    numeric = set(METRICS + BURDEN + ('global_bin_volume_ref', 'global_bin_volume_pred'))
    for column in numeric:
        try:
            frame[column] = frame[column].map(lambda value: float(value) if value.strip() else float('nan'))
        except (ValueError, TypeError) as error:
            raise ValueError(f'Invalid numeric value in {column}') from error
    # Presence and count fields must be defined; NaN/Infinity are valid metric outputs.
    for column in BURDEN + ('global_bin_volume_ref', 'global_bin_volume_pred'):
        values = frame[column].to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'Invalid nonfinite or negative count/volume: {column}')
        if column in BURDEN and (values != np.floor(values)).any():
            raise ValueError(f'Noninteger instance count: {column}')
    return frame.sort_values(['case_id', 'Labels'], kind='stable').reset_index(drop=True)


def statistics(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    result = dict(total_count=len(values), finite_count=len(finite), nan_count=int(np.isnan(values).sum()),
                  positive_infinity_count=int(np.isposinf(values).sum()), negative_infinity_count=int(np.isneginf(values).sum()))
    for name, function in (('mean', np.mean), ('std', np.std), ('median', np.median),
                           ('q25', lambda x: np.quantile(x, .25)), ('q75', lambda x: np.quantile(x, .75)),
                           ('min', np.min), ('max', np.max)):
        result[name] = float(function(finite)) if len(finite) else None
    return result


def overlap_analysis(original):
    required = ('case_count', 'cases_with_overlap', 'total_foreground_voxels', 'total_overlapping_voxels',
                'overall_overlap_fraction', 'maximum_case_overlap_fraction', 'maximum_overlap_channels_seen',
                'pairwise_overlap_counts')
    if any(key not in original for key in required):
        raise ValueError('overlap_summary.json missing required fields')
    result = dict(original)
    result['fraction_cases_with_overlap'] = original['cases_with_overlap'] / original['case_count'] if original['case_count'] else None
    denominator = original['total_overlapping_voxels']
    result['pairwise_fractions_relative_to_overlapping_voxels'] = {
        pair: count / denominator if denominator else None for pair, count in original['pairwise_overlap_counts'].items()}
    result['interpretation'] = 'Pairs are not mutually exclusive; fractions may sum above one. Zero denominators yield null.'
    return result


def tables(frame, top_k=10):
    if top_k < 1:
        raise ValueError('top-K must be positive')
    metrics, statuses, burdens, fps, extremes, globals_, lesions = [], [], [], [], [], [], []
    for region in REGIONS:
        group = frame[frame.Labels == region]
        ref, pred = group.global_bin_volume_ref > 0, group.global_bin_volume_pred > 0
        status = dict(region=region, reference_positive_cases=int(ref.sum()), reference_empty_cases=int((~ref).sum()),
                      prediction_positive_cases=int(pred.sum()), prediction_empty_cases=int((~pred).sum()),
                      both_positive=int((ref & pred).sum()), reference_positive_prediction_empty=int((ref & ~pred).sum()),
                      reference_empty_prediction_positive=int((~ref & pred).sum()), both_empty=int((~ref & ~pred).sum()))
        statuses.append(status)
        stats = {metric: statistics(group[metric]) for metric in METRICS}
        metrics.extend(dict(region=region, metric=metric, **values) for metric, values in stats.items())
        totals = {f'total_{name}': int(group[column].sum()) for name, column in zip(
            ('reference_instances', 'predicted_instances', 'tp', 'fp', 'fn'), BURDEN)}
        for column in BURDEN:
            burdens.append(dict(region=region, quantity=column, total=int(group[column].sum()), **statistics(group[column]), **totals))
        fps.append(dict(region=region, cases_with_fp=int((group.fp > 0).sum()), fraction_cases_with_fp=float((group.fp > 0).mean()),
                        total_fp=totals['total_fp'], mean_fp_per_case=float(group.fp.mean()), median_fp_per_case=float(group.fp.median()),
                        max_fp_in_single_case=int(group.fp.max())))
        for metric, ascending, criterion in (('global_bin_dsc', True, 'lowest_global_binary_dice'),
                ('global_bin_hd95', False, 'highest_finite_hd95'), ('fp', False, 'highest_fp'), ('fn', False, 'highest_fn')):
            ranked = group[np.isfinite(group[metric])].sort_values([metric, 'case_id'], ascending=[ascending, True], kind='stable').head(top_k)
            for rank, (_, row) in enumerate(ranked.iterrows(), 1):
                extremes.append(dict(case_id=row.case_id, region=region, criterion=criterion, rank=rank, metric=metric, value=float(row[metric])))
        global_row = dict(region=region, reference_positive_cases=status['reference_positive_cases'])
        lesion_row = dict(region=region, **totals)
        for metric, values in stats.items():
            target = global_row if metric.startswith('global_bin_') else lesion_row
            prefix = {'prec': 'precision', 'rec': 'recall'}.get(metric, metric)
            for key in COUNTS:
                target[f'{prefix}_{key}'] = values[key]
            if target is global_row:
                suffix = '_finite' if metric == 'global_bin_hd95' else ''
                for key in ('mean', 'std', 'median'):
                    target[f'{prefix}_{key}{suffix}'] = values[key]
            else:
                target[f'{prefix}_mean_finite'] = values['mean']
        globals_.append(global_row)
        lesions.append(lesion_row)
    return dict(metric_statistics=metrics, region_status=statuses, instance_burden=burdens,
                false_positive_burden=fps, extreme_cases=extremes, paper_global_metrics=globals_, paper_lesion_metrics=lesions)


def git_info(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[1],
                                       stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except (OSError, subprocess.SubprocessError):
        return None


def analyze(evaluation_dir, top_k=10, expected_cases=270):
    """Write derived reports only beneath analysis/. Expected count override is for tests."""
    root = Path(evaluation_dir).expanduser().resolve()
    source, overlap_path = root / 'official_metrics.csv', root / 'overlap_summary.json'
    for path in (source, overlap_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required evaluation artifact missing: {path}')
    # Hash the same bytes parsed, avoiding metadata/content drift during reads.
    csv_bytes, overlap_bytes = source.read_bytes(), overlap_path.read_bytes()
    frame = load_metrics(csv_bytes.decode('utf-8'), expected_cases)
    overlap = overlap_analysis(json.loads(overlap_bytes))
    if overlap['case_count'] != expected_cases:
        raise ValueError('Overlap case count disagrees with metrics case count')
    reports = tables(frame, top_k)
    dirty = git_info('status', '--porcelain')
    metadata = dict(analysis_timestamp=datetime.now(timezone.utc).isoformat(), input_csv_path=str(source),
                    official_metrics_sha256=hashlib.sha256(csv_bytes).hexdigest(), overlap_summary_path=str(overlap_path),
                    overlap_summary_sha256=hashlib.sha256(overlap_bytes).hexdigest(), git_commit=git_info('rev-parse', 'HEAD'),
                    git_dirty=None if dirty is None else bool(dirty), python_version=platform.python_version(),
                    pandas_version=pd.__version__, numpy_version=np.__version__, expected_case_count=expected_cases,
                    expected_regions=list(REGIONS), top_k=top_k)
    summary = dict(case_count=frame.case_id.nunique(), row_count=len(frame), regions=list(REGIONS),
                   statistics_policy='All upstream case observations; finite-only summaries with explicit nonfinite counts; no imputation.',
                   standard_deviation='population (ddof=0)', quantiles='linear interpolation',
                   extreme_selection='Finite values only; top-K per region/criterion; ties ordered lexicographically by case ID.',
                   metric_counts=[{key: row[key] for key in ('region', 'metric') + COUNTS} for row in reports['metric_statistics']],
                   artifacts=[name + '.csv' for name in reports] + ['overlap_analysis.json', 'analysis_metadata.json', 'analysis_summary.json'])
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    # Refuse symlink destinations that could redirect writes into frozen artifacts.
    if output.is_symlink() or any((output / name).is_symlink() for name in summary['artifacts']):
        raise ValueError('Analysis outputs must not be symlinks')
    for name, rows in reports.items():
        pd.DataFrame(rows).to_csv(output / f'{name}.csv', index=False, lineterminator='\n')
    for name, value in (('overlap_analysis', overlap), ('analysis_metadata', metadata), ('analysis_summary', summary)):
        (output / f'{name}.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return output
