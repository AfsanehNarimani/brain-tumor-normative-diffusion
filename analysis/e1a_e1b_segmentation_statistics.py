"""Paired E1-A versus E1-B internal presence-aware segmentation analysis."""

import csv
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon

from analysis.paired_evaluation_comparison import (
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, REGIONS, WILCOXON_PARAMETERS,
    _git, bootstrap_ci, holm_adjust,
)


EXPECTED_CASES = 270
EXPECTED_POSITIVE_COUNTS = {'NETC': 114, 'SNFH': 270, 'ET': 198,
                            'RC': 230, 'WT': 270, 'TC': 200}
TIE_TOLERANCE = 1e-8
CI_METHOD = 'paired percentile bootstrap of mean E1-B-minus-E1-A Dice; 2.5th and 97.5th percentiles'
DELTA_DEFINITION = 'E1-B Dice minus E1-A Dice'
ARTIFACTS = ('paired_segmentation_statistics.csv', 'paired_case_metrics.csv',
             'comparison_summary.json', 'comparison_metadata.json')
CASE_FIELDS = ('case_id', 'region', 'target_present', 'e1a_dice', 'e1b_dice',
               'delta_e1b_minus_e1a')
STAT_FIELDS = ('region', 'positive_case_count', 'e1a_mean_dice', 'e1b_mean_dice',
               'mean_delta_e1b_minus_e1a', 'median_delta',
               'bootstrap_mean_delta_ci_lower', 'bootstrap_mean_delta_ci_upper',
               'wilcoxon_raw_p', 'wilcoxon_holm_adjusted_p', 'e1b_win_count',
               'tie_count', 'e1a_win_count', 'e1b_win_fraction', 'e1a_win_fraction')


def _preview(values):
    values = sorted(values)
    return f'{values[:5]} ({len(values)} total)'


def _required_fields():
    return ('case_id', *(f'{region}_{suffix}' for region in REGIONS
                         for suffix in ('target_present', 'dice')))


def load_cases(path):
    """Load a strict internal per-case file without imputing absent-region Dice."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing internal per-case metrics: {path}')
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        expected = _required_fields()
        if len(fields) != len(set(fields)):
            raise ValueError(f'{path}: duplicate CSV column names')
        missing, unexpected = set(expected) - set(fields), set(fields) - set(expected)
        if missing or unexpected:
            raise ValueError(f'{path}: CSV columns mismatch: missing={sorted(missing)}, '
                             f'unexpected={sorted(unexpected)}')
        cases = {}
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'{path}:{line}: malformed CSV row')
            case_id = row['case_id']
            if not case_id or case_id != case_id.strip():
                raise ValueError(f'{path}:{line}: invalid case_id')
            if case_id in cases:
                raise ValueError(f'{path}:{line}: duplicate case_id {case_id}')
            regions = {}
            for region in REGIONS:
                present = row[f'{region}_target_present']
                if present not in ('0', '1'):
                    raise ValueError(f'{path}:{line}: invalid {region}_target_present: {present!r}')
                raw = row[f'{region}_dice']
                if present == '0':
                    if raw != '':
                        raise ValueError(f'{path}:{line}: target-absent {region} Dice must be blank')
                    regions[region] = (False, None)
                else:
                    if raw == '':
                        raise ValueError(f'{path}:{line}: target-present {region} Dice is missing')
                    try:
                        dice = float(raw)
                    except ValueError as error:
                        raise ValueError(f'{path}:{line}: invalid target-present {region} Dice') from error
                    if not math.isfinite(dice):
                        raise ValueError(f'{path}:{line}: non-finite target-present {region} Dice')
                    regions[region] = (True, dice)
            cases[case_id] = regions
    return cases


def validate_pair(e1a_csv, e1b_csv, expected_cases=EXPECTED_CASES,
                  expected_positive_counts=EXPECTED_POSITIVE_COUNTS):
    e1a, e1b = load_cases(e1a_csv), load_cases(e1b_csv)
    if len(e1a) != expected_cases or len(e1b) != expected_cases:
        raise ValueError(f'Expected {expected_cases} rows in each CSV; '
                         f'E1-A={len(e1a)}, E1-B={len(e1b)}')
    e1a_ids, e1b_ids = set(e1a), set(e1b)
    if e1a_ids != e1b_ids:
        raise ValueError('Case set mismatch: E1-A-only=' + _preview(e1a_ids - e1b_ids)
                         + ', E1-B-only=' + _preview(e1b_ids - e1a_ids))
    if set(expected_positive_counts) != set(REGIONS):
        raise ValueError('Expected positive counts must cover exactly six regions')
    case_rows, statistical_rows, counts = [], [], {}
    for case_id in sorted(e1a_ids):
        for region in REGIONS:
            present_a, dice_a = e1a[case_id][region]
            present_b, dice_b = e1b[case_id][region]
            if present_a != present_b:
                raise ValueError(f'Target-presence mismatch: {case_id} {region}')
            delta = dice_b - dice_a if present_a else None
            row = dict(case_id=case_id, region=region, target_present=present_a,
                       e1a_dice=dice_a, e1b_dice=dice_b,
                       delta_e1b_minus_e1a=delta)
            case_rows.append(row)
            if present_a:
                statistical_rows.append(row)
    for region in REGIONS:
        counts[region] = sum(row['target_present'] and row['region'] == region
                             for row in case_rows)
        if counts[region] != expected_positive_counts[region]:
            raise ValueError(f'{region} target-positive count mismatch: observed {counts[region]}, '
                             f'expected {expected_positive_counts[region]}')
    return case_rows, statistical_rows, counts


def statistics(statistical_rows, seed=BOOTSTRAP_SEED,
               resamples=BOOTSTRAP_RESAMPLES, tolerance=TIE_TOLERANCE):
    if resamples < 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Resamples must be positive and tie tolerance finite and nonnegative')
    rows = []
    for region in REGIONS:
        part = [row for row in statistical_rows if row['region'] == region]
        if not part:
            raise ValueError(f'{region}: no target-present paired cases')
        e1a = np.asarray([row['e1a_dice'] for row in part], dtype=float)
        e1b = np.asarray([row['e1b_dice'] for row in part], dtype=float)
        delta = e1b - e1a
        if not np.isfinite(e1a).all() or not np.isfinite(e1b).all():
            raise ValueError(f'{region}: non-finite analyzed Dice')
        lower, upper = bootstrap_ci(delta, seed, resamples)
        p = (1.0 if not np.any(delta)
             else float(wilcoxon(delta, **WILCOXON_PARAMETERS).pvalue))
        if not math.isfinite(p):
            raise ValueError(f'{region}: Wilcoxon returned a non-finite p-value')
        n = len(part)
        e1b_wins = int((delta > tolerance).sum())
        e1a_wins = int((delta < -tolerance).sum())
        rows.append(dict(region=region, positive_case_count=n,
                         e1a_mean_dice=float(e1a.mean()), e1b_mean_dice=float(e1b.mean()),
                         mean_delta_e1b_minus_e1a=float(delta.mean()),
                         median_delta=float(np.median(delta)),
                         bootstrap_mean_delta_ci_lower=lower,
                         bootstrap_mean_delta_ci_upper=upper, wilcoxon_raw_p=p,
                         wilcoxon_holm_adjusted_p=None, e1b_win_count=e1b_wins,
                         tie_count=int((np.abs(delta) <= tolerance).sum()),
                         e1a_win_count=e1a_wins, e1b_win_fraction=e1b_wins / n,
                         e1a_win_fraction=e1a_wins / n))
    adjusted = holm_adjust([row['wilcoxon_raw_p'] for row in rows])
    for row, value in zip(rows, adjusted):
        row['wilcoxon_holm_adjusted_p'] = value
    return rows


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(json.dumps(_json_ready(value), indent=2, allow_nan=False) + '\n')


def analyze(e1a_csv, e1b_csv, output_dir, expected_cases=EXPECTED_CASES,
            expected_positive_counts=EXPECTED_POSITIVE_COUNTS, seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES, tolerance=TIE_TOLERANCE):
    inputs = [Path(path).expanduser().resolve() for path in (e1a_csv, e1b_csv)]
    output = Path(output_dir).expanduser().resolve()
    input_dirs = [path.parent for path in inputs]
    if any(output == directory or output in directory.parents or directory in output.parents
           for directory in input_dirs):
        raise ValueError('Output directory must not overlap frozen input directories')
    if output.is_symlink() or any((output / name).is_symlink() for name in ARTIFACTS):
        raise ValueError('Analysis output paths must not be symlinks')
    if any((output / name).exists() for name in ARTIFACTS):
        raise FileExistsError(f'Analysis artifacts already exist in {output}')
    case_rows, statistical_rows, counts = validate_pair(
        *inputs, expected_cases, expected_positive_counts)
    stats = statistics(statistical_rows, seed, resamples, tolerance)
    settings = {
        'bootstrap_seed': seed, 'bootstrap_resamples': resamples,
        'bootstrap_ci_method': CI_METHOD, 'wilcoxon_parameters': WILCOXON_PARAMETERS,
        'all_zero_wilcoxon_policy': 'p = 1.0 only when all raw paired differences are exactly zero',
        'multiple_comparison_correction': 'Holm step-down across six regional Wilcoxon tests',
        'tie_tolerance': tolerance, 'delta_definition': DELTA_DEFINITION,
    }
    summary = {
        'case_count': expected_cases,
        'paired_case_region_rows': len(case_rows),
        'target_present_paired_rows': len(statistical_rows),
        'region_order': list(REGIONS), 'delta_definition': DELTA_DEFINITION,
        'bootstrap_settings': {key: settings[key] for key in
                               ('bootstrap_seed', 'bootstrap_resamples', 'bootstrap_ci_method')},
        'wilcoxon_settings': WILCOXON_PARAMETERS,
        'holm_correction_scope': settings['multiple_comparison_correction'],
        'tie_tolerance': tolerance, 'target_positive_counts': counts,
        'statistics': stats,
    }
    dirty = _git('status', '--porcelain')
    short = _git('status', '--short')
    metadata = {
        'analysis_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'e1a_input_path': str(inputs[0]), 'e1b_input_path': str(inputs[1]),
        'e1a_input_sha256': hashlib.sha256(inputs[0].read_bytes()).hexdigest(),
        'e1b_input_sha256': hashlib.sha256(inputs[1].read_bytes()).hexdigest(),
        'git_commit_sha': _git('rev-parse', 'HEAD'),
        'git_dirty': None if dirty is None else bool(dirty),
        'git_status_short': None if short is None else short.splitlines(),
        'python_version': platform.python_version(), 'numpy_version': np.__version__,
        'pandas_version': pd.__version__, 'scipy_version': scipy.__version__,
        'statistical_settings': settings,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / 'paired_segmentation_statistics.csv', stats, STAT_FIELDS)
    _write_csv(output / 'paired_case_metrics.csv', case_rows, CASE_FIELDS)
    _write_json(output / 'comparison_summary.json', summary)
    _write_json(output / 'comparison_metadata.json', metadata)
    return output
