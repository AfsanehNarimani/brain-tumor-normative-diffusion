"""Paired, presence-aware Dice analysis of frozen E1-A latent probes.

This reads evaluation artifacts only. It does not load images or checkpoints.
Positive delta means greater pathology accessibility from the disease
representation (Z_d) than the anatomy/background representation (Z_a).
"""

import csv
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import wilcoxon

from analysis.paired_evaluation_comparison import (
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, REGIONS, WILCOXON_PARAMETERS,
    _git, bootstrap_ci, holm_adjust,
)


EXPECTED_CASES = 270
EXPECTED_POSITIVE_COUNTS = {'NETC': 114, 'SNFH': 270, 'ET': 198,
                            'RC': 230, 'WT': 270, 'TC': 200}
WIN_TIE_TOLERANCE = 1e-8
CI_METHOD = 'paired percentile bootstrap of mean disease-minus-anatomy Dice; 2.5th and 97.5th percentiles'
ARTIFACTS = ('paired_statistics.csv', 'paired_differences.csv',
             'analysis_summary.json', 'analysis_provenance.json', 'paper_table.csv')
STAT_FIELDS = ('region', 'n', 'mean_anatomy_dice', 'mean_disease_dice',
               'mean_delta_disease_minus_anatomy', 'median_delta_disease_minus_anatomy',
               'bootstrap_ci95_low', 'bootstrap_ci95_high', 'wilcoxon_p',
               'holm_adjusted_p', 'disease_wins', 'ties', 'anatomy_wins')
DIFFERENCE_FIELDS = ('case_id', 'region', 'anatomy_dice', 'disease_dice',
                     'delta_disease_minus_anatomy')
PAPER_FIELDS = ('Region', 'N', 'Z_a Dice', 'Z_d Dice', 'Delta Dice', '95% CI',
                'Raw p', 'Holm p', 'Z_d wins', 'Ties', 'Z_a wins')


def _preview(values):
    values = sorted(values)
    return f'{values[:5]} ({len(values)} total)'


def load_cases(path):
    """Read every row strictly; absent targets must have blank Dice as exported."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing frozen per-case metrics: {path}')
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {'case_id', *(f'{region}_{suffix}' for region in REGIONS
                                 for suffix in ('target_present', 'dice'))}
        if len(fields) != len(set(fields)):
            raise ValueError(f'{path}: duplicate CSV column names')
        if required - set(fields):
            raise ValueError(f'{path}: missing required columns: {sorted(required - set(fields))}')
        cases = {}
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'{path}:{line}: malformed CSV row')
            case_id = row['case_id']
            if not case_id or case_id != case_id.strip():
                raise ValueError(f'{path}:{line}: invalid case_id')
            if case_id in cases:
                raise ValueError(f'{path}:{line}: duplicate case_id {case_id}')
            values = {}
            for region in REGIONS:
                flag = row[f'{region}_target_present']
                if flag not in ('0', '1'):
                    raise ValueError(f'{path}:{line}: invalid {region}_target_present: {flag!r}')
                raw_dice = row[f'{region}_dice']
                if flag == '0':
                    if raw_dice != '':
                        raise ValueError(f'{path}:{line}: target-negative {region} Dice must be blank')
                    values[region] = (False, None)
                else:
                    try:
                        dice = float(raw_dice)
                    except ValueError as exc:
                        raise ValueError(f'{path}:{line}: invalid target-positive {region} Dice') from exc
                    if not math.isfinite(dice):
                        raise ValueError(f'{path}:{line}: non-finite target-positive {region} Dice')
                    values[region] = (True, dice)
            cases[case_id] = values
    if not cases:
        raise ValueError(f'{path}: no case rows')
    return cases


def validate_pair(anatomy_csv, disease_csv, expected_cases=EXPECTED_CASES,
                  expected_positive_counts=EXPECTED_POSITIVE_COUNTS):
    anatomy, disease = load_cases(anatomy_csv), load_cases(disease_csv)
    if len(anatomy) != expected_cases or len(disease) != expected_cases:
        raise ValueError(f'Expected {expected_cases} cases in each CSV; anatomy={len(anatomy)}, disease={len(disease)}')
    anatomy_ids, disease_ids = set(anatomy), set(disease)
    if anatomy_ids != disease_ids:
        raise ValueError('Case set mismatch: anatomy-only=' + _preview(anatomy_ids - disease_ids)
                         + ', disease-only=' + _preview(disease_ids - anatomy_ids))
    if set(expected_positive_counts) != set(REGIONS):
        raise ValueError('Expected positive counts must cover exactly six regions')
    paired = []
    counts = {}
    for region in REGIONS:
        count = 0
        for case_id in sorted(anatomy_ids):
            present_a, dice_a = anatomy[case_id][region]
            present_d, dice_d = disease[case_id][region]
            if present_a != present_d:
                raise ValueError(f'Target-presence mismatch: {case_id} {region}')
            if present_a:
                count += 1
                paired.append(dict(case_id=case_id, region=region, anatomy_dice=dice_a,
                                   disease_dice=dice_d, delta_disease_minus_anatomy=dice_d - dice_a))
        counts[region] = count
        if count != expected_positive_counts[region]:
            raise ValueError(f'{region} target-positive count mismatch: observed {count}, '
                             f'expected {expected_positive_counts[region]}')
    return paired, counts, len(anatomy_ids)


def statistics(paired, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES,
               tolerance=WIN_TIE_TOLERANCE):
    if resamples < 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Resamples must be positive and win/tie tolerance finite and nonnegative')
    rows = []
    for region in REGIONS:
        part = [row for row in paired if row['region'] == region]
        if not part:
            raise ValueError(f'{region}: no target-positive paired cases')
        anatomy = np.array([row['anatomy_dice'] for row in part], dtype=float)
        disease = np.array([row['disease_dice'] for row in part], dtype=float)
        delta = disease - anatomy
        if not np.isfinite(anatomy).all() or not np.isfinite(disease).all():
            raise ValueError(f'{region}: non-finite analyzed Dice')
        ci_low, ci_high = bootstrap_ci(delta, seed, resamples)
        # scipy's wilcox zero method cannot test a vector of all zero deltas.
        p = 1.0 if not np.any(delta) else float(wilcoxon(delta, **WILCOXON_PARAMETERS).pvalue)
        if not math.isfinite(p):
            raise ValueError(f'{region}: Wilcoxon returned a non-finite p-value')
        rows.append(dict(region=region, n=len(part), mean_anatomy_dice=float(anatomy.mean()),
                         mean_disease_dice=float(disease.mean()),
                         mean_delta_disease_minus_anatomy=float(delta.mean()),
                         median_delta_disease_minus_anatomy=float(np.median(delta)),
                         bootstrap_ci95_low=ci_low, bootstrap_ci95_high=ci_high,
                         wilcoxon_p=p, holm_adjusted_p=None,
                         disease_wins=int((delta > tolerance).sum()),
                         ties=int((np.abs(delta) <= tolerance).sum()),
                         anatomy_wins=int((delta < -tolerance).sum())))
    adjusted = holm_adjust([row['wilcoxon_p'] for row in rows])
    for row, value in zip(rows, adjusted):
        row['holm_adjusted_p'] = value
    return rows


def _write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def paper_table(rows):
    """Four decimals for Dice/CI; three significant digits for p-values."""
    return [{
        'Region': row['region'], 'N': row['n'],
        'Z_a Dice': f"{row['mean_anatomy_dice']:.4f}",
        'Z_d Dice': f"{row['mean_disease_dice']:.4f}",
        'Delta Dice': f"{row['mean_delta_disease_minus_anatomy']:+.4f}",
        '95% CI': (f"[{row['bootstrap_ci95_low']:+.4f}, "
                   f"{row['bootstrap_ci95_high']:+.4f}]"),
        'Raw p': f"{row['wilcoxon_p']:.3g}",
        'Holm p': f"{row['holm_adjusted_p']:.3g}",
        'Z_d wins': row['disease_wins'], 'Ties': row['ties'],
        'Z_a wins': row['anatomy_wins'],
    } for row in rows]


def analyze(anatomy_csv, disease_csv, output_dir, expected_cases=EXPECTED_CASES,
            expected_positive_counts=EXPECTED_POSITIVE_COUNTS, seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES, tolerance=WIN_TIE_TOLERANCE):
    inputs = [Path(path).expanduser().resolve() for path in (anatomy_csv, disease_csv)]
    output = Path(output_dir).expanduser().resolve()
    frozen = Path('outputs/frozen').resolve()
    if output == frozen or frozen in output.parents or any(output == path.parent or output in path.parents
                                                            for path in inputs):
        raise ValueError('Analysis output must be separate from frozen inputs')
    paired, counts, case_count = validate_pair(*inputs, expected_cases, expected_positive_counts)
    rows = statistics(paired, seed, resamples, tolerance)
    if any((output / name).exists() for name in ARTIFACTS):
        raise FileExistsError(f'Analysis artifacts already exist in {output}')
    if output.is_symlink():
        raise ValueError('Analysis output must not be a symlink')
    dirty = _git('status', '--porcelain')
    provenance = {
        'analysis_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'anatomy_input': str(inputs[0]), 'disease_input': str(inputs[1]),
        'anatomy_input_sha256': hashlib.sha256(inputs[0].read_bytes()).hexdigest(),
        'disease_input_sha256': hashlib.sha256(inputs[1].read_bytes()).hexdigest(),
        'case_count': case_count, 'anatomy_case_count': case_count,
        'disease_case_count': case_count, 'target_positive_counts': counts,
        'expected_positive_counts': dict(expected_positive_counts),
        'bootstrap_seed': seed, 'bootstrap_resamples': resamples,
        'bootstrap_ci_method': CI_METHOD,
        'wilcoxon_parameters': WILCOXON_PARAMETERS,
        'all_zero_wilcoxon_policy': 'p = 1.0 when all paired differences are zero',
        'multiple_comparison_correction': 'Holm step-down across six regional Wilcoxon p-values',
        'win_tie_tolerance': tolerance,
        'paper_table_format': 'Dice, delta, CI: four decimals; p-values: three significant digits; source CSV/JSON retain full precision.',
        'git_commit_sha': _git('rev-parse', 'HEAD'),
        'git_dirty': None if dirty is None else bool(dirty),
        'python_version': platform.python_version(), 'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'interpretation': ('Relative pathology accessibility under identical probes; '
                           'does not establish disentanglement.'),
    }
    summary = {'case_count': case_count, 'regions': list(REGIONS),
               'delta_definition': 'disease representation (Z_d) minus anatomy/background representation (Z_a)',
               'target_positive_counts': counts, 'statistics': rows,
               'paired_difference_rows': len(paired),
               'interpretation': 'Descriptive paired comparison; scientific interpretation awaits inspection of results.'}
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / 'paired_statistics.csv', rows, STAT_FIELDS)
    _write_csv(output / 'paired_differences.csv', paired, DIFFERENCE_FIELDS)
    _write_json(output / 'analysis_summary.json', summary)
    _write_json(output / 'analysis_provenance.json', provenance)
    _write_csv(output / 'paper_table.csv', paper_table(rows), PAPER_FIELDS)
    return output
