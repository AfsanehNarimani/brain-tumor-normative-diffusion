"""Paired case-level comparison of frozen D0-B3 corruption operators."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np
import scipy
from scipy.stats import wilcoxon

EXPECTED_CASES = 200
OPERATORS = ('zero_fill', 'local_mean_fill', 'local_empirical_resample')
PAIRS = (('zero_fill', 'local_mean_fill'),
         ('zero_fill', 'local_empirical_resample'),
         ('local_mean_fill', 'local_empirical_resample'))
PRIMARY_METRIC = 'primary_delta_std_region_c_mean'
INPUT_CASE_SHA256 = '86fa2de9f033e82df566c0810b77e143dc1c2a080ba970e158e3fc22effcd0a6'
SUMMARY_SHA256 = '077a97bdc7c2726c9837158436e55c842955f72361d83b893d65acfe5b0aca47'
PROVENANCE_SHA256 = '0c4dc30064fca2985ffbce1d30ec41dbb87fdc412e4357b731089300332ba69e'
DEFAULT_CASE_CSV = 'outputs/d0b3_corruption_operator_audit/case_corruption_statistics.csv'
DEFAULT_SUMMARY = 'outputs/d0b3_corruption_operator_audit/corruption_operator_summary.json'
DEFAULT_PROVENANCE = 'outputs/d0b3_corruption_operator_audit/analysis_provenance.json'
DEFAULT_OUTPUT = 'outputs/d0b3_corruption_operator_paired_analysis'
BOOTSTRAP_SEED = 20260923
BOOTSTRAP_RESAMPLES = 10000
TIE_TOLERANCE = 1e-8
ALPHA = 0.05
WILCOXON_PARAMETERS = {'zero_method': 'wilcox', 'correction': False,
                       'alternative': 'two-sided', 'method': 'auto'}
REQUIRED_COLUMNS = {'case_id', 'operator', PRIMARY_METRIC}
OUTPUTS = ('pairwise_statistics.csv', 'paired_case_deltas.csv',
           'paired_analysis_summary.json', 'analysis_provenance.json')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_case_operator_rows(path, expected_cases=EXPECTED_CASES):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing D0-B3 case statistics: {path}')
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f'Missing required columns: {sorted(missing)}')
        rows = list(reader)
    if len(rows) != expected_cases * len(OPERATORS):
        raise ValueError(f'Expected {expected_cases * len(OPERATORS)} case/operator rows; got {len(rows)}')
    parsed = []
    seen = set()
    for row in rows:
        case = row['case_id']
        operator = row['operator']
        if not case or operator not in OPERATORS:
            raise ValueError('Invalid case_id or operator')
        key = (case, operator)
        if key in seen:
            raise ValueError(f'Duplicate case/operator row: {key}')
        seen.add(key)
        raw = row[PRIMARY_METRIC]
        value = None if raw in ('', 'None', 'null') else float(raw)
        if value is not None and not math.isfinite(value):
            raise ValueError(f'Non-finite primary metric for {key}')
        parsed.append({'case_id': case, 'operator': operator, PRIMARY_METRIC: value})
    validate_operator_completeness(parsed, expected_cases)
    return parsed


def validate_operator_completeness(rows, expected_cases=EXPECTED_CASES):
    cases = {row['case_id'] for row in rows}
    if len(cases) != expected_cases:
        raise ValueError(f'Expected {expected_cases} unique cases; got {len(cases)}')
    for case in cases:
        present = {row['operator'] for row in rows if row['case_id'] == case}
        if present != set(OPERATORS):
            raise ValueError(f'Case {case} does not contain all three operators')


def paired_values(rows, operator_a, operator_b):
    values = {(row['case_id'], row['operator']): row[PRIMARY_METRIC] for row in rows}
    paired = []
    for case in sorted({row['case_id'] for row in rows}):
        a, b = values[(case, operator_a)], values[(case, operator_b)]
        if a is None or b is None:
            continue
        paired.append((case, float(a), float(b), float(b - a)))
    return paired


def bootstrap_ci(delta, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    delta = np.asarray(delta, dtype=np.float64)
    if delta.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 512):
        stop = min(start + 512, resamples)
        indices = rng.integers(0, delta.size, size=(stop - start, delta.size))
        means[start:stop] = delta[indices].mean(axis=1)
    return tuple(float(value) for value in np.percentile(means, (2.5, 97.5)))


def classify_delta(delta, tolerance=TIE_TOLERANCE):
    if abs(delta) <= tolerance:
        return 'tie'
    return 'b_lower' if delta < -tolerance else 'a_lower'


def holm_adjust(pvalues):
    adjusted = [None] * len(pvalues)
    ordered = sorted(enumerate(pvalues), key=lambda item: item[1])
    running = 0.0
    for rank, (index, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalue))
        adjusted[index] = running
    return adjusted


def reject_after_holm(adjusted_p, alpha=ALPHA):
    return bool(adjusted_p <= alpha)


def pair_statistics(paired, operator_a, operator_b, seed=BOOTSTRAP_SEED,
                    resamples=BOOTSTRAP_RESAMPLES, tolerance=TIE_TOLERANCE):
    if not paired:
        raise ValueError('No defined paired primary metrics')
    a = np.asarray([item[1] for item in paired], dtype=np.float64)
    b = np.asarray([item[2] for item in paired], dtype=np.float64)
    delta = b - a
    low, high = bootstrap_ci(delta, seed, resamples)
    pvalue = 1.0 if np.all(np.abs(delta) <= tolerance) else float(wilcoxon(delta, **WILCOXON_PARAMETERS).pvalue)
    classes = [classify_delta(value, tolerance) for value in delta]
    return {'operator_a': operator_a, 'operator_b': operator_b,
            'paired_case_count': int(delta.size), 'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
            'mean_delta_b_minus_a': float(delta.mean()), 'median_delta_b_minus_a': float(np.median(delta)),
            'std_delta_b_minus_a': float(delta.std(ddof=0)), 'min_delta_b_minus_a': float(delta.min()),
            'max_delta_b_minus_a': float(delta.max()), 'bootstrap_ci95_low': low,
            'bootstrap_ci95_high': high, 'wilcoxon_raw_p': pvalue,
            'wilcoxon_holm_p': None, 'reject_after_holm': None,
            'a_lower_count': classes.count('a_lower'), 'b_lower_count': classes.count('b_lower'),
            'tie_count': classes.count('tie'), 'a_lower_fraction': classes.count('a_lower') / delta.size,
            'b_lower_fraction': classes.count('b_lower') / delta.size, 'tie_fraction': classes.count('tie') / delta.size}


def paired_delta_rows(paired, operator_a, operator_b, tolerance=TIE_TOLERANCE):
    return [{'case_id': case, 'operator_a': operator_a, 'operator_b': operator_b,
             'metric_a': a, 'metric_b': b, 'delta_b_minus_a': delta,
             'comparison_class': classify_delta(delta, tolerance)}
            for case, a, b, delta in paired]


def _git(*args):
    return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()


def analyze(case_csv=DEFAULT_CASE_CSV, summary_path=DEFAULT_SUMMARY,
            provenance_path=DEFAULT_PROVENANCE, output_dir=DEFAULT_OUTPUT,
            expected_case_sha256=None, expected_summary_sha256=None,
            expected_provenance_sha256=None, seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES, tolerance=TIE_TOLERANCE):
    if seed != BOOTSTRAP_SEED or resamples != BOOTSTRAP_RESAMPLES or tolerance != TIE_TOLERANCE:
        raise ValueError('D0-B3 paired protocol constants are locked')
    expected_case_sha256 = INPUT_CASE_SHA256 if expected_case_sha256 is None else expected_case_sha256
    expected_summary_sha256 = SUMMARY_SHA256 if expected_summary_sha256 is None else expected_summary_sha256
    expected_provenance_sha256 = PROVENANCE_SHA256 if expected_provenance_sha256 is None else expected_provenance_sha256
    input_path = Path(case_csv).expanduser().resolve()
    if file_sha256(input_path) != expected_case_sha256:
        raise ValueError('D0-B3 case statistics SHA256 mismatch')
    auxiliary = []
    for path, expected in ((summary_path, expected_summary_sha256), (provenance_path, expected_provenance_sha256)):
        if path is not None:
            candidate = Path(path).expanduser().resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f'Missing D0-B3 auxiliary artifact: {candidate}')
            if expected is not None and file_sha256(candidate) != expected:
                raise ValueError(f'D0-B3 auxiliary artifact SHA256 mismatch: {candidate.name}')
            auxiliary.append((candidate, expected))
    output = Path(output_dir).expanduser().resolve()
    frozen = (Path(__file__).resolve().parents[1] / 'outputs/frozen').resolve()
    if output == frozen or frozen in output.parents:
        raise ValueError('Output must not be under outputs/frozen')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is non-empty: {output}')
    rows = load_case_operator_rows(input_path)
    results, differences = [], []
    for operator_a, operator_b in PAIRS:
        paired = paired_values(rows, operator_a, operator_b)
        results.append(pair_statistics(paired, operator_a, operator_b, seed, resamples, tolerance))
        differences.extend(paired_delta_rows(paired, operator_a, operator_b, tolerance))
    adjusted = holm_adjust([row['wilcoxon_raw_p'] for row in results])
    for row, adjusted_p in zip(results, adjusted):
        row['wilcoxon_holm_p'] = adjusted_p
        row['reject_after_holm'] = reject_after_holm(adjusted_p)
    status = _git('status', '--short').splitlines()
    provenance = {
        'schema': 'd0b3_corruption_operator_paired_analysis', 'schema_version': 1,
        'analysis_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'runtime_git_branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'runtime_git_commit': _git('rev-parse', 'HEAD'), 'git_dirty': bool(status), 'git_status_short': status,
        'case_statistics_path': str(input_path), 'case_statistics_sha256': file_sha256(input_path),
        'summary_path': str(Path(summary_path).resolve()) if summary_path else None,
        'summary_sha256': file_sha256(summary_path) if summary_path else None,
        'source_provenance_path': str(Path(provenance_path).resolve()) if provenance_path else None,
        'source_provenance_sha256': file_sha256(provenance_path) if provenance_path else None,
        'primary_metric': PRIMARY_METRIC,
        'primary_metric_interpretation': 'case-level weighted standardized Za absolute difference over REGION_C, restricted to abnormal-positive patches',
        'statistical_unit': 'case', 'case_count': EXPECTED_CASES, 'operators': list(OPERATORS),
        'bootstrap_seed': seed, 'bootstrap_resamples': resamples, 'bootstrap_ci_method': 'percentile 2.5th and 97.5th percentiles',
        'wilcoxon_parameters': WILCOXON_PARAMETERS, 'holm_correction': 'Holm step-down across exactly 3 pairwise Wilcoxon p-values',
        'alpha': ALPHA, 'tie_tolerance': tolerance, 'validation_used': False, 'model_training': False,
        'diffusion_loaded': False, 'final_operator_selected': False, 'python_version': platform.python_version(),
        'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        'interpretation': 'Lower primary_delta_std_region_c_mean indicates lower collateral standardized Za shift outside the local Conv/Pool geometric receptive field of annotated abnormal tissue. REGION_C is not interpreted as guaranteed pathology-independent because GroupNorm introduces wider spatial statistical coupling.',
    }
    summary = {'case_count': EXPECTED_CASES, 'operators': list(OPERATORS), 'pair_definitions': [dict(operator_a=a, operator_b=b, delta='B - A') for a, b in PAIRS],
               'primary_metric': PRIMARY_METRIC, 'statistical_unit': 'case', 'paired_defined_case_counts': {f'{a}_vs_{b}': row['paired_case_count'] for (a, b), row in zip(PAIRS, results)},
               'bootstrap_seed': seed, 'bootstrap_resamples': resamples, 'wilcoxon_parameters': WILCOXON_PARAMETERS,
               'bootstrap_ci_method': 'percentile 2.5th and 97.5th percentiles', 'alpha': ALPHA,
               'holm_correction': 'Holm across exactly 3 comparisons', 'tie_tolerance': tolerance,
               'results': results, 'no_operator_selected': True, 'methodological_interpretation': provenance['interpretation']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.d0b3-paired-', dir=output.parent) as temporary:
        stage = Path(temporary)
        fields = list(results[0])
        with (stage / 'pairwise_statistics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(results)
        difference_fields = ['case_id', 'operator_a', 'operator_b', 'metric_a', 'metric_b', 'delta_b_minus_a', 'comparison_class']
        with (stage / 'paired_case_deltas.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=difference_fields); writer.writeheader(); writer.writerows(differences)
        (stage / 'paired_analysis_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        (stage / 'analysis_provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
        output.mkdir(exist_ok=True)
        for name in OUTPUTS:
            (stage / name).replace(output / name)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-statistics', default=DEFAULT_CASE_CSV)
    parser.add_argument('--summary', default=DEFAULT_SUMMARY)
    parser.add_argument('--provenance', default=DEFAULT_PROVENANCE)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(analyze(args.case_statistics, args.summary, args.provenance, args.output_dir))


if __name__ == '__main__':
    main()
