"""Frozen-protocol E1-B versus E1-C internal presence-aware Dice comparison.

Run with python -m analysis.e1b_e1c_segmentation_statistics.
The E1-A/E1-B pairing and statistical implementation remains unchanged.
"""

import argparse
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from analysis import e1a_e1b_segmentation_statistics as reference
from analysis.e1a_e1b_segmentation_statistics import (
    ARTIFACTS, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, EXPECTED_CASES,
    EXPECTED_POSITIVE_COUNTS, REGIONS, TIE_TOLERANCE, WILCOXON_PARAMETERS,
    _git, _write_csv, _write_json,
)

DEFAULT_E1B_CSV = 'outputs/frozen/e1b_joint_reconstruction/evaluation/per_case_metrics.csv'
DEFAULT_E1C_CSV = 'outputs/frozen/e1c_anatomy_constraint/evaluation/per_case_metrics.csv'
DEFAULT_OUTPUT = 'outputs/e1b_e1c_segmentation_comparison'
# Source: outputs/frozen/e1a_e1b_segmentation_comparison/comparison_metadata.json,
# e1b_input_sha256 (recorded before defining this constant).
EXPECTED_E1B_SHA256 = 'f80bba75f63d26af9e8f0e8ebe8bbeda1deff75b1d65f5eaf32891805abc4502'
EXPECTED_E1C_SHA256 = '2878b70f1d1a0d366d50388465cc46169eede0654b0f0dd11ba6d1f56210bcdc'
COMPARISON_IDENTITY = 'E1-B vs E1-C'
DELTA_DEFINITION = 'E1-C Dice minus E1-B Dice'
CI_METHOD = 'paired percentile bootstrap of mean E1-C-minus-E1-B Dice; 2.5th and 97.5th percentiles'


def _rename_field(field):
    return field.replace('e1b', '__right__').replace('e1a', 'e1b').replace('__right__', 'e1c')


CASE_FIELDS = tuple(map(_rename_field, reference.CASE_FIELDS))
STAT_FIELDS = tuple(map(_rename_field, reference.STAT_FIELDS))


def _rename_rows(rows):
    return [{_rename_field(key): value for key, value in row.items()} for row in rows]


def validate_pair(e1b_csv, e1c_csv):
    """Use frozen strict loading, identity/presence checks, and deterministic order."""
    try:
        cases, statistical, counts = reference.validate_pair(
            e1b_csv, e1c_csv, EXPECTED_CASES, EXPECTED_POSITIVE_COUNTS)
    except ValueError as error:
        message = str(error).replace('E1-B', '__right__').replace('E1-A', 'E1-B').replace('__right__', 'E1-C')
        raise ValueError(message) from error
    return _rename_rows(cases), _rename_rows(statistical), counts


def statistics(statistical_rows):
    """Map model labels only; all calculations use the unchanged frozen core."""
    rows = [dict(case_id=row['case_id'], region=row['region'],
                 target_present=row['target_present'], e1a_dice=row['e1b_dice'],
                 e1b_dice=row['e1c_dice']) for row in statistical_rows]
    return _rename_rows(reference.statistics(
        rows, BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES, TIE_TOLERANCE))


def analyze(e1b_csv=DEFAULT_E1B_CSV, e1c_csv=DEFAULT_E1C_CSV,
            output_dir=DEFAULT_OUTPUT):
    expected_cases = EXPECTED_CASES
    seed, resamples, tolerance = BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES, TIE_TOLERANCE
    inputs = [Path(path).expanduser().resolve() for path in (e1b_csv, e1c_csv)]
    requested_output = Path(output_dir).expanduser()
    if requested_output.is_symlink():
        raise ValueError('Analysis output paths must not be symlinks')
    output = requested_output.resolve()
    input_dirs = [path.parent for path in inputs]
    if any(output == directory or output in directory.parents or directory in output.parents
           for directory in input_dirs):
        raise ValueError('Output directory must not overlap frozen input directories')
    if output.is_symlink() or any((output / name).is_symlink() for name in ARTIFACTS):
        raise ValueError('Analysis output paths must not be symlinks')
    frozen = Path(__file__).resolve().parents[1] / 'outputs' / 'frozen'
    if output == frozen or frozen in output.parents:
        raise ValueError('Analysis output must not be under outputs/frozen')
    if output.exists():
        raise FileExistsError(f'Analysis artifacts already exist in {output}')
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs]
    for label, observed, expected in zip(('E1-B', 'E1-C'), hashes,
                                       (EXPECTED_E1B_SHA256, EXPECTED_E1C_SHA256)):
        if observed != expected:
            raise ValueError(f'{label} input SHA256 does not match frozen provenance')
    case_rows, statistical_rows, counts = validate_pair(*inputs)
    stats = statistics(statistical_rows)
    settings = {
        'bootstrap_seed': seed, 'bootstrap_resamples': resamples,
        'bootstrap_ci_method': CI_METHOD, 'wilcoxon_parameters': WILCOXON_PARAMETERS,
        'all_zero_wilcoxon_policy': 'p = 1.0 only when all raw paired differences are exactly zero',
        'multiple_comparison_correction': 'Holm step-down across six regional Wilcoxon tests',
        'tie_tolerance': tolerance, 'delta_definition': DELTA_DEFINITION,
    }
    summary = {
        'comparison_identity': COMPARISON_IDENTITY,
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
        'comparison_identity': COMPARISON_IDENTITY,
        'delta_definition': DELTA_DEFINITION,
        'case_identity_validation': {
            'case_count_per_input': expected_cases,
            'unique_case_count_per_input': expected_cases,
            'identical_case_id_sets': True, 'target_presence_agrees': True,
            'paired_case_region_rows': len(case_rows),
            'target_present_paired_rows': len(statistical_rows),
            'sorted_case_ids_sha256': hashlib.sha256(
                ('\n'.join(sorted({row['case_id'] for row in case_rows})) + '\n').encode()).hexdigest(),
        },
        'target_positive_counts': counts,
        'statistical_core': 'analysis.e1a_e1b_segmentation_statistics.statistics',
        'analysis_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'e1b_input_path': str(inputs[0]), 'e1c_input_path': str(inputs[1]),
        'e1b_input_sha256': hashes[0],
        'e1c_input_sha256': hashes[1],
        'git_commit_sha': _git('rev-parse', 'HEAD'),
        'git_dirty': None if dirty is None else bool(dirty),
        'git_status_short': None if short is None else short.splitlines(),
        'python_version': platform.python_version(), 'numpy_version': np.__version__,
        'pandas_version': pd.__version__, 'scipy_version': scipy.__version__,
        'statistical_settings': settings,
    }
    if hashes != [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs]:
        raise ValueError('Input files changed during analysis')
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / 'paired_segmentation_statistics.csv', stats, STAT_FIELDS)
    _write_csv(output / 'paired_case_metrics.csv', case_rows, CASE_FIELDS)
    _write_json(output / 'comparison_summary.json', summary)
    _write_json(output / 'comparison_metadata.json', metadata)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e1b-csv', default=DEFAULT_E1B_CSV)
    parser.add_argument('--e1c-csv', default=DEFAULT_E1C_CSV)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(analyze(**vars(args)))


if __name__ == '__main__':
    main()
