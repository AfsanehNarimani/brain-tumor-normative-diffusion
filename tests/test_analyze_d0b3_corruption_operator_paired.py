import csv
import hashlib
import json

import numpy as np
import pytest

from scripts import analyze_d0b3_corruption_operator_paired as analysis


def write_cases(path, values=None, duplicate=False, missing=False):
    values = values or {'zero_fill': 0.2, 'local_mean_fill': 0.1, 'local_empirical_resample': 0.15}
    cases = [f'case-{index:03d}' for index in range(analysis.EXPECTED_CASES)]
    rows = [{'case_id': case, 'operator': operator, analysis.PRIMARY_METRIC: values[operator] + index / 10000}
            for index, case in enumerate(cases) for operator in analysis.OPERATORS]
    if duplicate:
        rows[-1] = dict(rows[0])
    if missing:
        rows[-1] = dict(rows[-2])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'operator', analysis.PRIMARY_METRIC])
        writer.writeheader(); writer.writerows(rows)


def configured_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_sha_guard(tmp_path, monkeypatch):
    path = tmp_path / 'cases.csv'; write_cases(path)
    monkeypatch.setattr(analysis, 'INPUT_CASE_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        analysis.analyze(case_csv=path, summary_path=None, provenance_path=None, output_dir=tmp_path / 'out')


def test_pairing_integrity_duplicate_and_missing_operator_fail(tmp_path):
    duplicate = tmp_path / 'duplicate.csv'; write_cases(duplicate, duplicate=True)
    with pytest.raises(ValueError, match='Duplicate case/operator row'):
        analysis.load_case_operator_rows(duplicate)
    missing = tmp_path / 'missing.csv'; write_cases(missing, missing=True)
    with pytest.raises(ValueError, match='Duplicate case/operator row'):
        analysis.load_case_operator_rows(missing)


def test_missing_operator_set_check_is_explicit():
    rows = [{'case_id': 'case', 'operator': 'zero_fill', analysis.PRIMARY_METRIC: 1.0},
            {'case_id': 'case', 'operator': 'zero_fill', analysis.PRIMARY_METRIC: 1.0},
            {'case_id': 'case', 'operator': 'local_mean_fill', analysis.PRIMARY_METRIC: 1.0}]
    with pytest.raises(ValueError, match='does not contain all three operators'):
        analysis.validate_operator_completeness(rows, expected_cases=1)


def test_pairing_complete_and_delta_is_b_minus_a(tmp_path):
    path = tmp_path / 'cases.csv'; write_cases(path)
    rows = analysis.load_case_operator_rows(path)
    paired = analysis.paired_values(rows, 'zero_fill', 'local_mean_fill')
    assert len(paired) == 200
    assert paired[0][3] == pytest.approx(paired[0][2] - paired[0][1])


def test_tie_tolerance_and_counts():
    assert analysis.classify_delta(1e-8) == 'tie'
    assert analysis.classify_delta(-1e-8) == 'tie'
    assert analysis.classify_delta(-2e-8) == 'b_lower'
    paired = [('a', 1.0, 0.0, -1.0), ('b', 0.0, 1.0, 1.0), ('c', 1.0, 1.0, 0.0)]
    row = analysis.pair_statistics(paired, 'a', 'b', resamples=10_000)
    assert (row['b_lower_count'], row['a_lower_count'], row['tie_count']) == (1, 1, 1)


def test_bootstrap_is_deterministic_and_uses_paired_deltas():
    delta = np.array([-1.0, 2.0, 4.0])
    assert analysis.bootstrap_ci(delta, seed=20260923, resamples=1000) == analysis.bootstrap_ci(delta, seed=20260923, resamples=1000)


def test_all_zero_deltas_have_explicit_p_one():
    paired = [('a', 1.0, 1.0, 0.0)] * 5
    assert analysis.pair_statistics(paired, 'a', 'b', resamples=10_000)['wilcoxon_raw_p'] == 1.0


def test_holm_adjustment_known_values():
    assert analysis.holm_adjust([0.01, 0.04, 0.2]) == pytest.approx([0.03, 0.08, 0.2])


def test_undefined_primary_values_are_excluded_pairwise(tmp_path):
    path = tmp_path / 'cases.csv'; write_cases(path)
    rows = []
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row['case_id'] == 'case-000' and row['operator'] == 'local_mean_fill':
            row[analysis.PRIMARY_METRIC] = ''
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'operator', analysis.PRIMARY_METRIC]); writer.writeheader(); writer.writerows(rows)
    parsed = analysis.load_case_operator_rows(path)
    assert len(analysis.paired_values(parsed, 'zero_fill', 'local_mean_fill')) == 199


def test_outputs_schema_provenance_and_no_winner(tmp_path, monkeypatch):
    path = tmp_path / 'cases.csv'; write_cases(path)
    expected = configured_hash(path)
    monkeypatch.setattr(analysis, 'INPUT_CASE_SHA256', expected)
    output = analysis.analyze(case_csv=path, summary_path=None, provenance_path=None, output_dir=tmp_path / 'out')
    assert len(list(csv.DictReader((output / 'pairwise_statistics.csv').open(newline='')))) == 3
    assert len(list(csv.DictReader((output / 'paired_case_deltas.csv').open(newline='')))) == 600
    summary = json.loads((output / 'paired_analysis_summary.json').read_text())
    provenance = json.loads((output / 'analysis_provenance.json').read_text())
    assert summary['no_operator_selected'] is True and 'best_operator' not in summary
    assert provenance['bootstrap_seed'] == 20260923
    assert provenance['bootstrap_resamples'] == 10000
    assert provenance['tie_tolerance'] == 1e-8
    assert provenance['primary_metric'] == analysis.PRIMARY_METRIC
    assert summary['alpha'] == 0.05
    assert summary['bootstrap_ci_method'] == 'percentile 2.5th and 97.5th percentiles'


def test_holm_boundary_is_rejected_inclusive():
    results = [{'wilcoxon_raw_p': 0.01}, {'wilcoxon_raw_p': 0.02}, {'wilcoxon_raw_p': 0.05}]
    adjusted = analysis.holm_adjust([row['wilcoxon_raw_p'] for row in results])
    assert adjusted[-1] == pytest.approx(0.05)
    assert analysis.reject_after_holm(0.05) is True
