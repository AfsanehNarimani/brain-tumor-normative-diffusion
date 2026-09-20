"""Synthetic tests for paired E1-A versus E1-B internal segmentation analysis."""

import csv
import json
from types import SimpleNamespace

import pytest

from analysis import e1a_e1b_segmentation_statistics as analysis


def make_pair(tmp_path, cases=3, positive_counts=None):
    positive_counts = positive_counts or {region: cases for region in analysis.REGIONS}
    fields = list(analysis._required_fields())
    e1a, e1b = [], []
    for index in range(cases):
        left, right = {'case_id': f'case-{index:03d}'}, {'case_id': f'case-{index:03d}'}
        for region in analysis.REGIONS:
            present = index < positive_counts[region]
            left[f'{region}_target_present'] = right[f'{region}_target_present'] = str(int(present))
            left[f'{region}_dice'] = right[f'{region}_dice'] = '0.5' if present else ''
        e1a.append(left)
        e1b.append(right)
    input_a, input_b = tmp_path / 'e1a', tmp_path / 'e1b'
    input_a.mkdir()
    input_b.mkdir()
    paths = input_a / 'per_case_metrics.csv', input_b / 'per_case_metrics.csv'

    def write(extra_fields=()):
        output_fields = fields + list(extra_fields)
        for path, rows in zip(paths, (e1a, e1b)):
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=output_fields)
                writer.writeheader()
                writer.writerows(rows)

    write()
    return e1a, e1b, paths, write, positive_counts


def validate(paths, counts, cases=3):
    return analysis.validate_pair(*paths, expected_cases=cases,
                                  expected_positive_counts=counts)


def test_pairing_delta_direction_and_absent_rows(tmp_path):
    e1a, e1b, paths, write, counts = make_pair(tmp_path)
    e1a[0]['NETC_dice'], e1b[0]['NETC_dice'] = '0.2', '0.8'
    for rows in (e1a, e1b):
        rows[2]['NETC_target_present'], rows[2]['NETC_dice'] = '0', ''
    counts['NETC'] = 2
    e1b.reverse()
    write()
    cases, statistical, observed = validate(paths, counts)
    row = next(row for row in statistical if row['case_id'] == 'case-000' and row['region'] == 'NETC')
    assert row['delta_e1b_minus_e1a'] == pytest.approx(0.6)
    absent = next(row for row in cases if row['case_id'] == 'case-002' and row['region'] == 'NETC')
    assert absent['target_present'] is False
    assert absent['e1a_dice'] is absent['e1b_dice'] is absent['delta_e1b_minus_e1a'] is None
    assert len(cases) == 18 and len(statistical) == 17 and observed['NETC'] == 2


@pytest.mark.parametrize(('change', 'message'), [
    (lambda a, b: a[1].__setitem__('case_id', 'case-000'), 'duplicate case_id'),
    (lambda a, b: b[0].__setitem__('case_id', 'unexpected'), 'Case set mismatch'),
    (lambda a, b: b[0].update(NETC_target_present='0', NETC_dice=''), 'Target-presence mismatch'),
    (lambda a, b: a[0].__setitem__('NETC_dice', ''), 'Dice is missing'),
    (lambda a, b: a[0].__setitem__('NETC_dice', 'nan'), 'non-finite'),
])
def test_strict_pair_rejections(tmp_path, change, message):
    e1a, e1b, paths, write, counts = make_pair(tmp_path)
    change(e1a, e1b)
    write()
    with pytest.raises(ValueError, match=message):
        validate(paths, counts)


def test_target_absent_nonempty_dice_and_presence_count_rejected(tmp_path):
    e1a, e1b, paths, write, counts = make_pair(tmp_path)
    for rows in (e1a, e1b):
        rows[0]['NETC_target_present'] = '0'
        rows[0]['NETC_dice'] = '0.5'
    write()
    with pytest.raises(ValueError, match='target-absent NETC Dice must be blank'):
        validate(paths, counts)
    for rows in (e1a, e1b):
        rows[0]['NETC_dice'] = ''
    write()
    with pytest.raises(ValueError, match='NETC target-positive count mismatch'):
        validate(paths, counts)


def statistical_rows(deltas):
    return [dict(case_id=f'case-{index}', region=region, target_present=True,
                 e1a_dice=0.5, e1b_dice=0.5 + delta,
                 delta_e1b_minus_e1a=delta)
            for region in analysis.REGIONS for index, delta in enumerate(deltas)]


def test_tolerance_zero_policy_bootstrap_and_holm(monkeypatch):
    rows = statistical_rows((2e-8, -2e-8, 5e-9))
    calls = []
    pvalues = iter((0.01, 0.04, 0.03, 0.20, 0.50, 0.90))
    def fake_wilcoxon(delta, **options):
        calls.append(options)
        return SimpleNamespace(pvalue=next(pvalues))
    monkeypatch.setattr(analysis, 'wilcoxon', fake_wilcoxon)
    first = analysis.statistics(rows, seed=7, resamples=20)
    assert first[0]['e1b_win_count'] == first[0]['tie_count'] == first[0]['e1a_win_count'] == 1
    assert [row['wilcoxon_holm_adjusted_p'] for row in first] == pytest.approx(
        [0.06, 0.16, 0.15, 0.6, 1.0, 1.0])
    delta = [2e-8, -2e-8, 5e-9]
    assert analysis.bootstrap_ci(delta, 7, 20) == analysis.bootstrap_ci(delta, 7, 20)
    assert all(options == analysis.WILCOXON_PARAMETERS for options in calls)

    tiny_calls = []
    def tiny_wilcoxon(delta, **options):
        tiny_calls.append(delta.copy())
        return SimpleNamespace(pvalue=0.25)
    monkeypatch.setattr(analysis, 'wilcoxon', tiny_wilcoxon)
    tiny = analysis.statistics(statistical_rows((5e-9, -5e-9, 0.0)), resamples=10)
    assert len(tiny_calls) == len(analysis.REGIONS)
    assert all(row['wilcoxon_raw_p'] == 0.25 and row['tie_count'] == 3 for row in tiny)

    monkeypatch.setattr(analysis, 'wilcoxon', lambda *_a, **_k: pytest.fail('Wilcoxon called'))
    exact_zeros = analysis.statistics(statistical_rows((0.0, 0.0, 0.0)), resamples=10)
    assert all(row['wilcoxon_raw_p'] == 1.0 for row in exact_zeros)


def test_full_expected_artifacts_counts_and_overwrite_protection(tmp_path):
    e1a, e1b, paths, write, counts = make_pair(
        tmp_path, cases=270, positive_counts=dict(analysis.EXPECTED_POSITIVE_COUNTS))
    e1b[0]['NETC_dice'] = '0.6'
    write()
    output = analysis.analyze(*paths, tmp_path / 'analysis', resamples=20)
    assert {path.name for path in output.iterdir()} == set(analysis.ARTIFACTS)
    with (output / 'paired_case_metrics.csv').open(newline='') as stream:
        case_rows = list(csv.DictReader(stream))
    with (output / 'paired_segmentation_statistics.csv').open(newline='') as stream:
        stats = list(csv.DictReader(stream))
    assert len(case_rows) == 1620 and len(stats) == 6
    assert sum(row['target_present'] == 'True' for row in case_rows) == 1282
    summary = json.loads((output / 'comparison_summary.json').read_text())
    assert summary['case_count'] == 270
    assert summary['paired_case_region_rows'] == 1620
    assert summary['target_present_paired_rows'] == 1282
    assert summary['region_order'] == list(analysis.REGIONS)
    metadata = json.loads((output / 'comparison_metadata.json').read_text())
    assert len(metadata['e1a_input_sha256']) == len(metadata['e1b_input_sha256']) == 64
    assert metadata['statistical_settings']['bootstrap_seed'] == analysis.BOOTSTRAP_SEED
    with pytest.raises(FileExistsError, match='already exist'):
        analysis.analyze(*paths, output, resamples=20)
