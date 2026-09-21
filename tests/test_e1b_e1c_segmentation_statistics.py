"""Synthetic tests for paired E1-B versus E1-C internal segmentation analysis."""

import csv
import json
import hashlib

import numpy as np
from types import SimpleNamespace

import pytest

from analysis import e1b_e1c_segmentation_statistics as analysis


def make_pair(tmp_path, cases=270, positive_counts=None):
    positive_counts = positive_counts or dict(analysis.EXPECTED_POSITIVE_COUNTS)
    fields = list(analysis.reference._required_fields())
    e1b, e1c = [], []
    for index in range(cases):
        left, right = {'case_id': f'case-{index:03d}'}, {'case_id': f'case-{index:03d}'}
        for region in analysis.REGIONS:
            present = index < positive_counts[region]
            left[f'{region}_target_present'] = right[f'{region}_target_present'] = str(int(present))
            left[f'{region}_dice'] = right[f'{region}_dice'] = '0.5' if present else ''
        e1b.append(left)
        e1c.append(right)
    input_a, input_b = tmp_path / 'e1b', tmp_path / 'e1c'
    input_a.mkdir()
    input_b.mkdir()
    paths = input_a / 'per_case_metrics.csv', input_b / 'per_case_metrics.csv'

    def write(extra_fields=()):
        output_fields = fields + list(extra_fields)
        for path, rows in zip(paths, (e1b, e1c)):
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=output_fields)
                writer.writeheader()
                writer.writerows(rows)

    write()
    return e1b, e1c, paths, write, positive_counts


def validate(paths, counts, cases=270):
    return analysis.validate_pair(*paths)


def test_pairing_delta_direction_and_absent_rows(tmp_path):
    e1b, e1c, paths, write, counts = make_pair(tmp_path)
    e1b[0]['NETC_dice'], e1c[0]['NETC_dice'] = '0.2', '0.8'
    for rows in (e1b, e1c):
        rows[2]['NETC_target_present'], rows[2]['NETC_dice'] = '0', ''
    for rows in (e1b, e1c):
        rows[114]['NETC_target_present'], rows[114]['NETC_dice'] = '1', '0.5'
    e1c.reverse()
    write()
    cases, statistical, observed = validate(paths, counts)
    row = next(row for row in statistical if row['case_id'] == 'case-000' and row['region'] == 'NETC')
    assert row['delta_e1c_minus_e1b'] == pytest.approx(0.6)
    absent = next(row for row in cases if row['case_id'] == 'case-002' and row['region'] == 'NETC')
    assert absent['target_present'] is False
    assert absent['e1b_dice'] is absent['e1c_dice'] is absent['delta_e1c_minus_e1b'] is None
    assert len(cases) == 1620 and len(statistical) == 1282
    assert observed == analysis.EXPECTED_POSITIVE_COUNTS


@pytest.mark.parametrize(('change', 'message'), [
    (lambda a, b: a[1].__setitem__('case_id', 'case-000'), 'duplicate case_id'),
    (lambda a, b: b[0].__setitem__('case_id', 'unexpected'), 'Case set mismatch'),
    (lambda a, b: b[0].update(NETC_target_present='0', NETC_dice=''), 'Target-presence mismatch'),
    (lambda a, b: a[0].__setitem__('NETC_dice', ''), 'Dice is missing'),
    (lambda a, b: a[0].__setitem__('NETC_dice', 'nan'), 'non-finite'),
])
def test_strict_pair_rejections(tmp_path, change, message):
    e1b, e1c, paths, write, counts = make_pair(tmp_path)
    change(e1b, e1c)
    write()
    with pytest.raises(ValueError, match=message):
        validate(paths, counts)


def test_target_absent_nonempty_dice_and_presence_count_rejected(tmp_path):
    e1b, e1c, paths, write, counts = make_pair(tmp_path)
    for rows in (e1b, e1c):
        rows[0]['NETC_target_present'] = '0'
        rows[0]['NETC_dice'] = '0.5'
    write()
    with pytest.raises(ValueError, match='target-absent NETC Dice must be blank'):
        validate(paths, counts)
    for rows in (e1b, e1c):
        rows[0]['NETC_dice'] = ''
    write()
    with pytest.raises(ValueError, match='NETC target-positive count mismatch'):
        validate(paths, counts)


def statistical_rows(deltas):
    return [dict(case_id=f'case-{index}', region=region, target_present=True,
                 e1b_dice=0.5, e1c_dice=0.5 + delta,
                 delta_e1c_minus_e1b=delta)
            for region in analysis.REGIONS for index, delta in enumerate(deltas)]


def test_tolerance_zero_policy_bootstrap_and_holm(monkeypatch):
    rows = statistical_rows((2e-8, -2e-8, 5e-9))
    calls = []
    pvalues = iter((0.01, 0.04, 0.03, 0.20, 0.50, 0.90))
    def fake_wilcoxon(delta, **options):
        calls.append(options)
        return SimpleNamespace(pvalue=next(pvalues))
    monkeypatch.setattr(analysis.reference, 'wilcoxon', fake_wilcoxon)
    first = analysis.statistics(rows)
    assert first[0]['e1c_win_count'] == first[0]['tie_count'] == first[0]['e1b_win_count'] == 1
    assert [row['wilcoxon_holm_adjusted_p'] for row in first] == pytest.approx(
        [0.06, 0.16, 0.15, 0.6, 1.0, 1.0])
    delta = [2e-8, -2e-8, 5e-9]
    assert analysis.reference.bootstrap_ci(delta, 7, 20) == analysis.reference.bootstrap_ci(delta, 7, 20)
    assert all(options == analysis.WILCOXON_PARAMETERS for options in calls)

    tiny_calls = []
    def tiny_wilcoxon(delta, **options):
        tiny_calls.append(delta.copy())
        return SimpleNamespace(pvalue=0.25)
    monkeypatch.setattr(analysis.reference, 'wilcoxon', tiny_wilcoxon)
    tiny = analysis.statistics(statistical_rows((5e-9, -5e-9, 0.0)))
    assert len(tiny_calls) == len(analysis.REGIONS)
    for delta in tiny_calls:
        np.testing.assert_array_equal(delta, np.asarray([0.5 + 5e-9, 0.5 - 5e-9, 0.5]) - 0.5)
    assert all(row['wilcoxon_raw_p'] == 0.25 and row['tie_count'] == 3 for row in tiny)

    monkeypatch.setattr(analysis.reference, 'wilcoxon', lambda *_a, **_k: pytest.fail('Wilcoxon called'))
    exact_zeros = analysis.statistics(statistical_rows((0.0, 0.0, 0.0)))
    assert all(row['wilcoxon_raw_p'] == 1.0 for row in exact_zeros)


def test_full_expected_artifacts_counts_and_overwrite_protection(tmp_path, monkeypatch):
    e1b, e1c, paths, write, counts = make_pair(
        tmp_path, cases=270, positive_counts=dict(analysis.EXPECTED_POSITIVE_COUNTS))
    e1c[0]['NETC_dice'] = '0.6'
    write()
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    monkeypatch.setattr(analysis, 'EXPECTED_E1B_SHA256', hashes[0])
    monkeypatch.setattr(analysis, 'EXPECTED_E1C_SHA256', hashes[1])
    monkeypatch.setattr(analysis, '_git', lambda *args: {
        ('rev-parse', 'HEAD'): 'test-commit', ('status', '--porcelain'): '?? new.py',
        ('status', '--short'): '?? new.py'}[args])
    output = analysis.analyze(*paths, tmp_path / 'analysis')
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
    assert len(metadata['e1b_input_sha256']) == len(metadata['e1c_input_sha256']) == 64
    assert metadata['e1b_input_sha256'] == hashes[0]
    assert metadata['e1c_input_sha256'] == hashes[1]
    assert metadata['e1b_input_path'] == str(paths[0])
    assert metadata['e1c_input_path'] == str(paths[1])
    assert metadata['comparison_identity'] == 'E1-B vs E1-C'
    assert metadata['delta_definition'] == 'E1-C Dice minus E1-B Dice'
    assert metadata['target_positive_counts'] == counts
    assert metadata['case_identity_validation']['target_present_paired_rows'] == 1282
    assert metadata['case_identity_validation']['identical_case_id_sets'] is True
    assert metadata['git_commit_sha'] == 'test-commit'
    assert metadata['git_dirty'] is True and metadata['git_status_short'] == ['?? new.py']
    assert metadata['analysis_timestamp_utc'] and metadata['python_version'] and metadata['scipy_version']
    settings = metadata['statistical_settings']
    assert settings['bootstrap_seed'] == 20260917
    assert settings['bootstrap_resamples'] == 10000
    assert settings['wilcoxon_parameters'] == analysis.WILCOXON_PARAMETERS
    assert settings['tie_tolerance'] == 1e-8
    assert sum(int(row['positive_case_count']) for row in stats) == 1282
    assert all(row['e1b_dice'] == row['e1c_dice'] == row['delta_e1c_minus_e1b'] == ''
               for row in case_rows if row['target_present'] == 'False')
    with pytest.raises(FileExistsError, match='already exist'):
        analysis.analyze(*paths, output)


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('mutation', ['missing', 'extra', 'duplicate'])
def test_exact_270_unique_rows(tmp_path, side, mutation):
    a, b, paths, write, counts = make_pair(tmp_path)
    rows = (a, b)[side]
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'extra':
        rows.append({**rows[0], 'case_id': 'extra-case'})
    else:
        rows[1]['case_id'] = rows[0]['case_id']
    write()
    with pytest.raises(ValueError, match='Expected 270 rows|duplicate case_id'):
        analysis.validate_pair(*paths)


@pytest.mark.parametrize('side', [0, 1])
def test_input_hash_rejection(tmp_path, monkeypatch, side):
    _, _, paths, _, _ = make_pair(tmp_path)
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    hashes[side] = '0' * 64
    monkeypatch.setattr(analysis, 'EXPECTED_E1B_SHA256', hashes[0])
    monkeypatch.setattr(analysis, 'EXPECTED_E1C_SHA256', hashes[1])
    output = tmp_path / 'analysis'
    with pytest.raises(ValueError, match='SHA256'):
        analysis.analyze(*paths, output)
    assert not output.exists()


def test_empty_existing_and_frozen_output_protected(tmp_path):
    _, _, paths, _, _ = make_pair(tmp_path)
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(FileExistsError):
        analysis.analyze(*paths, output)
    with pytest.raises(ValueError, match='outputs/frozen'):
        analysis.analyze(*paths, 'outputs/frozen/forbidden-comparison')


def test_protocol_matches_frozen_core_and_is_reproducible():
    rows = statistical_rows((0.1, -0.2, 0.0, 5e-9))
    first = analysis.statistics(rows)
    assert first == analysis.statistics(rows)
    old_rows = [dict(case_id=r['case_id'], region=r['region'], target_present=True,
                     e1a_dice=r['e1b_dice'], e1b_dice=r['e1c_dice']) for r in rows]
    expected = analysis.reference.statistics(old_rows)
    assert first == analysis._rename_rows(expected)


def test_pinned_hashes_match_recorded_provenance():
    from pathlib import Path
    metadata = json.loads(Path(
        'outputs/frozen/e1a_e1b_segmentation_comparison/comparison_metadata.json').read_text())
    assert analysis.EXPECTED_E1B_SHA256 == metadata['e1b_input_sha256']
    assert analysis.EXPECTED_E1C_SHA256 == '2878b70f1d1a0d366d50388465cc46169eede0654b0f0dd11ba6d1f56210bcdc'


def test_cli_defaults_and_explicit_paths(monkeypatch):
    import sys
    calls = []
    monkeypatch.setattr(analysis, 'analyze', lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, 'argv', ['analysis'])
    analysis.main()
    assert calls[-1] == dict(e1b_csv=analysis.DEFAULT_E1B_CSV,
                             e1c_csv=analysis.DEFAULT_E1C_CSV,
                             output_dir=analysis.DEFAULT_OUTPUT)
    monkeypatch.setattr(sys, 'argv', ['analysis', '--e1b-csv', 'b.csv',
                                     '--e1c-csv', 'c.csv', '--output-dir', 'comparison'])
    analysis.main()
    assert calls[-1] == dict(e1b_csv='b.csv', e1c_csv='c.csv', output_dir='comparison')
