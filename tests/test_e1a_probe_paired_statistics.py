"""Synthetic tests for frozen probe paired Dice analysis; no BraTS data needed."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import e1a_probe_paired_statistics as analysis


def pair(tmp_path, cases=3):
    fields = ['case_id', *(f'{region}_{suffix}' for region in analysis.REGIONS
                           for suffix in ('target_present', 'dice'))]
    anatomy, disease = [], []
    for index in range(cases):
        base = {'case_id': f'case-{index}'}
        for region in analysis.REGIONS:
            base[f'{region}_target_present'] = '1'
            base[f'{region}_dice'] = '0.5'
        anatomy.append(base.copy())
        disease.append(base.copy())
    paths = (tmp_path / 'anatomy.csv', tmp_path / 'disease.csv')

    def write():
        for path, rows in zip(paths, (anatomy, disease)):
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    write()
    return anatomy, disease, paths, write, {region: cases for region in analysis.REGIONS}


def validate(paths, counts, cases=3):
    return analysis.validate_pair(*paths, expected_cases=cases, expected_positive_counts=counts)


def test_pair_by_case_id_delta_sign_and_target_negative_exclusion(tmp_path):
    anatomy, disease, paths, write, counts = pair(tmp_path)
    anatomy[0]['NETC_dice'] = '0.2'
    disease[0]['NETC_dice'] = '0.8'
    anatomy[1]['NETC_dice'] = '0.9'
    disease[1]['NETC_dice'] = '0.1'
    for rows in (anatomy, disease):
        rows[2]['NETC_target_present'] = '0'
        rows[2]['NETC_dice'] = ''
    counts['NETC'] = 2
    disease.reverse()  # Pair by case_id, never by CSV row position.
    write()
    paired, observed, case_count = validate(paths, counts)
    netc = [row for row in paired if row['region'] == 'NETC']
    assert case_count == 3 and observed['NETC'] == 2
    assert [(row['case_id'], row['delta_disease_minus_anatomy']) for row in netc] == [
        ('case-0', pytest.approx(0.6)), ('case-1', pytest.approx(-0.8))]
    assert all(row['case_id'] != 'case-2' for row in netc)
    stats = analysis.statistics(paired, resamples=20)
    assert stats[0]['mean_delta_disease_minus_anatomy'] == pytest.approx(-0.1)


@pytest.mark.parametrize('change, error', [
    (lambda a, d: d[0].__setitem__('case_id', 'other'), 'Case set mismatch'),
    (lambda a, d: a[1].__setitem__('case_id', 'case-0'), 'duplicate case_id'),
    (lambda a, d: d[0].update(NETC_target_present='0', NETC_dice=''), 'Target-presence mismatch'),
    (lambda a, d: a[0].pop('NETC_dice'), 'invalid target-positive NETC Dice'),
    (lambda a, d: d[0].__setitem__('NETC_dice', 'nan'), 'non-finite'),
])
def test_strict_pair_validation(tmp_path, change, error):
    anatomy, disease, paths, write, counts = pair(tmp_path)
    change(anatomy, disease)
    write()
    with pytest.raises(ValueError, match=error):
        validate(paths, counts)


def test_missing_file_and_required_header(tmp_path):
    anatomy, disease, paths, write, counts = pair(tmp_path)
    with pytest.raises(FileNotFoundError):
        validate((paths[0], tmp_path / 'absent.csv'), counts)
    text = paths[0].read_text()
    paths[0].write_text(text.replace('NETC_dice,', 'wrong_name,', 1))
    with pytest.raises(ValueError, match='missing required columns'):
        validate(paths, counts)


def test_expected_positive_count_mismatch(tmp_path):
    _, _, paths, _, counts = pair(tmp_path)
    counts['NETC'] = 2
    with pytest.raises(ValueError, match='NETC target-positive count mismatch'):
        validate(paths, counts)


def test_bootstrap_is_deterministic_and_resamples_paired_deltas():
    delta = np.array([0.2, -0.4, 0.8])
    seed, resamples = 73, 20
    indices = np.random.default_rng(seed).integers(0, len(delta), (resamples, len(delta)))
    expected = tuple(np.percentile(delta[indices].mean(axis=1), [2.5, 97.5]))
    from analysis.paired_evaluation_comparison import bootstrap_ci
    assert bootstrap_ci(delta, seed, resamples) == pytest.approx(expected)
    assert bootstrap_ci(delta, seed, resamples) == bootstrap_ci(delta, seed, resamples)


def test_wilcoxon_zero_policy_options_holm_and_tolerance(monkeypatch):
    paired = []
    for region in analysis.REGIONS:
        for index, delta in enumerate((2e-8, -2e-8, 5e-9)):
            paired.append(dict(case_id=f'case-{index}', region=region, anatomy_dice=0.5,
                               disease_dice=0.5 + delta))
    calls = []
    pvalues = iter((0.01, 0.04, 0.03, 0.20, 0.50, 0.90))

    def fake_wilcoxon(delta, **kwargs):
        calls.append((delta.copy(), kwargs))
        return SimpleNamespace(pvalue=next(pvalues))

    monkeypatch.setattr(analysis, 'wilcoxon', fake_wilcoxon)
    rows = analysis.statistics(paired, resamples=20)
    assert [row['region'] for row in rows] == list(analysis.REGIONS)
    assert rows[0]['disease_wins'] == rows[0]['ties'] == rows[0]['anatomy_wins'] == 1
    assert [row['holm_adjusted_p'] for row in rows] == pytest.approx([0.06, 0.16, 0.15, 0.6, 1.0, 1.0])
    assert all(options == analysis.WILCOXON_PARAMETERS for _, options in calls)

    monkeypatch.setattr(analysis, 'wilcoxon', lambda *a, **k: pytest.fail('all-zero Wilcoxon called'))
    zeros = [dict(row, disease_dice=row['anatomy_dice']) for row in paired]
    assert all(row['wilcoxon_p'] == 1.0 for row in analysis.statistics(zeros, resamples=20))


def test_end_to_end_synthetic_artifacts_and_provenance(tmp_path):
    anatomy, disease, paths, write, counts = pair(tmp_path)
    disease[0]['ET_dice'] = '0.8'
    write()
    output = analysis.analyze(*paths, tmp_path / 'working', expected_cases=3,
                              expected_positive_counts=counts, seed=9, resamples=20)
    assert {path.name for path in output.iterdir()} == set(analysis.ARTIFACTS)
    with (output / 'paired_differences.csv').open(newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 3 * len(analysis.REGIONS)
    with (output / 'paired_statistics.csv').open(newline='') as stream:
        assert len(list(csv.DictReader(stream))) == len(analysis.REGIONS)
    provenance = json.loads((output / 'analysis_provenance.json').read_text())
    assert provenance['bootstrap_seed'] == 9 and provenance['bootstrap_resamples'] == 20
    assert provenance['expected_positive_counts'] == counts
    assert len(provenance['anatomy_input_sha256']) == len(provenance['disease_input_sha256']) == 64
    summary = json.loads((output / 'analysis_summary.json').read_text())
    assert summary['statistics'][2]['mean_delta_disease_minus_anatomy'] == pytest.approx(0.1)
    with pytest.raises(FileExistsError):
        analysis.analyze(*paths, output, expected_cases=3, expected_positive_counts=counts, resamples=20)
