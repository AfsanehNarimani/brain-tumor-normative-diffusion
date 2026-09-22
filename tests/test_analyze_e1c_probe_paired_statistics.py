"""E1-C wrapper tests using synthetic CSVs only."""
import csv
import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts import analyze_e1c_probe_paired_statistics as script


@pytest.fixture
def pair(tmp_path, monkeypatch):
    rows = []
    for index in range(270):
        row = {'case_id': f'case-{index:03d}'}
        for region, count in script.core.EXPECTED_POSITIVE_COUNTS.items():
            row[f'{region}_target_present'] = int(index < count)
            row[f'{region}_dice'] = 0.5 if index < count else ''
        rows.append(row)
    anatomy = [dict(row) for row in rows]
    disease = [dict(row) for row in reversed(rows)]
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    paths = inputs / 'anatomy.csv', inputs / 'disease.csv'

    def write():
        for path, data in zip(paths, (anatomy, disease)):
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(data)
        # Replace only expected digests for synthetic fixtures, never skip hashing.
        for name, path in zip(('ANATOMY_SHA256', 'DISEASE_SHA256'), paths):
            monkeypatch.setattr(script, name, hashlib.sha256(path.read_bytes()).hexdigest())

    write()
    return anatomy, disease, paths, write, tmp_path / 'working'


def test_happy_path_schema_direction_and_provenance(pair):
    _, disease, paths, write, output = pair
    disease[-1]['NETC_dice'] = 0.8
    write()
    assert script.analyze(*paths, output) == output
    assert {p.name for p in output.iterdir()} == set(script.core.ARTIFACTS)
    for name, fields in [('paired_statistics.csv', script.core.STAT_FIELDS),
                         ('paired_differences.csv', script.core.DIFFERENCE_FIELDS),
                         ('paper_table.csv', script.core.PAPER_FIELDS)]:
        with (output / name).open(newline='') as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == list(fields)
            rows = list(reader)
        assert len(rows) == (1282 if name == 'paired_differences.csv' else 6)
        if name == 'paired_differences.csv':
            assert rows[0]['case_id'] == 'case-000'
            assert float(rows[0]['delta_disease_minus_anatomy']) == pytest.approx(0.3)
    summary = json.loads((output / 'analysis_summary.json').read_text())
    assert summary['statistics'][0]['mean_delta_disease_minus_anatomy'] == pytest.approx(0.3 / 114)
    assert summary['target_positive_counts'] == script.core.EXPECTED_POSITIVE_COUNTS
    provenance = json.loads((output / 'analysis_provenance.json').read_text())
    assert provenance['analysis_name'] == 'E1-C latent-probe paired comparison'
    assert provenance['delta_direction'] == 'Zd - Za'
    assert provenance['confidence_level'] == 0.95
    assert provenance['bootstrap_seed'] == 20260917 and provenance['bootstrap_resamples'] == 10000
    assert provenance['win_tie_tolerance'] == 1e-8
    assert provenance['wilcoxon_parameters'] == script.core.WILCOXON_PARAMETERS
    assert 'Holm' in provenance['multiple_comparison_correction']
    assert 'git_commit_sha' in provenance
    assert provenance['case_count'] == 270
    for label, path in zip(('anatomy', 'disease'), paths):
        assert provenance[f'{label}_input'] == str(path)
        assert provenance[f'{label}_input_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_scientific_hash_constants():
    assert script.ANATOMY_SHA256 == 'd1328d3b800490cf56ee44532d185c9fd2e76cb9ca470fa713df809607212880'
    assert script.DISEASE_SHA256 == 'f68053782d926051c06ea2b6619652c3ee95f9f366f0065b163764d32515185b'


@pytest.mark.parametrize('side', [0, 1])
def test_hash_failure_precedes_statistics(pair, monkeypatch, side):
    _, _, paths, _, output = pair
    paths[side].write_text(paths[side].read_text() + '\n')
    analyze = MagicMock()
    monkeypatch.setattr(script.core, 'analyze', analyze)
    with pytest.raises(ValueError, match='SHA256'):
        script.analyze(*paths, output)
    analyze.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize('mutation,message', [
    ('duplicate', 'duplicate case_id'), ('set', 'Case set mismatch'),
    ('presence', 'Target-presence mismatch'), ('count', 'target-positive count mismatch'),
    ('missing', 'Expected 270 cases'), ('extra', 'Expected 270 cases'),
])
def test_integrity_failures_before_statistics(pair, monkeypatch, mutation, message):
    anatomy, disease, paths, write, output = pair
    if mutation == 'duplicate':
        anatomy[1]['case_id'] = anatomy[0]['case_id']
    elif mutation == 'set':
        disease[0]['case_id'] = 'other'
    elif mutation == 'presence':
        disease[-1].update(NETC_target_present=0, NETC_dice='')
    elif mutation == 'count':
        for row in (anatomy[0], disease[-1]):
            row.update(NETC_target_present=0, NETC_dice='')
    elif mutation == 'missing':
        anatomy.pop()
    else:
        disease.append({**disease[0], 'case_id': 'extra'})
    write()
    statistics = MagicMock()
    monkeypatch.setattr(script.core, 'statistics', statistics)
    with pytest.raises(ValueError, match=message):
        script.analyze(*paths, output)
    statistics.assert_not_called()
    assert not output.exists()


def test_bootstrap_reproducibility(pair):
    _, disease, paths, write, output = pair
    disease[-1]['NETC_dice'] = 0.8
    disease[-2]['NETC_dice'] = 0.1
    write()
    script.analyze(*paths, output)
    other = output.with_name('repeated')
    script.analyze(*paths, other)
    assert (output / 'paired_statistics.csv').read_bytes() == (other / 'paired_statistics.csv').read_bytes()
    delta = np.array([0.3, -0.4, 0.0])
    indices = np.random.default_rng(20260917).integers(0, 3, (10000, 3))
    expected = np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])
    assert script.core.bootstrap_ci(delta, 20260917, 10000) == pytest.approx(expected)


def test_holm_six_regions_and_raw_tie_deltas(pair, monkeypatch):
    _, disease, paths, write, output = pair
    for region in script.core.REGIONS:
        disease[-1][f'{region}_dice'] = 0.5 + 5e-9
    write()
    pvalues = iter([0.01, 0.04, 0.03, 0.2, 0.5, 0.9])
    calls = []
    def wilcoxon(delta, **options):
        calls.append((delta.copy(), options))
        return SimpleNamespace(pvalue=next(pvalues))
    monkeypatch.setattr(script.core, 'wilcoxon', wilcoxon)
    script.analyze(*paths, output)
    rows = json.loads((output / 'analysis_summary.json').read_text())['statistics']
    assert [r['holm_adjusted_p'] for r in rows] == pytest.approx([0.06, 0.16, 0.15, 0.6, 1, 1])
    assert len(calls) == 6
    assert all(options == script.core.WILCOXON_PARAMETERS for _, options in calls)
    assert all(delta[0] == (0.5 + 5e-9) - 0.5 for delta, _ in calls)
    assert all(row['ties'] == row['n'] for row in rows)


def test_all_zero_policy(pair, monkeypatch):
    _, _, paths, _, output = pair
    monkeypatch.setattr(script.core, 'wilcoxon', lambda *a, **k: pytest.fail('all-zero test called'))
    script.analyze(*paths, output)
    rows = json.loads((output / 'analysis_summary.json').read_text())['statistics']
    assert all(r['wilcoxon_p'] == r['holm_adjusted_p'] == 1 for r in rows)
    assert all(r['bootstrap_ci95_low'] == r['bootstrap_ci95_high'] == 0 for r in rows)


def test_existing_and_frozen_outputs_rejected(pair):
    _, _, paths, _, output = pair
    output.mkdir()
    with pytest.raises(FileExistsError):
        script.analyze(*paths, output)
    with pytest.raises(ValueError, match='frozen'):
        script.analyze(*paths, 'outputs/frozen/forbidden-comparison')


def test_cli_defaults(monkeypatch):
    analyze = MagicMock()
    monkeypatch.setattr(script, 'analyze', analyze)
    monkeypatch.setattr(sys, 'argv', ['analyze_e1c_probe_paired_statistics'])
    script.main()
    analyze.assert_called_once_with(script.ANATOMY_CSV, script.DISEASE_CSV, script.OUTPUT_DIR)
