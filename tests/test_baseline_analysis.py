"""Frozen synthetic experiment analysis, without data or accelerator dependencies."""
import csv
import json
import pytest
from analysis.baseline_analysis import REGIONS, analyze, mean_present, netc_analysis, write_csv


def test_selection_and_collapse():
    history = []
    for epoch, value in enumerate([.2, .6, 1e-10, 0], 1):
        history.append({'epoch': epoch, 'NETC_positive_cases': 1, 'NETC_case_dice': value, 'NETC_dataset_dice': value})
    assert mean_present(history[1]) == .6
    result = netc_analysis(history, 2)
    assert result['best_NETC_epoch'] == 2
    assert result['first_zero_NETC_epoch'] == 3
    assert result['collapse_epoch'] == 3
    assert result['consecutive_zero_epochs_after_first_zero'] == 2
    assert netc_analysis(history[:2], 2)['first_zero_NETC_epoch'] is None
    assert mean_present({'NETC_positive_cases': 0, 'NETC_case_dice': 1}) is None


def test_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match='Missing required experiment artifacts'):
        analyze(tmp_path)


def test_complete_reports_and_frozen_inputs(tmp_path):
    (tmp_path / 'evaluation').mkdir()
    history = []
    for epoch, value in enumerate([.2, .8, 0], 1):
        row = {'epoch': epoch, 'training_loss': 1 / epoch, 'validation_loss': 1 / epoch}
        for r in REGIONS:
            row.update({f'{r}_positive_cases': 1, f'{r}_case_dice': value, f'{r}_dataset_dice': value})
        history.append(row)
    write_csv(tmp_path / 'history.csv', history, list(history[0]))
    (tmp_path / 'summary.json').write_text(json.dumps({'best_epoch': 2, 'best_validation_metric': .8}))
    regions = {r: {'positive_case_count': 1, 'case_dice': .8, 'dataset_dice': .8} for r in REGIONS}
    (tmp_path / 'evaluation/aggregate_metrics.json').write_text(json.dumps({'regions': regions, 'case_count': 2, 'mean_validation_metric': .8}))
    cases = [{'case_id': 'good'}, {'case_id': 'empty'}]
    for r in REGIONS:
        cases[0].update({f'{r}_target_present': 1, f'{r}_dice': .8})
        cases[1].update({f'{r}_target_present': 0, f'{r}_dice': ''})
    write_csv(tmp_path / 'evaluation/per_case_metrics.csv', cases, list(cases[0]))
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = analyze(tmp_path)
    assert result['best_epoch'] == 2 and result['minimum_validation_loss_epoch'] == 3
    assert result['NETC_first_zero_epoch'] == 3
    assert all(p.read_bytes() == value for p, value in before.items())
    output = tmp_path / 'analysis'
    stats = list(csv.DictReader((output / 'per_region_statistics.csv').open()))
    assert all(row['count'] == '1' and float(row['mean']) == .8 for row in stats)
    difficult = list(csv.DictReader((output / 'difficult_cases.csv').open()))
    assert difficult[0]['case_id'] == 'good'
    assert difficult[1]['mean_present_region_dice'] == ''
    table = list(csv.DictReader((output / 'baseline_metrics.csv').open()))
    assert [r['region'] for r in table] == list(REGIONS)
    for name in ('loss_curve', 'region_dice_curves', 'selection_metric_curve', 'netc_dice_curve', 'case_dice_boxplot'):
        for suffix in ('png', 'pdf'):
            assert (output / 'figures' / f'{name}.{suffix}').stat().st_size > 0
    assert (output / 'baseline_metrics.md').exists()
    assert (output / 'analysis_summary.json').exists()


def test_initial_zero_is_not_a_collapse():
    history = [{'epoch': 1, 'NETC_positive_cases': 2, 'NETC_case_dice': 0}]
    result = netc_analysis(history, 1)
    assert result['first_zero_NETC_epoch'] == 1
    assert result['collapse_epoch'] is None


def test_missing_positive_observation_does_not_invent_selection():
    assert mean_present({'NETC_positive_cases': 1, 'NETC_case_dice': '',
                         'ET_positive_cases': 1, 'ET_case_dice': .8}) is None
