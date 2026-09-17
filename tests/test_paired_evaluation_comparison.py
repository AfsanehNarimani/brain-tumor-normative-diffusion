import json

import numpy as np
import pandas as pd
import pytest

from analysis import paired_evaluation_comparison as c


def fixtures(tmp_path, cases=2):
    rows = []
    for i in range(cases):
        for region in c.REGIONS:
            row = dict(case_id=f'case-{i}', Labels=region, n_ref_instances=1,
                       global_bin_volume_ref=1, global_bin_volume_pred=1,
                       n_pred_instances=1, tp=1, fp=0, fn=0)
            row.update({m: 1.0 for m in c.METRICS})
            rows.append(row)
    a, b = pd.DataFrame(rows), pd.DataFrame(rows)
    p, q = tmp_path/'a.csv', tmp_path/'b.csv'
    a.to_csv(p, index=False)
    b.to_csv(q, index=False)
    return a, b, p, q


def write_pair(a, b, p, q):
    a.to_csv(p, index=False)
    b.to_csv(q, index=False)


@pytest.mark.parametrize('change,match', [
    (lambda b: b.__setitem__('case_id', ['other']*6 + ['case-1']*6), 'Mismatched'),
    (lambda b: b.__setitem__('Labels', ['NETC']*2 + list(b.Labels.iloc[2:])), 'duplicate'),
    (lambda b: b.__setitem__('Labels', ['MISSING' if x == 'TC' else x for x in b.Labels]), 'regions'),
    (lambda b: b.__setitem__('n_ref_instances', [2] + [1]*11), 'invariant'),
    (lambda b: b.__setitem__('global_bin_volume_ref', [2] + [1]*11), 'invariant'),
])
def test_strict_pair_validation(tmp_path, change, match):
    a, b, p, q = fixtures(tmp_path)
    change(b)
    write_pair(a, b, p, q)
    with pytest.raises(ValueError, match=match):
        c.validate_pair(p, q, expected_cases=2)


def test_missing_input(tmp_path):
    _, _, p, q = fixtures(tmp_path)
    with pytest.raises(FileNotFoundError):
        c.validate_pair(p, tmp_path/'missing.csv', expected_cases=2)


def test_nan_reference_invariant_and_undefined_single_pair(tmp_path):
    a, b, p, q = fixtures(tmp_path)
    a.loc[0, 'global_bin_volume_ref'] = np.nan
    b.loc[0, 'global_bin_volume_ref'] = np.nan
    a.loc[6, 'global_bin_dsc'] = np.inf
    b.loc[6, 'global_bin_dsc'] = np.nan
    write_pair(a, b, p, q)
    paired = c.validate_pair(p, q, expected_cases=2)
    row = c.metric_statistics(paired, resamples=10).iloc[0]
    assert row.both_finite == 1
    assert row.cohens_dz_undefined_reason == 'insufficient_finite_pairs'


def test_finite_counts_delta_and_directions(tmp_path):
    a, b, p, q = fixtures(tmp_path, cases=5)
    idx = a.Labels.eq('NETC')
    a.loc[idx, 'global_bin_dsc'] = [1, np.nan, 1, np.inf, -np.inf]
    b.loc[idx, 'global_bin_dsc'] = [2, 2, np.nan, np.inf, 1]
    a.loc[idx, 'global_bin_hd95'] = [3, 3, 3, 3, 3]
    b.loc[idx, 'global_bin_hd95'] = [2, 2, 2, 2, 2]
    write_pair(a, b, p, q)
    paired = c.validate_pair(p, q, expected_cases=5)
    stats = c.metric_statistics(paired, resamples=20)
    dice = stats[(stats.region == 'NETC') & (stats.metric == 'global_bin_dsc')].iloc[0]
    assert (dice.total_paired_rows, dice.both_finite, dice.e0_finite_e1a_nonfinite,
            dice.e0_nonfinite_e1a_finite, dice.both_nonfinite) == (5, 1, 1, 2, 1)
    assert (dice.e0_nan_count, dice.e1a_nan_count, dice.e0_posinf_count, dice.e1a_posinf_count,
            dice.e0_neginf_count) == (1, 1, 1, 1, 1)
    assert dice.mean_delta == 1 and dice.improved_case_count == 1
    hd = stats[(stats.region == 'NETC') & (stats.metric == 'global_bin_hd95')].iloc[0]
    assert hd.mean_delta == -1 and hd.improved_case_count == 5
    case = c.case_metrics(paired)
    assert case.loc[0, 'global_bin_dsc_delta'] == 1
    assert case.loc[1, 'global_bin_dsc_finite_pair'] == False
    assert np.isnan(case.loc[1, 'global_bin_dsc_delta'])
    assert np.isposinf(case.loc[4, 'global_bin_dsc_delta'])


def test_all_zero_dz_bootstrap_and_holm(tmp_path):
    a, b, p, q = fixtures(tmp_path)
    paired = c.validate_pair(p, q, expected_cases=2)
    row = c.metric_statistics(paired, resamples=20).iloc[0]
    assert row.wilcoxon_raw_p == 1 and row.wilcoxon_reason == 'all_zero_differences'
    assert pd.isna(row.cohens_dz) and row.cohens_dz_undefined_reason == 'zero_delta_variance'
    assert c.bootstrap_ci([1, 2, 3], 42, 100) == c.bootstrap_ci([1, 2, 3], 42, 100)
    adjusted = c.holm_adjust([.01, .04, .03, None])
    assert adjusted[:3] == pytest.approx([.03, .06, .06])
    assert adjusted[3] is None
    assert c.holm_adjust([None, None]) == [None, None]


def test_components_and_reference_empty(tmp_path):
    a, b, p, q = fixtures(tmp_path, cases=4)
    idx = a.Labels.eq('NETC')
    a.loc[idx, 'global_bin_volume_ref'] = 0
    b.loc[idx, 'global_bin_volume_ref'] = 0
    a.loc[idx, 'global_bin_volume_pred'] = [0, 0, 1, 1]
    b.loc[idx, 'global_bin_volume_pred'] = [0, 1, 0, 1]
    b.loc[idx, 'tp'] = [1, 2, 3, 4]
    write_pair(a, b, p, q)
    paired = c.validate_pair(p, q, expected_cases=4)
    transition = c.reference_empty_transitions(paired).iloc[0]
    assert [transition[k] for k in ('negative_to_negative', 'negative_to_positive',
                                    'positive_to_negative', 'positive_to_positive')] == [1, 1, 1, 1]
    component = c.component_comparison(paired)
    tp = component[(component.region == 'NETC') & (component.quantity == 'tp')].iloc[0]
    assert (tp.e0_total, tp.e1a_total, tp.absolute_change, tp.relative_change) == (4, 10, 6, 1.5)
    assert (tp.e1a_lower_count, tp.e1a_equal_count, tp.e1a_higher_count) == (0, 1, 3)


def test_overlap_and_ordering(tmp_path):
    source = dict(case_count=270, cases_with_overlap=2, total_foreground_voxels=100,
                  total_overlapping_voxels=10, overall_overlap_fraction=.1,
                  maximum_case_overlap_fraction=.2, maximum_overlap_channels_seen=2,
                  pairwise_overlap_counts={'NETC_RC': 4, 'ET_TC': 0})
    target = {**source, 'cases_with_overlap': 3,
              'pairwise_overlap_counts': {'NETC_RC': 8, 'ET_TC': 2}}
    result = c.compare_overlap(source, target)
    assert result['pairwise_overlap_counts']['NETC_RC']['relative_change'] == 1
    assert result['pairwise_overlap_counts']['ET_TC']['relative_change'] is None
    with pytest.raises(ValueError, match='case_count'):
        c.compare_overlap({**source, 'case_count': 269}, target)
    a, b, p, q = fixtures(tmp_path)
    paired = c.validate_pair(p, q, expected_cases=2)
    assert list(dict.fromkeys(paired.region)) == list(c.REGIONS)
    stats = c.metric_statistics(paired, resamples=10)
    assert stats.metric.tolist()[:6] == ['global_bin_dsc']*6
    assert stats.region.tolist()[:6] == list(c.REGIONS)


def test_end_to_end_artifacts(tmp_path):
    _, _, p, q = fixtures(tmp_path)
    overlap = dict(case_count=2, cases_with_overlap=0, total_foreground_voxels=10,
                   total_overlapping_voxels=0, overall_overlap_fraction=0,
                   maximum_case_overlap_fraction=0, maximum_overlap_channels_seen=1,
                   pairwise_overlap_counts={'NETC_RC': 0})
    o0, o1 = tmp_path/'o0.json', tmp_path/'o1.json'
    o0.write_text(json.dumps(overlap))
    o1.write_text(json.dumps(overlap))
    out = c.compare(p, q, o0, o1, tmp_path/'comparison', expected_cases=2, resamples=10)
    assert {file.name for file in out.iterdir()} == set(c.ARTIFACTS)
    assert json.loads((out/'comparison_metadata.json').read_text())['expected_case_count'] == 2
    assert len(pd.read_csv(out/'paired_case_metrics.csv')) == 12
