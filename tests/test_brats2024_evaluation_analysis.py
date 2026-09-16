import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from analysis.brats2024_evaluation_analysis import (
    COLUMNS, REGIONS, analyze, load_metrics, overlap_analysis, statistics, tables,
)


def fixture_frame(count=270):
    rows = []
    for index in range(count):
        for region in REGIONS:
            row = dict.fromkeys(COLUMNS, 1.0)
            row.update(case_id=f'case-{index:03d}', Labels=region)
            rows.append(row)
    return pd.DataFrame(rows)


def validated(frame):
    return load_metrics(frame.to_csv(index=False), frame.case_id.nunique())


def overlap(count=270):
    return dict(case_count=count, cases_with_overlap=2, total_foreground_voxels=10,
                total_overlapping_voxels=2, overall_overlap_fraction=.2,
                maximum_case_overlap_fraction=.5, maximum_overlap_channels_seen=3,
                pairwise_overlap_counts={'NETC_SNFH': 2, 'NETC_ET': 1, 'SNFH_ET': 1})


def test_exact_integrity():
    frame = load_metrics(fixture_frame().to_csv(index=False))
    assert len(frame) == 1620
    assert frame.case_id.nunique() == 270
    with pytest.raises(ValueError, match='270'):
        load_metrics(fixture_frame(269).to_csv(index=False))


@pytest.mark.parametrize('change,match', [
    (lambda f: pd.concat([f, f.iloc[[0]]]), 'Duplicate'),
    (lambda f: f.iloc[1:], 'rows'),
    (lambda f: f.assign(Labels=f.Labels.replace('NETC', 'OTHER')), 'regions'),
    (lambda f: f.drop(columns=['rq']), 'schema'),
    (lambda f: f.assign(case_id=''), 'case ID'),
    (lambda f: f.assign(rq='invalid'), 'numeric'),
])
def test_invalid_inputs(change, match):
    with pytest.raises(ValueError, match=match):
        load_metrics(change(fixture_frame()).to_csv(index=False))


def test_nonfinite_preservation_and_statistics():
    frame = fixture_frame(6)
    frame.loc[frame.Labels == 'NETC', 'sq_dsc'] = [0., 1., 3., np.nan, np.inf, -np.inf]
    values = validated(frame).query("Labels == 'NETC'").sq_dsc.to_numpy()
    assert np.isnan(values[3]) and np.isposinf(values[4]) and np.isneginf(values[5])
    stats = statistics(values)
    assert [stats[k] for k in ('total_count', 'finite_count', 'nan_count', 'positive_infinity_count', 'negative_infinity_count')] == [6, 3, 1, 1, 1]
    assert stats['mean'] == pytest.approx(4/3)
    assert stats['std'] == pytest.approx(np.std([0, 1, 3]))
    assert stats['median'] == 1 and stats['q25'] == .5 and stats['q75'] == 2
    assert stats['min'] == 0 and stats['max'] == 3
    assert statistics([np.nan, np.inf])['mean'] is None


def test_status_and_instance_burden():
    frame = fixture_frame(4)
    netc = frame.Labels == 'NETC'
    frame.loc[netc, 'global_bin_volume_ref'] = [1, 1, 0, 0]
    frame.loc[netc, 'global_bin_volume_pred'] = [1, 0, 1, 0]
    frame.loc[netc, 'fp'] = [0, 0, 3, 0]
    frame.loc[netc, 'global_bin_dsc'] = [1, 0, 0, np.nan]
    report = tables(validated(frame))
    status = report['region_status'][0]
    assert status['reference_positive_cases'] == status['prediction_positive_cases'] == 2
    for key in ('both_positive', 'reference_positive_prediction_empty', 'reference_empty_prediction_positive', 'both_empty'):
        assert status[key] == 1
    fp = report['false_positive_burden'][0]
    assert fp['total_fp'] == 3 and fp['cases_with_fp'] == 1
    assert fp['fraction_cases_with_fp'] == .25 and fp['mean_fp_per_case'] == .75
    assert fp['median_fp_per_case'] == 0 and fp['max_fp_in_single_case'] == 3
    burden = report['instance_burden'][0]
    assert burden['total_reference_instances'] == 4 and burden['total_fp'] == 3
    assert burden['mean'] == 1
    global_row = report['paper_global_metrics'][0]
    assert global_row['global_bin_dsc_nan_count'] == 1
    assert global_row['global_bin_dsc_mean'] == pytest.approx(1/3)
    assert report['paper_lesion_metrics'][0]['total_fp'] == 3


def test_deterministic_extremes():
    frame = validated(fixture_frame(12))
    frame.loc[0, 'global_bin_hd95'] = np.inf
    first = tables(frame, 2)['extreme_cases']
    assert first == tables(frame.sample(frac=1, random_state=4), 2)['extreme_cases']
    assert len(first) == 6 * 4 * 2
    assert first[0]['case_id'] == 'case-000'
    assert all(np.isfinite(row['value']) for row in first)
    with pytest.raises(ValueError, match='positive'):
        tables(frame, 0)


def test_overlap():
    original = overlap()
    result = overlap_analysis(original)
    assert all(result[k] == v for k, v in original.items())
    assert result['fraction_cases_with_overlap'] == 2/270
    assert sum(result['pairwise_fractions_relative_to_overlapping_voxels'].values()) == 2
    original['total_overlapping_voxels'] = 0
    assert all(v is None for v in overlap_analysis(original)['pairwise_fractions_relative_to_overlapping_voxels'].values())


def test_artifacts_hashes_and_read_only(tmp_path):
    source = tmp_path/'official_metrics.csv'
    source.write_text(fixture_frame().to_csv(index=False))
    upstream = tmp_path/'overlap_summary.json'
    upstream.write_text(json.dumps(overlap()))
    frozen = {p: p.read_bytes() for p in (source, upstream)}
    output = analyze(tmp_path)
    expected = {'analysis_summary.json', 'metric_statistics.csv', 'region_status.csv', 'instance_burden.csv',
                'false_positive_burden.csv', 'extreme_cases.csv', 'overlap_analysis.json',
                'paper_global_metrics.csv', 'paper_lesion_metrics.csv', 'analysis_metadata.json'}
    assert {p.name for p in output.iterdir()} == expected
    metadata = json.loads((output/'analysis_metadata.json').read_text())
    assert metadata['official_metrics_sha256'] == hashlib.sha256(frozen[source]).hexdigest()
    assert metadata['overlap_summary_sha256'] == hashlib.sha256(frozen[upstream]).hexdigest()
    assert metadata['expected_case_count'] == 270 and metadata['top_k'] == 10
    assert metadata['pandas_version'] == pd.__version__
    before = {p: p.read_bytes() for p in output.iterdir() if p.name != 'analysis_metadata.json'}
    analyze(tmp_path)
    assert all(p.read_bytes() == b for p, b in before.items())
    assert all(p.read_bytes() == b for p, b in frozen.items())


def test_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match='Required evaluation artifact missing'):
        analyze(tmp_path)


def test_overlapping_case_count_mismatch(tmp_path):
    (tmp_path/'official_metrics.csv').write_text(fixture_frame().to_csv(index=False))
    (tmp_path/'overlap_summary.json').write_text(json.dumps(overlap(5)))
    with pytest.raises(ValueError, match='Overlap case count'):
        analyze(tmp_path)
    assert not (tmp_path/'analysis').exists()
