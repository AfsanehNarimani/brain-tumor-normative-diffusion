import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analyze_d0c3_pathology_association.py'
spec = importlib.util.spec_from_file_location('analyze_d0c3_pathology_association', SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_contract_is_predeclared_and_not_result_bearing():
    contract = json.loads(analysis.CONTRACT_PATH.read_text())
    assert hashlib.sha256(analysis.CONTRACT_PATH.read_bytes()).hexdigest() == analysis.CONTRACT_SHA256
    assert contract['status'] == 'DESIGN_LOCKED_PRE_ANALYSIS'
    assert contract['analysis_started'] is False
    assert contract['results_observed'] is False
    assert contract['endpoints'] == {'primary': 'WT', 'secondary': ['TC', 'ET']}
    assert contract['bootstrap']['replicates'] == 10000
    assert contract['bootstrap']['seed'] == 20260923


def test_high_end_only_padding_and_exact_block_mapping():
    labels = np.zeros((16, 8, 8), dtype=np.uint8)
    labels[0, 0, 0] = 1
    padded = analysis.high_end_pad(labels, (24, 16, 8))
    assert padded.shape == (24, 16, 8)
    assert np.array_equal(padded[:16, :8, :], labels)
    assert np.all(padded[16:] == 0)
    assert analysis.exact_block_reduce(padded == 1).shape == (3, 2, 1)
    region_masks, clean_background, result = analysis.map_gt_to_latent(labels, labels.shape, (24, 16, 8))
    assert result.shape == (24, 16, 8)
    assert region_masks[1][0, 0, 0]
    assert clean_background[1, 0, 0]  # high-end pad became a clean zero block


def test_region_labels_rc_exclusion_and_valid_support():
    gt = np.zeros((24, 8, 8), dtype=np.uint8)
    gt[:8] = 4  # RC must not be clean background for any region
    gt[16, 0, 0] = 1  # one-voxel overlap makes this whole cell positive
    valid = np.array([[[True]], [[True]], [[False]]]).reshape(3, 1, 1)
    masks, _ = analysis.comparison_masks(gt, valid, gt.shape, gt.shape)
    for region in ('WT', 'TC', 'ET'):
        positive, background = masks[region]
        assert not background[0, 0, 0]
        assert background[1, 0, 0]
        assert not positive[2, 0, 0]  # unsupported positive is excluded
    assert masks['WT'][0][2, 0, 0] is np.False_ or bool(masks['WT'][0][2, 0, 0]) is False
    assert not bool(masks['TC'][0][2, 0, 0])
    assert not bool(masks['ET'][0][2, 0, 0])


def test_region_specific_abnormal_cells_are_not_clean_background():
    gt = np.zeros((32, 8, 8), dtype=np.uint8)
    gt[:8] = 2  # SNFH only: WT positive; TC and ET exclude, not negative
    gt[8:16] = 4  # RC excluded from clean background
    gt[16:24] = 1  # NETC: WT/TC positive, ET excluded
    masks, _ = analysis.comparison_masks(gt, np.ones((4, 1, 1), dtype=bool), gt.shape, gt.shape)
    wt_pos, wt_bg = masks['WT']
    tc_pos, tc_bg = masks['TC']
    et_pos, et_bg = masks['ET']
    assert wt_pos[:, 0, 0].tolist() == [True, False, True, False]
    assert wt_bg[:, 0, 0].tolist() == [False, False, False, True]
    assert tc_pos[:, 0, 0].tolist() == [False, False, True, False]
    assert tc_bg[:, 0, 0].tolist() == [False, False, False, True]
    assert et_pos[:, 0, 0].tolist() == [False, False, False, False]
    assert et_bg[:, 0, 0].tolist() == [False, False, False, True]


def test_auc_auprc_baseline_and_lift():
    got = analysis.binary_metrics([3, 4], [1, 2])
    assert got['auroc'] == 1.0
    assert got['auprc'] == 1.0
    assert got['prevalence'] == 0.5
    assert got['auprc_lift'] == 0.5
    tied = analysis.binary_metrics([1], [1])
    assert tied['auroc'] == 0.5
    assert tied['auprc'] == 0.5


def test_per_case_metrics_mean_median_and_n_evaluable_reasons():
    # Spatial grid is fixed; only first two cells are supported and labelled.
    delta = np.zeros((1, 23, 28, 23), dtype=np.float32)
    support = np.zeros_like(delta, dtype=bool)
    support[0, 0, 0, 0:2] = True
    gt = np.zeros((182, 218, 182), dtype=np.uint8)
    gt[:8, :8, :8] = 1
    delta[0, 0, 0, 0] = 2.0
    delta[0, 0, 0, 1] = 1.0
    rows = analysis.per_case_region_metrics('synthetic', delta, support, gt, gt.shape, (184, 224, 184))
    wt = rows[0]
    assert wt['n_positive_cells'] == 1 and wt['n_background_cells'] == 1
    assert wt['positive_DeltaZ_mean'] == 2.0
    assert wt['background_DeltaZ_mean'] == 1.0
    assert wt['mean_difference'] == 1.0 and wt['median_difference'] == 1.0
    assert wt['AUROC'] == 1.0 and wt['AUPRC'] == 1.0
    assert wt['AUPRC_baseline_prevalence'] == 0.5

    no_positive = np.zeros((182, 218, 182), dtype=np.uint8)
    result = analysis.per_case_region_metrics('none', delta, support, no_positive, no_positive.shape, (184, 224, 184))
    assert not result[0]['evaluable']
    assert result[0]['evaluability_reason'] == 'no_positive_supported_cells'
    tumor_everywhere = np.ones_like(no_positive)
    result = analysis.per_case_region_metrics('no-bg', delta, support, tumor_everywhere, tumor_everywhere.shape, (184, 224, 184))
    assert not result[0]['evaluable']
    assert result[0]['evaluability_reason'] == 'no_clean_background_supported_cells'


def test_bootstrap_deterministic_wilcoxon_and_holm():
    vals = [0.2, 0.5, 0.7, 0.9]
    a = analysis.bootstrap_median_ci(vals, replicates=1000, seed=20260923)
    b = analysis.bootstrap_median_ci(vals, replicates=1000, seed=20260923)
    assert a == b
    assert a[0] == 0.6 and a[1] <= a[0] <= a[2]
    test = analysis.paired_wilcoxon([0.6, 0.7, 0.8], 0.5)
    assert test['n'] == 3 and 0 <= test['p_value'] <= 1
    assert analysis.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_aggregate_is_case_level_and_holm_by_family():
    rows = []
    for case_id, value in [('a', 0.1), ('b', 0.2), ('c', -0.1)]:
        for region in ('WT', 'TC', 'ET'):
            rows.append(dict(case_id=case_id, region=region, evaluable=True,
                AUROC=0.6+value, AUPRC=0.5+value, AUPRC_lift=value,
                mean_difference=value, median_difference=value))
    summaries = analysis.aggregate_case_metrics(rows, replicates=50, seed=20260923)
    assert len(summaries) == 3
    assert all(row['total_cases'] == 3 and row['evaluable_cases'] == 3 for row in summaries)
    assert summaries[0]['median_median_difference'] == 0.1
    assert 'AUROC_vs_0.5_p_holm' in summaries[0]


def test_shape_label_and_block_geometry_failures():
    with pytest.raises(ValueError, match='GT shape mismatch'):
        analysis.map_gt_to_latent(np.zeros((8, 8, 8)), (16, 8, 8), (16, 8, 8))
    bad = np.zeros((8, 8, 8), dtype=np.int16); bad[0, 0, 0] = 5
    with pytest.raises(ValueError, match='unknown BraTS labels'):
        analysis.map_gt_to_latent(bad, bad.shape, bad.shape)
    with pytest.raises(ValueError, match='DeltaZ shape mismatch'):
        analysis.per_case_region_metrics('bad', np.zeros((1, 2, 2, 2)), np.ones((1, 2, 2, 2)),
                                         np.zeros((182, 218, 182)), (182, 218, 182), (184, 224, 184))
    with pytest.raises(ValueError, match='divisible'):
        analysis.exact_block_reduce(np.zeros((9, 8, 8), dtype=bool))


def test_reading_analysis_inputs_does_not_modify_source_tensor(tmp_path):
    case = tmp_path / 'case'; case.mkdir()
    shape = (1, 23, 28, 23)
    payload = dict(DeltaZ_global=torch.ones(shape), valid_support_global=torch.ones(shape, dtype=torch.bool),
        contribution_count_global=torch.ones(shape, dtype=torch.int32), global_latent_shape=torch.tensor([23,28,23]),
        original_shape=torch.tensor([182,218,182]), padded_shape=torch.tensor([184,224,184]))
    path = case / 'case_deltaz.pt'; torch.save(payload, path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    delta, valid, original, padded = analysis._load_primary_tensor(case)
    analysis.per_case_region_metrics('synthetic', delta, valid, np.zeros((182,218,182), dtype=np.uint8), original, padded)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
