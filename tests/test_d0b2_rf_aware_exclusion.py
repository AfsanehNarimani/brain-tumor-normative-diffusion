"""Synthetic D0-B2 geometry and occupancy tests."""
import csv
import hashlib
import json

import pytest
import torch

from analysis.rf_geometry import derive_encoder_geometry, encoder_1d_layers
from models.encoder import Encoder
from scripts import audit_d0b2_rf_aware_exclusion as audit


def test_encoder_geometry_is_derived_and_clipped():
    geometry = derive_encoder_geometry(Encoder())
    assert (geometry.input_size, geometry.output_size, geometry.effective_stride, geometry.theoretical_rf_size) == (96, 12, 8, 68)
    assert geometry.theoretical_intervals == tuple((-30 + 8*i, 38 + 8*i) for i in range(12))
    assert geometry.clipped_intervals[0] == (0, 38)
    assert geometry.clipped_intervals[-1] == (58, 96)
    assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(geometry.theoretical_intervals, geometry.theoretical_intervals[1:]))
    assert all(b[0] - a[0] == 8 and b[1] - a[1] == 8 for a, b in zip(geometry.theoretical_intervals, geometry.theoretical_intervals[1:]))
    assert len(encoder_1d_layers(Encoder())) == 11


def test_rf_integral_query_matches_direct_reference():
    geometry = derive_encoder_geometry(Encoder())
    abnormal = torch.zeros(audit.PATCH_SHAPE, dtype=torch.bool)
    abnormal[20, 20, 20] = True
    fast = audit.rf_contaminated(abnormal, geometry)
    reference = torch.zeros_like(fast)
    for d, interval_d in enumerate(geometry.clipped_intervals):
        for h, interval_h in enumerate(geometry.clipped_intervals):
            for w, interval_w in enumerate(geometry.clipped_intervals):
                reference[d,h,w] = abnormal[interval_d[0]:interval_d[1], interval_h[0]:interval_h[1], interval_w[0]:interval_w[1]].any()
    assert torch.equal(fast, reference)
    outside = torch.zeros_like(abnormal); outside[95,95,95] = True
    # Boundary RFs overlap substantially; the corner voxel is shared by the
    # final 4x4x4 latent positions under the clipped 68-voxel intervals.
    assert audit.rf_contaminated(outside, geometry).sum() == 64


def test_rf_spatial_specificity_for_one_latent_cell():
    geometry = derive_encoder_geometry(Encoder())
    target = (6, 6, 6)
    interval = geometry.clipped_intervals[target[0]]
    inside = torch.zeros(audit.PATCH_SHAPE, dtype=torch.bool)
    inside[interval[0], interval[0], interval[0]] = True
    outside = torch.zeros_like(inside)
    outside[0, 0, 0] = True
    assert audit.rf_contaminated(inside, geometry)[target]
    assert not audit.rf_contaminated(outside, geometry)[target]


def test_eligibility_and_zero_denominator():
    geometry = derive_encoder_geometry(Encoder())
    image = torch.ones((4, *audit.PATCH_SHAPE)); segmentation = torch.zeros(audit.PATCH_SHAPE, dtype=torch.int64)
    segmentation[20,20,20] = 1
    metrics = audit.direct_and_rf_metrics(image, segmentation, geometry)
    assert metrics['valid_support_count'] == 1728
    assert metrics['eligible_count'] == 1728 - metrics['rf_contaminated_count']
    zero = audit.direct_and_rf_metrics(torch.zeros_like(image), segmentation, geometry)
    assert zero['rf_valid_support_excluded_fraction'] is None and zero['eligible_fraction_of_valid_support'] is None


def test_support_and_direct_block_rules_are_preserved():
    geometry = derive_encoder_geometry(Encoder())
    image = torch.zeros((4, *audit.PATCH_SHAPE))
    # One complete 8^3 support block makes exactly one direct latent cell valid.
    image[:, :8, :8, :8] = 1
    segmentation = torch.zeros(audit.PATCH_SHAPE, dtype=torch.int64)
    segmentation[4, 4, 4] = 4
    metrics = audit.direct_and_rf_metrics(image, segmentation, geometry)
    assert metrics['valid_support_count'] == 1
    assert metrics['direct_abnormal_count_gt0'] == 1
    assert metrics['direct_valid_support_excluded_count_gt0'] == 1
    assert metrics['eligible_count'] == 0


def test_all_raw_abnormal_labels_are_included():
    geometry = derive_encoder_geometry(Encoder())
    image = torch.ones((4, *audit.PATCH_SHAPE))
    segmentation = torch.zeros(audit.PATCH_SHAPE, dtype=torch.int64)
    segmentation[0, 0, 0] = 1
    segmentation[0, 0, 1] = 2
    segmentation[0, 1, 0] = 3
    segmentation[1, 0, 0] = 4
    metrics = audit.direct_and_rf_metrics(image, segmentation, geometry)
    assert metrics['abnormal_positive_patch'] is True
    assert metrics['direct_abnormal_count_gt0'] == 1


def test_train_split_sha_guard_precedes_processing(tmp_path, monkeypatch):
    split = tmp_path / 'train.txt'; split.write_text('case\n')
    config = tmp_path / 'data.yaml'; config.write_text(f'train_cases: {split}\nroot: {tmp_path}\n')
    monkeypatch.setattr(audit, 'TRAIN_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='Training split SHA256 mismatch'):
        audit.run(config, tmp_path / 'manifest.csv', tmp_path / 'out')


def test_manifest_sha_guard_is_enforced(tmp_path, monkeypatch):
    split = tmp_path/'train.txt'; split.write_text('case\n')
    manifest = tmp_path/'manifest.csv'; manifest.write_text('bad')
    monkeypatch.setattr(audit, 'EXPECTED_CASES', 1); monkeypatch.setattr(audit, 'TRAIN_SHA256', hashlib.sha256(split.read_bytes()).hexdigest())
    config = tmp_path/'data.yaml'; config.write_text(f"train_cases: {split}\nroot: {tmp_path}\n")
    with pytest.raises(ValueError, match='manifest SHA256 mismatch'):
        audit.run(config, manifest, tmp_path/'out')


def _configure_single_case_run(tmp_path, monkeypatch, expected_direct=0.0, image=None, raw_mask=None):
    split = tmp_path / 'train.txt'; split.write_text('case-001\n')
    manifest = tmp_path / 'manifest.csv'; manifest.write_text('synthetic manifest\n')
    config = tmp_path / 'data.yaml'; config.write_text(f'train_cases: {split}\nroot: {tmp_path}\n')
    summary = tmp_path / 'b1.json'; summary.write_text(json.dumps({'distributions': {'all_patches': {'fraction_valid_support_excluded_gt_0': {'mean': expected_direct}}}}))
    rows = [
        {'case_id': 'case-001', 'patch_index': 0, 'seed': 1,
         'crop_start_d': 0, 'crop_start_h': 0, 'crop_start_w': 0,
         'crop_stop_d': 96, 'crop_stop_h': 96, 'crop_stop_w': 96},
        {'case_id': 'case-001', 'patch_index': 1, 'seed': 2,
         'crop_start_d': 96, 'crop_start_h': 0, 'crop_start_w': 0,
         'crop_stop_d': 192, 'crop_stop_h': 96, 'crop_stop_w': 96},
    ]
    monkeypatch.setattr(audit, 'EXPECTED_CASES', 1)
    monkeypatch.setattr(audit, 'TRAIN_SHA256', hashlib.sha256(split.read_bytes()).hexdigest())
    monkeypatch.setattr(audit, 'MANIFEST_SHA256', hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(audit, 'FROZEN_B1_SUMMARY', str(summary))
    monkeypatch.setattr(audit, 'load_manifest', lambda *_: rows)
    default_image = torch.zeros((4, 192, 96, 96)); default_image[:, :96] = 1
    default_mask = torch.zeros((192, 96, 96), dtype=torch.int64)
    monkeypatch.setattr(audit, 'load_case', lambda *_: (image if image is not None else default_image, raw_mask if raw_mask is not None else default_mask))
    return config, manifest


def test_case_aggregation_retains_zero_support_case_and_undefined_ratios(tmp_path, monkeypatch):
    config, manifest = _configure_single_case_run(tmp_path, monkeypatch)
    output = audit.run(config, manifest, tmp_path / 'out')
    with (output / 'case_rf_statistics.csv').open(newline='') as stream:
        row = next(csv.DictReader(stream))
    assert row['case_id'] == 'case-001'
    assert row['patch_count'] == '2'
    assert row['rf_valid_support_excluded_fraction_mean'] == '0.0'
    assert row['rf_valid_support_excluded_fraction_defined_patch_count'] == '1'
    with (output / 'patch_rf_statistics.csv').open(newline='') as stream:
        patch_rows = list(csv.DictReader(stream))
    assert patch_rows[1]['rf_valid_support_excluded_fraction'] == ''
    summary = json.loads((output / 'rf_exclusion_summary.json').read_text())
    assert row['valid_support_zero_denominator_patch_count'] == '1'
    assert summary['cases_with_zero_defined_eligible_support_ratio'] == 0


def test_direct_b1_consistency_failure_is_loud(tmp_path, monkeypatch):
    config, manifest = _configure_single_case_run(tmp_path, monkeypatch, expected_direct=0.5)
    with pytest.raises(ValueError, match='D0-B1 direct exclusion consistency failed'):
        audit.run(config, manifest, tmp_path / 'out')


def test_generated_provenance_records_audit_guards(tmp_path, monkeypatch):
    image = torch.ones((4, 192, 96, 96))
    config, manifest = _configure_single_case_run(tmp_path, monkeypatch, image=image)
    output = audit.run(config, manifest, tmp_path / 'out')
    provenance = json.loads((output / 'analysis_provenance.json').read_text())
    for field in ('validation_data_used', 'model_training_performed', 'diffusion_model_loaded',
                  'new_random_sampling_performed', 'rejection_sampling_used', 'augmentation_used',
                  'final_corruption_strategy_selected', 'final_abnormal_policy_selected'):
        assert provenance[field] is False
    assert 'ANY modality nonzero before normalization' in provenance['support_definition']
    assert 'corresponding 8x8x8 input block' in provenance['direct_abnormal_rule']
    assert 'clipped real-input theoretical receptive-field box' in provenance['rf_contamination']


def test_constants_and_architecture_only_entrypoint(tmp_path, monkeypatch):
    # The audit has no model import/load path beyond architecture-only Encoder construction.
    assert audit.EXPERIMENT_BASE_COMMIT == '7c0fcd74f04cfee82a0c0f24e7d4d350d24e6ce9'
    assert audit.SUPPORT_THRESHOLD == 0.50
    assert audit.RAW_ABNORMAL_LABELS == (1,2,3,4)
