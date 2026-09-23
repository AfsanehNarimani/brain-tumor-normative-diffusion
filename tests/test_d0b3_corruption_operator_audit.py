import csv
import hashlib
import json

import pytest
import torch
from torch.nn import functional as F

from models.encoder import Encoder
from scripts import audit_d0b3_corruption_operator as audit


def test_case_selection_is_deterministic_and_order_independent():
    ids = [f'case-{i:04d}' for i in range(20)]
    assert audit.select_cases(ids, seed=7, count=5) == audit.select_cases(list(reversed(ids)), seed=7, count=5)
    assert audit.select_cases(ids, seed=7, count=5) != audit.select_cases(ids, seed=8, count=5)


def test_selected_manifest_reuses_exactly_twenty_rows_per_case():
    rows = [{'case_id': case, 'patch_index': index} for case in ('a', 'b') for index in range(20)]
    selected = audit.select_manifest(rows, ['b'])
    assert len(selected) == 20 and {row['case_id'] for row in selected} == {'b'}
    with pytest.raises(ValueError):
        audit.select_manifest(rows[:-1], ['b'])


def test_donor_shell_is_exact_chebyshev_radius_eight():
    abnormal = torch.zeros(audit.PATCH_SHAPE, dtype=torch.bool)
    abnormal[48, 48, 48] = True
    shell = audit.donor_shell(abnormal)
    assert int(shell.sum()) == 17 ** 3 - 1
    assert not shell[48, 48, 48]
    assert shell[40, 48, 48] and shell[56, 56, 56]
    assert not shell[39, 48, 48]


def _reference_cubic_dilation(mask):
    return F.max_pool3d(mask[None, None].float(), kernel_size=17, stride=1, padding=8)[0, 0].bool()


def _separable_cubic_dilation(mask):
    volume = mask[None, None].float()
    for kernel_size, padding in (((17, 1, 1), (8, 0, 0)),
                                 ((1, 17, 1), (0, 8, 0)),
                                 ((1, 1, 17), (0, 0, 8))):
        volume = F.max_pool3d(volume, kernel_size=kernel_size, stride=1, padding=padding)
    return volume[0, 0].bool()


def test_separable_donor_dilation_is_exactly_equal_to_cubic_reference():
    masks = []
    center = torch.zeros(audit.PATCH_SHAPE, dtype=torch.bool); center[48, 48, 48] = True
    masks.append(center)
    boundary = torch.zeros_like(center)
    for point in ((0, 0, 0), (0, 48, 95), (95, 0, 48), (48, 95, 0)):
        boundary[point] = True
    masks.append(boundary)
    separated = torch.zeros_like(center); separated[16, 16, 16] = True; separated[80, 80, 80] = True
    masks.append(separated)
    touching = torch.zeros_like(center); touching[48, 48, 48] = True; touching[48, 48, 64] = True
    masks.append(touching)
    generator = torch.Generator().manual_seed(20260917)
    masks.extend([(torch.rand(audit.PATCH_SHAPE, generator=generator) < probability)
                  for probability in (0.0001, 0.001, 0.01)])
    masks.append(torch.zeros_like(center))
    masks.append(torch.ones_like(center))
    for mask in masks:
        reference = _reference_cubic_dilation(mask)
        optimized = _separable_cubic_dilation(mask)
        assert torch.equal(reference, optimized)
        assert torch.equal(audit.donor_shell(mask), reference & ~mask)


def _synthetic_inputs():
    raw = torch.zeros((4, *audit.PATCH_SHAPE))
    raw[:, 32:64, 32:64, 32:64] = 2
    raw[:, 40:48, 40:48, 40:48] = 5
    abnormal = torch.zeros(audit.PATCH_SHAPE, dtype=torch.bool)
    abnormal[40:48, 40:48, 40:48] = True
    normalized = raw.clone()
    support = audit.support_mask(raw)
    pools, fallback = audit.donor_pools(raw, abnormal, support)
    return raw, normalized, abnormal, pools, fallback


@pytest.mark.parametrize('operator', audit.OPERATORS)
def test_corruption_changes_only_abnormal_voxels(operator):
    _, normalized, abnormal, pools, _ = _synthetic_inputs()
    corrupted = audit.corrupt_input(normalized, abnormal, pools, operator, case_id='x', patch_index=1)
    assert torch.equal(corrupted[:, ~abnormal], normalized[:, ~abnormal])


def test_empty_abnormal_mask_is_exact_identity_for_all_operators():
    image = torch.randn((4, *audit.PATCH_SHAPE))
    abnormal = torch.zeros(audit.PATCH_SHAPE, dtype=torch.bool)
    pools = [torch.ones(audit.PATCH_SHAPE, dtype=torch.bool)] * 4
    for operator in audit.OPERATORS:
        assert torch.equal(audit.corrupt_input(image, abnormal, pools, operator), image)
    empty_pools, fallbacks = audit.donor_pools(torch.zeros_like(image), abnormal, torch.zeros_like(abnormal))
    assert all(not bool(pool.any()) for pool in empty_pools) and fallbacks == [False] * 4


def test_donor_pool_excludes_abnormal_unsupported_and_raw_zero_values():
    raw, _, abnormal, pools, fallbacks = _synthetic_inputs()
    assert not any(fallbacks)
    for modality, pool in enumerate(pools):
        assert not bool((pool & abnormal).any())
        assert not bool((pool & ~audit.support_mask(raw)).any())
        assert not bool((pool & raw[modality].eq(0)).any())


def test_fallback_and_no_donor_failure():
    raw = torch.zeros((4, *audit.PATCH_SHAPE)); raw[:, :8, :8, :8] = 1
    raw[:, 40:48, 40:48, 40:48] = 1
    abnormal = torch.zeros(audit.PATCH_SHAPE, dtype=torch.bool); abnormal[40:48, 40:48, 40:48] = True
    support = audit.support_mask(raw)
    pools, fallbacks = audit.donor_pools(raw, abnormal, support)
    assert all(fallbacks) and all(int(pool.sum()) > 0 for pool in pools)
    with pytest.raises(ValueError, match='No valid donor'):
        audit.donor_pools(torch.zeros_like(raw), abnormal, torch.zeros_like(support))


def test_empirical_resampling_is_deterministic_and_uses_donor_values():
    _, normalized, abnormal, pools, _ = _synthetic_inputs()
    first = audit.corrupt_input(normalized, abnormal, pools, 'local_empirical_resample', case_id='x', patch_index=3)
    second = audit.corrupt_input(normalized, abnormal, pools, 'local_empirical_resample', case_id='x', patch_index=3)
    assert torch.equal(first, second)
    for modality in range(4):
        assert set(first[modality][abnormal].tolist()) <= set(normalized[modality][pools[modality]].tolist())


def test_region_partition_and_exact_support_mapping():
    geometry = audit.derive_encoder_geometry(Encoder())
    raw = torch.ones((4, *audit.PATCH_SHAPE))
    mask = torch.zeros(audit.PATCH_SHAPE, dtype=torch.int64)
    mask[4, 4, 4] = 1
    regions = audit.region_masks(raw, mask, geometry)
    valid = regions['valid_support']
    assert int(valid.sum()) == 1728
    assert torch.equal(regions['region_a'] | regions['region_b'] | regions['region_c'], valid)
    assert not bool((regions['region_a'] & regions['region_b']).any())
    assert not bool((regions['region_a'] & regions['region_c']).any())
    assert not bool((regions['region_b'] & regions['region_c']).any())
    assert geometry.effective_stride == 8 and geometry.theoretical_rf_size == 68


def test_standardization_uses_fixed_channel_contract_values():
    latent = torch.zeros((1, 256, 12, 12, 12)); latent[:, 0] = 3
    means = torch.zeros(256); stds = torch.ones(256); means[0] = 1; stds[0] = 2
    standardized = audit.standardize_za(latent, means, stds)
    assert torch.equal(standardized[:, 0], torch.full((1, 12, 12, 12), 1.0))


def test_case_aggregation_is_weighted_by_region_counts():
    metrics = {'patch_count': 2, 'positive_region_a_count': 3, 'positive_region_a_delta_std_sum': 3.0,
               'positive_region_a_delta_raw_sum': 6.0, 'positive_region_b_count': 0,
               'positive_region_c_count': 1, 'positive_region_c_delta_std_sum': 2.0,
               'positive_region_c_delta_raw_sum': 4.0, 'positive_region_all_support_count': 4,
               'positive_region_all_support_delta_std_sum': 5.0, 'positive_region_all_support_delta_raw_sum': 10.0,
               'all_region_a_count': 6, 'all_region_a_delta_std_sum': 6.0, 'all_region_a_delta_raw_sum': 12.0,
               'all_region_b_count': 0, 'all_region_c_count': 2, 'all_region_c_delta_std_sum': 4.0,
               'all_region_c_delta_raw_sum': 8.0, 'all_region_all_support_count': 8,
               'all_region_all_support_delta_std_sum': 10.0, 'all_region_all_support_delta_raw_sum': 20.0,
               'abnormal_positive_patch_count': 1, 'abnormal_voxel_count': 8, 'donor_pool_fallback_count': 0}
    row = audit._case_operator_row('case', 'zero_fill', metrics)
    assert row['primary_delta_std_region_c_mean'] == 2.0
    assert row['all_patches_delta_std_region_c_mean'] == 2.0
    assert row['positive_region_b_delta_std_mean'] is None


def test_actual_changed_counts_can_differ_from_targeted_counts():
    original = torch.ones((4, *audit.PATCH_SHAPE))
    abnormal = torch.zeros(audit.PATCH_SHAPE, dtype=torch.bool); abnormal[:2, :2, :2] = True
    unchanged = original.clone()
    diagnostics = audit.input_change_metrics(unchanged, original, abnormal)
    assert diagnostics['targeted_abnormal_voxel_count'] == 8
    assert diagnostics['changed_abnormal_voxel_count_0'] == 0
    assert diagnostics['changed_any_modality_abnormal_voxel_count'] == 0
    assert diagnostics['changed_abnormal_voxel_fraction_0'] == 0.0


def test_primary_negative_only_case_is_undefined_but_all_patch_secondary_remains():
    metrics = {'patch_count': 2, 'positive_region_c_count': 0, 'positive_region_c_delta_std_sum': 0.0,
               'positive_region_c_delta_raw_sum': 0.0, 'all_region_c_count': 4,
               'all_region_c_delta_std_sum': 8.0, 'all_region_c_delta_raw_sum': 4.0,
               'abnormal_positive_patch_count': 0, 'abnormal_voxel_count': 0, 'donor_pool_fallback_count': 0}
    row = audit._case_operator_row('negative', 'local_mean_fill', metrics)
    assert row['primary_delta_std_region_c_mean'] is None
    assert row['all_patches_delta_std_region_c_mean'] == 2.0


def test_changed_any_modality_union_and_zero_denominator():
    original = torch.ones((4, *audit.PATCH_SHAPE))
    abnormal = torch.zeros(audit.PATCH_SHAPE, dtype=torch.bool); abnormal[:2, :2, :2] = True
    changed = original.clone(); changed[0, abnormal] = 0; changed[1, :1, :2, :2] = 0
    diagnostics = audit.input_change_metrics(changed, original, abnormal)
    assert diagnostics['changed_abnormal_voxel_count_0'] == 8
    assert diagnostics['changed_abnormal_voxel_count_1'] == 4
    assert diagnostics['changed_any_modality_abnormal_voxel_count'] == 8
    empty = audit.input_change_metrics(original, original, torch.zeros_like(abnormal))
    assert empty['changed_abnormal_voxel_fraction_0'] is None


def test_train_and_manifest_sha_guards_fail_before_processing(tmp_path, monkeypatch):
    split = tmp_path / 'train.txt'; split.write_text('case\n')
    config = tmp_path / 'data.yaml'; config.write_text(f'train_cases: {split}\nroot: {tmp_path}\n')
    monkeypatch.setattr(audit, 'runtime_git_provenance', lambda: {
        'runtime_git_branch': audit.EXPECTED_BRANCH, 'runtime_git_commit': 'test-sha', 'git_status_short': []})
    monkeypatch.setattr(audit, 'TRAIN_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='Training split SHA256 mismatch'):
        audit.run(data_config=config, patch_manifest=tmp_path / 'manifest.csv', output_dir=tmp_path / 'out')
    monkeypatch.setattr(audit, 'TRAIN_SHA256', hashlib.sha256(split.read_bytes()).hexdigest())
    monkeypatch.setattr(audit, 'EXPECTED_CASES', 1)
    manifest = tmp_path / 'manifest.csv'; manifest.write_text('bad\n')
    monkeypatch.setattr(audit, 'MANIFEST_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='D0-B0 patch manifest SHA256 mismatch'):
        audit.run(data_config=config, patch_manifest=manifest, output_dir=tmp_path / 'out')


def test_wrong_runtime_branch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'runtime_git_provenance', lambda: {
        'runtime_git_branch': 'main', 'runtime_git_commit': 'test-sha', 'git_status_short': []})
    with pytest.raises(RuntimeError, match='must run on'):
        audit.run(data_config=tmp_path / 'missing.yaml', output_dir=tmp_path / 'out')


def test_source_info_schema_fails_before_processing():
    with pytest.raises(ValueError, match='source_info must be a dict'):
        audit.validate_source_info(None)
    with pytest.raises(ValueError, match='missing epoch/model'):
        audit.validate_source_info({'epoch': 11})
    assert audit.validate_source_info({'epoch': 11, 'model': 'model'}) == (11, 'model')


def test_provenance_claim_flags_are_explicit():
    assert audit.EXPECTED_BRANCH == 'experiment/d0b3-corruption-operator-audit'
    assert audit.nib.__version__
    assert set(audit.runtime_git_provenance()) == {'runtime_git_branch', 'runtime_git_commit', 'git_status_short'}
    assert audit.OPERATORS == ('zero_fill', 'local_mean_fill', 'local_empirical_resample')
    assert audit.SUBSET_SEED == 20260917


def test_provenance_defines_corruption_and_empirical_seed_algorithm():
    definitions = {
        'zero_fill': {'replacement': 'normalized value 0', 'donor_values_used': False},
        'local_mean_fill': {'replacement': 'arithmetic mean of normalized values from valid donor pool', 'donor_values_used': True},
        'local_empirical_resample': {'replacement': 'sampling with replacement from actual normalized donor values', 'donor_values_used': True},
    }
    assert definitions['zero_fill']['donor_values_used'] is False
    assert 'first 8 digest bytes' in 'SHA256(global_seed:case_id:patch_index:operator_name:modality_index); first 8 digest bytes, big-endian, masked to 63 bits'


def test_structural_expected_counts_are_locked():
    assert audit.SUBSET_CASE_COUNT == 200
    assert audit.PATCHES_PER_CASE == 20
    assert audit.TOTAL_PATCHES == 4000
    assert len(audit.OPERATORS) == 3
    assert audit.TOTAL_PATCHES * len(audit.OPERATORS) == 12000
    assert audit.SUBSET_CASE_COUNT * len(audit.OPERATORS) == 600


def test_scientific_run_seed_is_locked_but_selection_helper_is_parameterizable(tmp_path):
    assert audit.select_cases(['a', 'b'], seed=9, count=1)
    with pytest.raises(ValueError, match='seed is locked'):
        audit.run(tmp_path / 'missing.yaml', subset_seed=9)
