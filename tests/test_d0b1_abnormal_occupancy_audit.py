"""Synthetic D0-B1 tests; no BraTS data or scientific manifest required."""
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from scripts import audit_d0b1_abnormal_occupancy as audit


def test_occupancy_values_and_threshold_semantics():
    mask = torch.zeros(audit.PATCH_SHAPE, dtype=torch.bool)
    mask[:4, :8, :8] = True
    occupancy = audit.block_occupancy(mask)
    assert occupancy[0, 0, 0] == 0.5 and occupancy.sum() == 0.5
    assert occupancy.min() >= 0 and occupancy.max() <= 1

    segmentation = torch.zeros(audit.PATCH_SHAPE, dtype=torch.int64)
    segmentation[:4, :8, :8] = 1
    image = torch.zeros((4, *audit.PATCH_SHAPE))
    image[:, :8, :8, :8] = 1
    metrics = audit.patch_metrics(image, segmentation)
    assert metrics['abnormal_positive_patch'] is True
    assert metrics['fraction_all_latent_abnormal_gt_0'] == pytest.approx(1 / 1728)
    assert metrics['fraction_all_latent_abnormal_ge_0_50'] == pytest.approx(1 / 1728)
    assert metrics['valid_support_latent_count'] == 1
    assert metrics['fraction_valid_support_excluded_ge_0_50'] == 1


def test_patch_std_uses_population_convention():
    image = torch.ones((4, *audit.PATCH_SHAPE))
    segmentation = torch.zeros(audit.PATCH_SHAPE, dtype=torch.int64)
    segmentation[:8, :8, :8] = 1
    metrics = audit.patch_metrics(image, segmentation)
    expected = np.std([1.0] + [0.0] * 1727, ddof=0)
    assert metrics['latent_abnormal_occupancy_std'] == pytest.approx(expected)


@pytest.mark.parametrize('label', [1, 2, 3, 4])
def test_all_abnormal_labels_count(label):
    segmentation = torch.full(audit.PATCH_SHAPE, label, dtype=torch.int64)
    image = torch.zeros((4, *audit.PATCH_SHAPE))
    metrics = audit.patch_metrics(image, segmentation)
    assert metrics['input_abnormal_fraction'] == 1
    assert metrics['abnormal_positive_patch'] is True


def test_zero_and_one_abnormal_and_zero_valid_support():
    image = torch.zeros((4, *audit.PATCH_SHAPE))
    zero = audit.patch_metrics(image, torch.zeros(audit.PATCH_SHAPE, dtype=torch.int64))
    assert zero['abnormal_positive_patch'] is False
    assert zero['valid_support_latent_count'] == 0
    assert zero['fraction_valid_support_excluded_gt_0'] is None
    assert zero['valid_support_zero_denominator'] is True
    one = audit.patch_metrics(image + 1, torch.ones(audit.PATCH_SHAPE, dtype=torch.int64))
    assert one['latent_abnormal_occupancy_min'] == one['latent_abnormal_occupancy_max'] == 1
    assert one['fraction_all_latent_abnormal_ge_0_50'] == 1


def test_segmentation_unknown_and_shape_rejected():
    image = torch.zeros((4, *audit.PATCH_SHAPE))
    bad = torch.full(audit.PATCH_SHAPE, 9, dtype=torch.int64)
    with pytest.raises(ValueError, match='Unknown'):
        audit.patch_metrics(image, bad)
    with pytest.raises(ValueError, match='segmentation patch'):
        audit.patch_metrics(image, bad[:10])


def test_manifest_schema_and_integrity_guards(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.csv'
    fields = list(audit.MANIFEST_FIELDS)
    rows = []
    cases = [f'case-{i}' for i in range(1080)]
    for case in cases:
        for index in range(20):
            rows.append(dict(case_id=case, patch_index=index, seed=index,
                             crop_start_d=0, crop_start_h=0, crop_start_w=0,
                             crop_stop_d=96, crop_stop_h=96, crop_stop_w=96))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    monkeypatch.setattr(audit, 'MANIFEST_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    loaded = audit.read_manifest(path, set(cases))
    assert len(loaded) == 21600
    path.write_text(path.read_text().replace('case-0,0', 'case-1,0', 1))
    with pytest.raises(ValueError, match='SHA256'):
        audit.read_manifest(path, set(cases))


def test_staged_run_reuses_coordinates_and_provenance(tmp_path, monkeypatch):
    split = tmp_path / 'train.txt'; split.write_text('a\nb\n')
    config = tmp_path / 'data.yaml'; config.write_text(f'root: {tmp_path / "root"}\ntrain_cases: {split}\n')
    manifest = tmp_path / 'manifest.csv'
    fields = list(audit.MANIFEST_FIELDS)
    rows = [dict(case_id=case, patch_index=index, seed=0, crop_start_d=0,
                 crop_start_h=0, crop_start_w=0, crop_stop_d=96,
                 crop_stop_h=96, crop_stop_w=96) for case in ('a', 'b') for index in range(20)]
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    monkeypatch.setattr(audit, 'EXPECTED_CASES', 2)
    monkeypatch.setattr(audit, 'TOTAL_PATCHES', 40)
    monkeypatch.setattr(audit, 'TRAIN_SHA256', hashlib.sha256(split.read_bytes()).hexdigest())
    monkeypatch.setattr(audit, 'MANIFEST_SHA256', hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(audit, 'load_case', lambda root, case: (torch.zeros((4,96,96,96)), torch.zeros((96,96,96), dtype=torch.int64)))
    output = tmp_path / 'output'
    audit.run(config, manifest, output)
    assert len(list(csv.DictReader((output / 'patch_abnormal_statistics.csv').open()))) == 40
    assert len(list(csv.DictReader((output / 'case_abnormal_statistics.csv').open()))) == 2
    provenance = json.loads((output / 'analysis_provenance.json').read_text())
    assert provenance['validation_data_used'] is False
    assert provenance['new_random_sampling_performed'] is False
    assert provenance['final_abnormal_threshold_selected'] is False
    assert provenance['experiment_base_commit'] == audit.EXPERIMENT_BASE_COMMIT
    assert provenance['data_config_path'] == str(config.resolve())
    assert provenance['data_config_sha256'] == audit.sha256(config)
    assert provenance['dataset_root'] == str((tmp_path / 'root').resolve())
    summary = json.loads((output / 'abnormal_occupancy_summary.json').read_text())
    assert summary['abnormal_positive_patch_count'] == 0
    case_row = next(csv.DictReader((output / 'case_abnormal_statistics.csv').open()))
    assert 'fraction_all_latent_abnormal_ge_0_50_mean' in case_row
    assert 'fraction_valid_support_excluded_ge_0_50_defined_patch_count' in case_row
    assert case_row['fraction_valid_support_excluded_ge_0_50_mean'] == ''
    assert case_row['fraction_valid_support_excluded_ge_0_50_defined_patch_count'] == '0'
    with pytest.raises(FileExistsError):
        audit.run(config, manifest, output)


def test_cli_and_no_model_loading(monkeypatch):
    import sys
    run = MagicMock()
    monkeypatch.setattr(audit, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['audit', '--patch-manifest', 'm.csv', '--overwrite'])
    audit.main()
    assert run.call_args.kwargs['patch_manifest'] == 'm.csv'
    assert run.call_args.kwargs['overwrite'] is True


def test_single_voxel_uses_strict_positive_vs_ge_001_semantics():
    image = torch.zeros((4, *audit.PATCH_SHAPE))
    image[:, :4, :8, :8] = 1  # support occupancy is exactly 0.50 in one cell
    segmentation = torch.zeros(audit.PATCH_SHAPE, dtype=torch.int64)
    segmentation[0, 0, 0] = 1
    metrics = audit.patch_metrics(image, segmentation)
    assert metrics['latent_abnormal_occupancy_max'] == pytest.approx(1 / 512)
    assert metrics['fraction_all_latent_abnormal_gt_0'] == pytest.approx(1 / 1728)
    assert metrics['fraction_all_latent_abnormal_ge_0_01'] == 0
    assert metrics['valid_support_latent_count'] == 1
    assert metrics['fraction_valid_support_excluded_gt_0'] == 1
    assert metrics['fraction_valid_support_excluded_ge_0_01'] == 0


def test_train_split_sha_failure_precedes_scientific_processing(tmp_path, monkeypatch):
    split = tmp_path / 'train.txt'
    split.write_text('syntactically-valid-case\n')
    config = tmp_path / 'data.yaml'
    config.write_text(f'root: {tmp_path / "root"}\ntrain_cases: {split}\n')
    load_case = MagicMock(side_effect=AssertionError('scientific processing was reached'))
    monkeypatch.setattr(audit, 'load_case', load_case)
    with pytest.raises(ValueError, match='Training split SHA256 mismatch'):
        audit.run(config, patch_manifest=tmp_path / 'unused.csv', output_dir=tmp_path / 'output')
    load_case.assert_not_called()
    assert not (tmp_path / 'output').exists()
