"""Synthetic train-only D0-B0 contracts; no scientific volumes or model required."""
import builtins
import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np
import pytest
import torch

from datasets.patch_sampling import PatchSampler
from scripts import audit_d0b_support_mask as script


@pytest.mark.parametrize('filled', [False, True])
def test_uniform_support(filled):
    support = torch.full((96, 96, 96), filled, dtype=torch.bool)
    occupancy = script.latent_occupancy(support)
    assert occupancy.shape == (12, 12, 12) and occupancy.dtype == torch.float64
    assert torch.all(occupancy == int(filled))
    metrics = script.patch_statistics(support)
    assert metrics['input_support_fraction'] == metrics['latent_occupancy_mean'] == int(filled)
    assert metrics['latent_occupancy_std'] == 0
    assert all(metrics[field] == int(filled) for field in script.FRACTION_FIELDS)


@pytest.mark.parametrize('depth,value', [(8, 1.0), (4, 0.5)])
def test_exact_block_reduction(depth, value):
    support = torch.zeros((96, 96, 96), dtype=torch.bool)
    support[16:16+depth, 24:32, 32:40] = True
    occupancy = script.latent_occupancy(support)
    assert occupancy[2, 3, 4] == value
    assert (occupancy != 0).sum() == 1
    assert occupancy.min() >= 0 and occupancy.max() <= 1
    assert occupancy.mean() == support.double().mean()


def test_candidate_threshold_fractions_and_population_std():
    counts = [0, 51, 52, 128, 256, 384, 460, 461, 512]
    support = torch.zeros((96, 96, 96), dtype=torch.bool)
    for index, count in enumerate(counts):
        support[:8, :8, index*8:(index+1)*8] = (torch.arange(512) < count).reshape(8, 8, 8)
    values = np.zeros(1728)
    values[:len(counts)] = np.asarray(counts) / 512
    metrics = script.patch_statistics(support)
    assert metrics['latent_occupancy_std'] == pytest.approx(values.std(ddof=0))
    for field, threshold in zip(script.FRACTION_FIELDS, script.THRESHOLDS):
        assert metrics[field] == np.mean(values >= threshold)
    assert metrics['input_support_fraction'] == metrics['latent_occupancy_mean']


@pytest.mark.parametrize('shape,dtype', [((95, 96, 96), torch.bool), ((96, 96), torch.bool),
                                       ((96, 96, 96), torch.float32)])
def test_malformed_support_fails(shape, dtype):
    with pytest.raises(ValueError, match='boolean support'):
        script.latent_occupancy(torch.zeros(shape, dtype=dtype))


def test_raw_modality_union_including_negative_and_tiny_values():
    image = torch.zeros(4, 96, 96, 96)
    for channel, value in enumerate((1, -1, 1e-20, 8)):
        image[channel, channel, 0, 0] = value
    support = script.image_support(image)
    assert support.sum() == 4
    assert all(support[channel, 0, 0] for channel in range(4))
    image[0, 0, 1, 1] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        script.image_support(image)
    with pytest.raises(ValueError, match='Expected raw'):
        script.image_support(torch.zeros(3, 96, 96, 96))


def test_shared_raw_loader_mask_cannot_affect_support(tmp_path, monkeypatch):
    folder = tmp_path / 'train-case'
    folder.mkdir()
    images = {}
    for index, key in enumerate((*script.MODALITIES, 'mask')):
        filename = f'train-case-{key.lower() if key != "mask" else "seg"}.nii.gz'
        (folder / filename).touch()
        array = np.zeros((96, 96, 96), dtype=np.float32)
        array[index, 0, 0] = index + 1
        images[filename] = nib.Nifti1Image(array, np.eye(4))
    monkeypatch.setattr(nib, 'load', lambda path: images[Path(path).name])
    first = script.image_support(script.load_raw_image(tmp_path, 'train-case'))
    images['train-case-seg.nii.gz'] = nib.Nifti1Image(np.full((96,96,96), 123, dtype=np.float32), np.eye(4))
    second = script.image_support(script.load_raw_image(tmp_path, 'train-case'))
    assert torch.equal(first, second) and first.sum() == 4
    bad_affine = np.eye(4)
    bad_affine[0, 3] = 5
    images['train-case-t2f.nii.gz'] = nib.Nifti1Image(np.zeros((96,96,96), dtype=np.float32), bad_affine)
    with pytest.raises(ValueError, match='affine/grid'):
        script.load_raw_image(tmp_path, 'train-case')


def test_seed_stability_random_sampler_equivalence_and_no_filtering(monkeypatch):
    support = torch.zeros((98, 100, 102), dtype=torch.bool)
    expected = int.from_bytes(hashlib.sha256(b'[20260917,"case",0]').digest()[:8], 'big') & ((1 << 63)-1)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, 'hash', lambda *_: pytest.fail('Python hash used'))
        assert script.stable_seed(20260917, 'case', 0) == expected
    assert script.stable_seed(20260917, 'case', 1) != expected
    first = list(script.random_patches(support, 'case', 20260917))
    torch.manual_seed(19)
    second = list(script.random_patches(~support, 'case', 20260917))
    assert len(first) == len(second) == 20
    assert len({row['seed'] for row, _ in first}) == 20
    for (row, patch), (other, _) in zip(first, second):
        assert row == other  # Neither support values nor global RNG affect coordinates.
        assert patch.shape == (96,96,96) and not patch.any()
        original = PatchSampler((96,96,96), 0, torch.Generator().manual_seed(row['seed']))
        slices = original.sample_slices(support)
        assert tuple(row[f'crop_start_{axis}'] for axis in ('d','h','w')) == tuple(s.start for s in slices)
    fixed = list(script.random_patches(torch.zeros((96,96,96), dtype=torch.bool), 'case', 1))
    assert len(fixed) == 20  # All 20 identical coordinates are retained.
    assert all(row['crop_start_d'] == row['crop_start_h'] == row['crop_start_w'] == 0 for row, _ in fixed)
    assert script.EXPECTED_CASES * script.PATCHES_PER_CASE == 21600


def test_train_split_guards(tmp_path, monkeypatch):
    path = tmp_path / 'train.txt'
    path.write_text('\n'.join(f'case-{i}' for i in range(1080)))
    with pytest.raises(ValueError, match='SHA256'):
        script.train_ids(path)
    monkeypatch.setattr(script, 'sha256', lambda path: script.TRAIN_SHA256)
    assert len(script.train_ids(path)) == 1080
    path.write_text('a\na\n')
    with pytest.raises(ValueError, match='Duplicate'):
        script.train_ids(path)
    path.write_text('a\nb\n')
    with pytest.raises(ValueError, match='1080'):
        script.train_ids(path)


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    split = tmp_path / 'train.txt'
    split.write_text('a\nb\n')
    config = tmp_path / 'data.yaml'
    config.write_text(f'root: {tmp_path / "dataset"}\ntrain_cases: {split}\nval_cases: MUST_NOT_READ\nnormalize: true\naugmentation:\n  enabled: true\n')
    monkeypatch.setattr(script, 'EXPECTED_CASES', 2)
    real_sha = script.sha256
    monkeypatch.setattr(script, 'sha256', lambda path: script.TRAIN_SHA256 if Path(path) == split else real_sha(path))
    loader = MagicMock(side_effect=lambda root, case: torch.full((4,96,96,96), float(case == 'b')))
    monkeypatch.setattr(script, 'load_raw_image', loader)
    return config, tmp_path / 'audit', loader


def test_outputs_aggregation_provenance_and_reproducibility(setup_run):
    config, output, loader = setup_run
    assert script.run(config, output) == output
    assert [call.args[1] for call in loader.call_args_list] == ['a','b']
    assert {p.name for p in output.iterdir()} == set(script.ARTIFACTS)
    patches = list(csv.DictReader((output / 'patch_support_statistics.csv').open()))
    cases = list(csv.DictReader((output / 'case_support_statistics.csv').open()))
    manifest = list(csv.DictReader((output / 'patch_manifest.csv').open()))
    assert len(patches) == len(manifest) == 40 and len(cases) == 2
    assert len({(p['case_id'], p['patch_index']) for p in manifest}) == 40
    assert all({key: row[key] for key in script.IDENTITY_FIELDS} == identity for row, identity in zip(patches, manifest))
    assert {field for field in script.METRICS} <= patches[0].keys()
    for index, case in enumerate(cases):
        assert int(case['patch_count']) == 20
        for field in ('input_support_fraction', 'latent_occupancy_mean'):
            for suffix in ('mean','median','min','max'):
                assert float(case[f'{field}_{suffix}']) == index
    summary = json.loads((output / 'support_mask_summary.json').read_text())
    assert summary['case_count'] == 2 and summary['total_patch_count'] == 40
    assert summary['patch_shape'] == [96]*3 and summary['latent_block_shape'] == [8]*3
    for field in script.METRICS:
        dist = summary['empirical_patch_distributions'][field]
        assert dist['mean'] == dist['p50'] == 0.5
        assert dist['p01'] == 0 and dist['p99'] == 1
    provenance = json.loads((output / 'analysis_provenance.json').read_text())
    assert provenance['analysis_identity'] == 'D0-B0 latent support-mask audit'
    assert provenance['experiment_base_commit'] == script.EXPERIMENT_BASE_COMMIT
    assert provenance['train_split_sha256'] == script.TRAIN_SHA256
    for field in ('validation_data_used', 'augmentation_used', 'rejection_sampling_used',
                  'segmentation_used_for_support', 'segmentation_used_for_sampling',
                  'model_training', 'optimizer_used', 'backward_pass', 'e1c_forward_used', 'support_threshold_selected'):
        assert provenance[field] is False
    assert provenance['global_audit_seed'] == 20260917
    assert provenance['patch_manifest_sha256'] == script.sha256(output / 'patch_manifest.csv')
    other = output.with_name('repeat')
    script.run(config, other)
    for filename in set(script.ARTIFACTS) - {'analysis_provenance.json'}:
        assert (output / filename).read_bytes() == (other / filename).read_bytes()


def test_overwrite_safety_and_failure_preserves_outputs(setup_run):
    config, output, loader = setup_run
    script.run(config, output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        script.run(config, output)
    loader.side_effect = ValueError('synthetic input failure')
    with pytest.raises(ValueError, match='synthetic'):
        script.run(config, output, overwrite=True)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match='outputs/frozen'):
        script.run(config, 'outputs/frozen/forbidden', overwrite=True)
    (output / 'unrelated').write_text('keep')
    with pytest.raises(ValueError, match='unrelated'):
        script.run(config, output, overwrite=True)


def test_cli(monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(script, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['audit'])
    script.main()
    run.assert_called_once_with(data_config='configs/data.yaml', output_dir='outputs/d0b_support_mask_audit',
                                audit_seed=20260917, overwrite=False)
