"""Synthetic D0-A contracts; never load scientific volumes/checkpoints."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn

from analysis import latent_distribution_audit as stats
from models import latent_probe
from models.anatomy_constrained_joint_reconstruction_unet3d import AnatomyConstrainedJointReconstructionUNet3D
from scripts import audit_e1c_latent_distribution as script


def test_streaming_population_moments():
    values = np.array([[1e9 + 1, 1e9 + 2, 1e9 + 8], [-5, 0, 7]], dtype=np.float64)
    accumulator = stats.Moments(2)
    accumulator.update(values[:, :1])
    accumulator.update(values[:, 1:])
    for channel in range(2):
        row = accumulator.row(channel)
        assert row['count'] == 3
        assert row['mean'] == pytest.approx(values[channel].mean())
        assert row['std'] == pytest.approx(values[channel].std(ddof=0))
        assert row['minimum'] == values[channel].min()
        assert row['maximum'] == values[channel].max()


def synthetic_samples(values, ids=('a', 'b')):
    return [(metadata, values[case_index, metadata['patch_index_within_case']])
            for case_index, case in enumerate(ids) for metadata in script.patch_plan([case], script.AUDIT_SEED)]


def test_exact_disk_percentiles_and_statistics(tmp_path):
    values = np.array([[[[-8, 0, 2]], [[1, 3, 7]]],
                       [[[-2, 4, 9]], [[-9, 0, 11]]]], dtype=np.float32)
    values = np.stack([values + i for i in range(10)], axis=1)
    # [cases, patches, channels, 1, 3]; all ten patches enter each case aggregate.
    global_stats, summary = stats.audit_latents(synthetic_samples(values), ['a', 'b'], tmp_path, (2, 1, 3))
    assert global_stats['count'] == values.size
    assert global_stats['mean'] == pytest.approx(values.mean(dtype=np.float64))
    assert global_stats['std'] == pytest.approx(values.astype(np.float64).std())
    assert [global_stats[f'p{p:02d}'] for p in stats.PERCENTILES] == pytest.approx(
        np.percentile(values.astype(np.float64), stats.PERCENTILES))
    channels = list(csv.DictReader((tmp_path / 'latent_channel_statistics.csv').open()))
    cases = list(csv.DictReader((tmp_path / 'latent_case_statistics.csv').open()))
    for index, row in enumerate(channels):
        channel = values[:, :, index].astype(np.float64)
        assert int(row['channel_index']) == index and int(row['count']) == channel.size
        assert float(row['std']) == pytest.approx(channel.std())
        assert float(row['mean']) == pytest.approx(channel.mean())
        assert [float(row[f'p{p:02d}']) for p in stats.PERCENTILES] == pytest.approx(
            np.percentile(channel, stats.PERCENTILES))
    assert len(cases) == 2
    patches = list(csv.DictReader((tmp_path / 'latent_patch_statistics.csv').open()))
    assert len(patches) == 20
    for row, value in zip(patches, values.reshape(20, 2, 1, 3)):
        expected = stats.distribution_row(value)
        for key, result in expected.items():
            assert float(row[key]) == pytest.approx(result)
    for index, row in enumerate(cases):
        case = values[index].astype(np.float64)
        assert int(row['latent_element_count']) == case.size
        assert float(row['RMS']) == pytest.approx(np.sqrt(np.mean(case ** 2)))
        assert float(row['mean_absolute_value']) == pytest.approx(np.abs(case).mean())
        assert float(row['std']) == pytest.approx(case.std())
    rms = [np.sqrt(np.mean(v.astype(np.float64) ** 2)) for v in values]
    assert summary['total_patch_count'] == 20
    assert summary['per_case_RMS'] == stats.summarize(rms)
    assert summary['per_case_standard_deviation'] == stats.summarize([v.astype(np.float64).std() for v in values])
    assert not list(tmp_path.glob('.d0-scratch-*'))
    second = tmp_path / 'second'
    second.mkdir()
    assert stats.audit_latents(synthetic_samples(values), ['a', 'b'], second, (2, 1, 3)) == (global_stats, summary)


@pytest.mark.parametrize('values', [[-0., 0., -1e30, 1e30, 1e-30, -1e-30], [3.] * 20, [-7.]])
def test_percentile_order_statistics_extremes(values):
    values = np.asarray(values, dtype=np.float32)
    keys = np.sort(stats.ordered_keys(values))[None, :]
    expected = np.percentile(values.astype(np.float64), stats.PERCENTILES)
    assert list(stats.exact_percentiles(keys).values()) == pytest.approx(expected)
    assert list(stats.exact_percentiles(keys, 0).values()) == pytest.approx(expected)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match='Non-finite latent'):
        stats.audit_latents([(next(script.patch_plan(['a'], 1)), np.full((1, 1, 1), bad, dtype=np.float32))], ['a'], tmp_path, (1, 1, 1))
    assert not list(tmp_path.glob('.d0-scratch-*'))


class FakeSource(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.encoder = MagicMock(side_effect=self.encode)
        self.projection = SimpleNamespace(anatomy_projector=MagicMock(side_effect=lambda x: x + 2))
        self.decoder = self.reconstruction_decoder = self.anatomy_decoder = MagicMock(
            side_effect=AssertionError('Decoder used'))
        self.requires_grad_(False).eval()

    def encode(self, image):
        assert not torch.is_grad_enabled()
        return [torch.zeros(1), torch.ones((1, *script.LATENT_SHAPE))]


def test_extraction_correct_frozen_eval_no_decoders():
    source = FakeSource()
    image = torch.zeros(1, 4, 96, 96, 96)
    latent = script.extract_anatomy(source, image)
    assert latent.shape == (256, 12, 12, 12)
    assert np.all(latent == 3)
    source.decoder.assert_not_called()
    assert not source.training and not source.weight.requires_grad and source.weight.grad is None
    source.weight.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen'):
        script.extract_anatomy(source, image)
    source.requires_grad_(False).train()
    with pytest.raises(ValueError, match='eval mode'):
        script.extract_anatomy(source, image)


@pytest.mark.parametrize('kind', ['shape', 'nan'])
def test_extraction_rejects_invalid_latent(kind):
    source = FakeSource()
    bad = torch.zeros(1, 256, 1, 1, 1) if kind == 'shape' else torch.full((1, *script.LATENT_SHAPE), float('nan'))
    source.projection.anatomy_projector.side_effect = lambda x: bad
    with pytest.raises(ValueError, match='shape|Non-finite'):
        script.extract_anatomy(source, torch.zeros(1, 4, 96, 96, 96))


@pytest.mark.parametrize('mutation', ['sha', 'epoch', 'identity', 'experiment', 'state'])
def test_real_source_loader_enforcement(tmp_path, monkeypatch, mutation):
    path = tmp_path / 'source.pt'
    path.write_bytes(b'synthetic')
    config = {**latent_probe.E1C_SOURCE_MODEL, 'base_channels': 2, 'latent_channels': 16}
    monkeypatch.setitem(latent_probe.SOURCE_SPECS, 'e1c', {**latent_probe.SOURCE_SPECS['e1c'], 'model': config})
    source = AnatomyConstrainedJointReconstructionUNet3D(base_channels=2, latent_channels=16)
    state = dict(epoch=11, model=source.state_dict(), config=dict(model=dict(config),
                 experiment_name=latent_probe.E1C_SOURCE_EXPERIMENT))
    monkeypatch.setattr(latent_probe, 'file_sha256', lambda p: 'bad' if mutation == 'sha' else latent_probe.E1C_SOURCE_SHA256)
    if mutation == 'epoch':
        state['epoch'] = 12
    elif mutation == 'identity':
        state['config']['model']['name'] = 'wrong'
    elif mutation == 'experiment':
        state['config']['experiment_name'] = 'wrong'
    elif mutation == 'state':
        state['model'].pop(next(iter(state['model'])))
    monkeypatch.setattr(torch, 'load', MagicMock(return_value=state))
    with pytest.raises((ValueError, RuntimeError)):
        script.load_source(path, source_experiment='e1c', expected_sha256=latent_probe.E1C_SOURCE_SHA256, expected_epoch=11)


def test_train_split_guards(tmp_path, monkeypatch):
    path = tmp_path / 'train.txt'
    path.write_text('\n'.join(f'case-{i}' for i in range(1080)))
    with pytest.raises(ValueError, match='SHA256'):
        script.validate_train_split(path)
    monkeypatch.setattr(script, 'file_sha256', lambda p: script.TRAIN_SHA256)
    assert len(script.validate_train_split(path)) == 1080
    path.write_text('a\na\n')
    with pytest.raises(ValueError, match='unique IDs'):
        script.validate_train_split(path)
    path.write_text('a\nb\n')
    with pytest.raises(ValueError, match='1080'):
        script.validate_train_split(path)


def test_train_only_manifest(monkeypatch):
    calls = []
    def dataset(**kwargs):
        assert kwargs['training'] is False and kwargs['normalize'] is True
        rows = list(csv.DictReader(Path(kwargs['manifest']).open()))
        assert [r['subject_id'] for r in rows] == ['train-only']
        calls.append(rows)
        return SimpleNamespace(records=[{'subject_id': 'train-only'}])
    monkeypatch.setattr(script, 'BraTSDataset', dataset)
    script.make_train_dataset(dict(root='unused', patch_size=[96]*3, val_cases='DO_NOT_OPEN'), ['train-only'])
    assert len(calls) == 1


@pytest.fixture
def run_setup(tmp_path, monkeypatch):
    source_dir = tmp_path / 'inputs'
    source_dir.mkdir()
    checkpoint = source_dir / 'source.pt'
    checkpoint.write_bytes(b'fake')
    split = source_dir / 'train.txt'
    split.write_text('a\nb\n')
    config = source_dir / 'data.yaml'
    config.write_text(f'root: unused\ntrain_cases: {split}\nval_cases: DO_NOT_OPEN\n')
    monkeypatch.setattr(script, 'EXPECTED_CASES', 2)
    monkeypatch.setattr(script, 'LATENT_SHAPE', (2, 1, 2))
    monkeypatch.setattr(script, 'file_sha256', lambda path: script.TRAIN_SHA256 if Path(path) == split else 'config-hash')
    source = FakeSource()
    loader = MagicMock(return_value=(source, dict(model=latent_probe.E1C_SOURCE_MODEL['name'],
                       path=str(checkpoint), sha256=latent_probe.E1C_SOURCE_SHA256, epoch=11)))
    monkeypatch.setattr(script, 'load_source', loader)
    monkeypatch.setattr(script, 'make_train_dataset', lambda data, ids: object())
    def samples(source, dataset, ids, device, seed, manifest):
        with Path(manifest).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=script.MANIFEST_FIELDS)
            writer.writeheader()
            for metadata in script.patch_plan(ids, seed):
                writer.writerow(dict(metadata, foreground_fallback=False))
                yield metadata, np.full((2, 1, 2), ids.index(metadata['case_id']), dtype=np.float32)
    monkeypatch.setattr(script, 'samples', samples)
    return config, checkpoint, tmp_path / 'audit', loader


def test_output_schema_provenance_overwrite(run_setup):
    config, checkpoint, output, loader = run_setup
    script.run(config, checkpoint, output)
    loader.assert_called_once_with(checkpoint, source_experiment='e1c',
                                   expected_sha256=latent_probe.E1C_SOURCE_SHA256, expected_epoch=11)
    assert {p.name for p in output.iterdir()} == set(script.ARTIFACTS)
    provenance = json.loads((output / 'analysis_provenance.json').read_text())
    assert provenance['patches_per_case'] == 10
    assert provenance['foreground_patches_per_case'] == 7 and provenance['random_patches_per_case'] == 3
    assert provenance['total_patch_count'] == 20 and provenance['case_count'] == 2
    assert provenance['audit_sampling_seed'] == 20260917
    assert provenance['patch_manifest_artifact'] == 'patch_manifest.csv'
    assert provenance['full_scratch_file_mapped'] is False
    assert provenance['validation_data_used'] is False
    assert provenance['validation_statement'] == 'validation data were not used'
    assert provenance['statistics_convention'] == 'population standard deviation, ddof=0'
    assert provenance['normalization_decision_made'] is False
    assert provenance['diffusion_model_trained'] is False
    assert provenance['model_optimization_occurred'] is False
    assert provenance['source_checkpoint_sha256'] == latent_probe.E1C_SOURCE_SHA256
    assert provenance['train_split_sha256'] == script.TRAIN_SHA256
    assert provenance['percentile_method'] == stats.PERCENTILE_METHOD
    assert json.loads((output / 'latent_global_statistics.json').read_text())['std'] == 0.5
    with pytest.raises(FileExistsError):
        script.run(config, checkpoint, output)
    script.run(config, checkpoint, output, overwrite=True)
    assert {p.name for p in output.iterdir()} == set(script.ARTIFACTS)
    (output / 'unrelated').write_text('keep')
    with pytest.raises(ValueError, match='unrelated'):
        script.run(config, checkpoint, output, overwrite=True)


def test_frozen_directory_never_overwritten(run_setup):
    config, checkpoint, _, _ = run_setup
    with pytest.raises(ValueError, match='outputs/frozen'):
        script.run(config, checkpoint, 'outputs/frozen/forbidden', overwrite=True)


@pytest.mark.parametrize('kind', ['shape', 'dtype', 'missing', 'identity'])
def test_latent_stream_integrity(tmp_path, kind):
    value = np.zeros((2, 1, 2), dtype=np.float32)
    case = 'a'
    if kind == 'shape':
        value = value.reshape(1, 2, 2)
    elif kind == 'dtype':
        value = value.astype(np.float64)
    elif kind == 'identity':
        case = 'other'
    samples = [] if kind == 'missing' else [(next(script.patch_plan([case], 1)), value)]
    with pytest.raises(ValueError):
        stats.audit_latents(samples, ['a'], tmp_path, (2, 1, 2))
    assert not list(tmp_path.glob('.d0-scratch-*'))


def test_failed_overwrite_preserves_previous_artifacts(run_setup, monkeypatch):
    config, checkpoint, output, _ = run_setup
    script.run(config, checkpoint, output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    monkeypatch.setattr(script, 'samples', lambda *args: iter([
        (next(script.patch_plan(['a'], 1)), np.full((2, 1, 2), np.nan, dtype=np.float32))]))
    with pytest.raises(ValueError, match='Non-finite'):
        script.run(config, checkpoint, output, overwrite=True)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_cli_delegation(monkeypatch):
    import sys
    run = MagicMock()
    monkeypatch.setattr(script, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['audit', '--data-config', 'data.yaml', '--checkpoint', 'source.pt',
                                     '--output-dir', 'audit-output', '--device', 'cpu', '--overwrite'])
    script.main()
    run.assert_called_once_with(data_config='data.yaml', checkpoint='source.pt',
                                output_dir='audit-output', device='cpu', overwrite=True, audit_seed=script.AUDIT_SEED)


def test_patch_plan_counts_and_stable_seed():
    import builtins
    import hashlib
    from collections import Counter
    ids = [f'case-{i}' for i in range(1080)]
    plan = list(script.patch_plan(ids, 20260917))
    assert len(plan) == 10800
    assert len({(r['case_id'], r['patch_index_within_case']) for r in plan}) == 10800
    counts = Counter((r['case_id'], r['patch_type']) for r in plan)
    assert all(counts[case, 'foreground'] == 7 and counts[case, 'random'] == 3 for case in ids)
    assert plan == list(script.patch_plan(ids, 20260917))
    payload = b'[20260917,"case-0",0,"foreground"]'
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') & ((1 << 63) - 1)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, 'hash', lambda *_: (_ for _ in ()).throw(AssertionError('Python hash used')))
        assert script.patch_seed(20260917, 'case-0', 0, 'foreground') == expected
    assert len({script.patch_seed(20260917, 'case-0', i, script.patch_type(i)) for i in range(10)}) == 10
    assert script.patch_seed(1, 'case-0', 0, 'foreground') != expected
    assert script.patch_seed(20260917, 'case-0', 0, 'random') != expected


@pytest.mark.parametrize('kind', ['foreground', 'random'])
@pytest.mark.parametrize('mask_type', ['boundary', 'dense', 'empty', 'rc_only'])
def test_existing_sampler_exact_equivalence(kind, mask_type):
    from datasets.patch_sampling import PatchSampler
    mask = torch.zeros(12, 13, 14, dtype=torch.int64)
    if mask_type == 'boundary':
        mask[0, 0, 0], mask[-1, -1, -1] = 1, 3
    elif mask_type == 'dense':
        mask[:] = 2
    elif mask_type == 'rc_only':
        mask[0, -1, 0] = 4
    foreground = script.training_foreground(mask)
    assert torch.equal(foreground, (mask == 1) | (mask == 2) | (mask == 3) | (mask == 4))
    assert script.RecordedPatchSampler.sample_slices is PatchSampler.sample_slices
    for seed in range(12):
        metadata = dict(case_id='case', patch_index_within_case=0 if kind == 'foreground' else 7,
                        patch_type=kind, seed=seed)
        original = PatchSampler((8, 8, 8), 1 if kind == 'foreground' else 0,
                                torch.Generator().manual_seed(seed))
        expected = original.sample_slices(foreground)
        slices, row = script.select_patch(foreground, metadata, (8, 8, 8))
        assert slices == expected
        assert script.select_patch(foreground, metadata, (8, 8, 8)) == (slices, row)
        assert row['foreground_fallback'] == (kind == 'foreground' and mask_type == 'empty')
        if kind == 'foreground' and mask_type != 'empty':
            center = tuple(row[f'center_{axis}'] for axis in ('d', 'h', 'w'))
            assert foreground[center]
            assert all(s.start <= c < s.stop for s, c in zip(slices, center))
        else:
            assert all(row[f'center_{axis}'] == '' for axis in ('d', 'h', 'w'))


@pytest.mark.parametrize('kind', ['foreground', 'random'])
def test_preaugmentation_dataset_crop_equivalence(monkeypatch, kind):
    from datasets import brats_dataset as data_module
    from datasets.label_mapping import segmentation_to_tissue_masks
    from datasets.patch_sampling import PatchSampler
    # Real __getitem__, normalization, label conversion, and crop sampler;
    # only file loading and header lookup are replaced with synthetic volumes.
    image = torch.arange(4 * 12 * 13 * 14, dtype=torch.float32).reshape(4, 12, 13, 14)
    mask = torch.zeros(12, 13, 14)
    mask[0, 0, 0] = 4
    mask[8, 8, 8] = 1
    monkeypatch.setattr(data_module, 'load_subject', lambda record: (image.clone(), mask.clone()))
    monkeypatch.setattr(data_module.nib, 'load', lambda path: SimpleNamespace(
        header=SimpleNamespace(get_zooms=lambda: (1, 1, 1))))
    dataset = data_module.BraTSDataset.__new__(data_module.BraTSDataset)
    dataset.records = [dict(subject_id='case', paths={'T1n': 'unused'})]
    dataset.normalize, dataset.label_mapping, dataset.augmentation = True, segmentation_to_tissue_masks, None
    dataset.patch_sampler = None
    full = dataset[0]
    metadata = dict(case_id='case', patch_index_within_case=0, patch_type=kind, seed=123)
    slices, row = script.select_patch(script.training_foreground(full['raw_mask']), metadata, (8, 8, 8))
    dataset.patch_sampler = PatchSampler((8, 8, 8), 1 if kind == 'foreground' else 0,
                                         torch.Generator().manual_seed(123))
    training = dataset[0]
    torch.testing.assert_close(full['image'][(slice(None), *slices)], training['image'], rtol=0, atol=0)
    torch.testing.assert_close(full['raw_mask'][slices], training['raw_mask'], rtol=0, atol=0)
    assert training['crop_start'] == tuple(row[f'start_{axis}'] for axis in ('d', 'h', 'w'))


def test_patch_manifest_and_reproducible_sampling(tmp_path, monkeypatch):
    mask = torch.zeros(98, 100, 102, dtype=torch.int64)
    mask[0, 0, 0], mask[97, 99, 101] = 4, 1
    image = torch.arange(4 * 98 * 100 * 102, dtype=torch.float32).reshape(4, 98, 100, 102)
    dataset = [dict(case_id='a', image=image, raw_mask=mask)]
    monkeypatch.setattr(script, 'extract_anatomy', lambda source, patch: patch[0, :1, :1, :1, :1].numpy())
    first_path, second_path = tmp_path / 'first.csv', tmp_path / 'second.csv'
    first = list(script.samples(None, dataset, ['a'], 'cpu', 20260917, first_path))
    torch.manual_seed(100)  # Per-patch generators must ignore the global RNG state.
    second = list(script.samples(None, dataset, ['a'], 'cpu', 20260917, second_path))
    assert first_path.read_bytes() == second_path.read_bytes()
    rows = list(csv.DictReader(first_path.open()))
    assert list(rows[0]) == list(script.MANIFEST_FIELDS) and len(rows) == 10
    assert sum(r['patch_type'] == 'foreground' for r in rows) == 7
    for (m1, z1), (m2, z2), row in zip(first, second, rows):
        assert m1 == m2
        np.testing.assert_array_equal(z1, z2)
        starts = tuple(int(row[f'start_{axis}']) for axis in ('d', 'h', 'w'))
        assert z1.item() == image[(0, *starts)].item()
        assert all(int(row[f'stop_{axis}']) - int(row[f'start_{axis}']) == 96 for axis in ('d', 'h', 'w'))


def test_disk_mapping_bounded_to_one_channel(tmp_path, monkeypatch):
    original = np.memmap
    sizes = []
    def mapped(*args, **kwargs):
        sizes.append(kwargs['shape'])
        return original(*args, **kwargs)
    monkeypatch.setattr(stats.np, 'memmap', mapped)
    values = np.arange(2 * 10 * 2 * 3, dtype=np.float32).reshape(2, 10, 2, 1, 3)
    result, _ = stats.audit_latents(synthetic_samples(values), ['a', 'b'], tmp_path, (2, 1, 3))
    assert sizes and all(shape == (60,) for shape in sizes)
    assert result['p50'] == pytest.approx(np.median(values))
    scratch = 1080 * 10 * 256 * 12**3 * 4
    assert scratch == 19110297600
    assert scratch / 1024**3 == 17.7978515625


def test_equivalence_to_original_seventy_percent_sampler():
    from datasets.patch_sampling import PatchSampler
    foreground = torch.zeros(12, 13, 14, dtype=torch.bool)
    foreground[0, 0, 0] = foreground[11, 12, 13] = True
    kinds = set()
    for seed in range(30):
        draw = float(torch.rand((), generator=torch.Generator().manual_seed(seed)))
        kind = 'foreground' if draw < 0.7 else 'random'
        kinds.add(kind)
        original = PatchSampler((8, 8, 8), 0.7, torch.Generator().manual_seed(seed))
        metadata = dict(case_id='a', patch_index_within_case=0, patch_type=kind, seed=seed)
        selected, _ = script.select_patch(foreground, metadata, (8, 8, 8))
        assert selected == original.sample_slices(foreground)
    assert kinds == {'foreground', 'random'}


def test_empty_foreground_cannot_be_silently_counted_as_foreground(tmp_path):
    dataset = [dict(case_id='a', image=torch.zeros(4, 96, 96, 96),
                    raw_mask=torch.zeros(96, 96, 96, dtype=torch.int64))]
    with pytest.raises(ValueError, match='empty foreground'):
        list(script.samples(None, dataset, ['a'], 'cpu', 1, tmp_path / 'patches.csv'))
