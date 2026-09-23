import pytest
import torch

from models.latent_diffusion_unet3d import (
    LatentDiffusionUNet3D, cosine_schedule, masked_epsilon_mse, q_sample,
    sinusoidal_timestep_embedding,
)
from scripts.preflight_d0b4_diffusion_memory import (
    EXPECTED_BRANCH, assert_za_shape, validate_microbatch, validate_runtime_branch,
)


def test_model_shape_and_locked_structure():
    model = LatentDiffusionUNet3D()
    x = torch.randn(1, 256, 12, 12, 12)
    out = model(x, torch.tensor([10]))
    assert out.shape == x.shape
    assert model.input_conv.out_channels == 64
    assert model.down1.out_channels == 128
    assert model.down2.out_channels == 256
    assert model.output_conv.out_channels == 256
    assert not any("attention" in name.lower() for name, _ in model.named_modules())
    assert torch.count_nonzero(model.output_conv.weight) == 0
    assert torch.count_nonzero(model.output_conv.bias) == 0


def test_timestep_schedule_and_q_sample_mask():
    assert sinusoidal_timestep_embedding(torch.tensor([0, 1]), 64).shape == (2, 64)
    schedule = cosine_schedule()
    assert schedule["beta"].shape == (1000,)
    assert torch.isfinite(schedule["beta"]).all()
    assert float(schedule["beta"].max()) <= .999001
    assert torch.all(schedule["alpha_bar"][1:] <= schedule["alpha_bar"][:-1])
    z = torch.ones(1, 256, 12, 12, 12)
    mask = torch.zeros(1, 1, 12, 12, 12); mask[..., 0, 0, 0] = 1
    out = q_sample(z, torch.tensor([500]), mask, schedule, noise=torch.ones_like(z))
    assert torch.equal(out * (1 - mask), torch.zeros_like(out))


def test_masked_loss_exact_and_zero_support_guard():
    pred = torch.tensor([[[[[1.]], [[2.]]], [[[3.]], [[4.]]]]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[[[[1.]], [[0.]]]]])
    assert masked_epsilon_mse(pred, target, mask).item() == pytest.approx((1 + 3 * 3) / 2)
    with pytest.raises(ValueError):
        masked_epsilon_mse(pred, target, torch.zeros_like(mask))


def test_preflight_guards():
    assert [validate_microbatch(x) for x in (8, 4, 2, 1)] == [1, 2, 4, 8]
    with pytest.raises(ValueError): validate_microbatch(3)
    validate_runtime_branch(EXPECTED_BRANCH)
    with pytest.raises(RuntimeError): validate_runtime_branch("main")
    with pytest.raises(ValueError): assert_za_shape(torch.zeros(1, 4, 12, 12, 12))


def test_exact_locked_widths_and_modules():
    from torch import nn
    model = LatentDiffusionUNet3D()
    for blocks, widths in [(model.level0, [(64, 64)] * 2),
                           (model.level1, [(128, 128)] * 2),
                           (model.level2, [(256, 256)] * 2),
                           (model.middle, [(256, 256)] * 2),
                           (model.decoder1, [(256, 128), (128, 128)]),
                           (model.decoder0, [(128, 64), (64, 64)])]:
        assert [(b.conv1.in_channels, b.conv1.out_channels) for b in blocks] == widths
    assert (model.up1.in_channels, model.up1.out_channels) == (256, 128)
    assert (model.up2.in_channels, model.up2.out_channels) == (128, 64)
    assert not any(isinstance(m, (nn.MultiheadAttention, nn.Dropout, nn.Dropout3d)) for m in model.modules())
    assert not any('attention' in type(m).__name__.lower() for m in model.modules())
    assert model.time_embedding(torch.tensor([0, 999])).shape == (2, 256)
    assert sum(p.numel() for p in model.parameters()) == 22825280


def test_cosine_post_clamp_indexing_and_first_step():
    schedule = cosine_schedule()
    assert all(x.shape == (1000,) and torch.isfinite(x).all() for x in schedule.values())
    beta, alpha, ab = (schedule[k] for k in ('beta', 'alpha', 'alpha_bar'))
    assert ((beta > 0) & (beta <= .999)).all()
    assert torch.equal(alpha, 1 - beta)
    assert torch.equal(ab, torch.cumprod(alpha, 0))
    assert ab[0] == alpha[0] and ab[0] < 1
    assert (ab[1:] <= ab[:-1]).all()
    z = torch.zeros(1, 2, 2, 2, 2)
    mask = torch.ones(1, 1, 2, 2, 2)
    result = q_sample(z, torch.tensor([0]), mask, schedule, torch.ones_like(z))
    assert result.shape == z.shape
    assert torch.all(result > 0)
    torch.testing.assert_close(result, torch.full_like(z, (1 - alpha[0]).sqrt()))


def test_outside_support_errors_do_not_affect_loss():
    pred = torch.tensor([[[[[1., 1000.]]], [[[3., -1000.]]]]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[[[[1., 0.]]]]])
    assert masked_epsilon_mse(pred, target, mask).item() == 5
    pred[..., 1] = 0
    assert masked_epsilon_mse(pred, target, mask).item() == 5


def synthetic_manifest():
    ids = [f'case_{i}' for i in range(12)]
    rows = [dict(case_id=case, patch_index=p, crop_start_d=p, crop_start_h=0,
                 crop_start_w=0, crop_stop_d=p+96, crop_stop_h=96, crop_stop_w=96)
            for case in ids for p in range(20)]
    return ids, rows


def test_effective_batch_and_randomness_are_partition_invariant():
    from scripts.preflight_d0b4_diffusion_memory import (
        select_effective_batch, fingerprint, sample_randomness, randomness_metadata,
    )
    ids, rows = synthetic_manifest()
    selected = select_effective_batch(ids, rows)
    assert len(selected) == len({r['case_id'] for r in selected}) == 8
    assert all(r in rows for r in selected)
    assert selected == select_effective_batch(list(reversed(ids)), list(reversed(rows)))
    reference_t, reference_noise = sample_randomness(selected, (2, 3, 3, 3), 'cpu')
    for mb in (8, 4, 2, 1):
        resolved = select_effective_batch(ids, rows)
        assert fingerprint(resolved) == fingerprint(selected)
        chunks = [resolved[i:i+mb] for i in range(0, 8, mb)]
        values = [sample_randomness(chunk, (2, 3, 3, 3), 'cpu') for chunk in chunks]
        assert torch.equal(torch.cat([v[0] for v in values]), reference_t)
        assert torch.equal(torch.cat([v[1] for v in values]), reference_noise)
        assert [m for chunk in chunks for m in randomness_metadata(chunk)] == randomness_metadata(selected)


def test_output_uses_current_microbatch():
    from scripts.preflight_d0b4_diffusion_memory import validate_output_shape
    for mb in (8, 4, 2, 1):
        z = torch.empty(mb, 256, 12, 12, 12)
        validate_output_shape(torch.empty_like(z), z)
        with pytest.raises(ValueError):
            validate_output_shape(torch.empty(mb+1, 256, 12, 12, 12), z)


def test_output_overwrite_precedes_any_processing(tmp_path, monkeypatch):
    import scripts.preflight_d0b4_diffusion_memory as preflight
    path = tmp_path / 'existing.json'
    path.write_text('untouched')
    def forbidden(*args, **kwargs):
        pytest.fail('Processing before overwrite guard')
    monkeypatch.setattr(preflight, 'runtime_git_provenance', forbidden)
    monkeypatch.setattr(preflight, 'read_config', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    with pytest.raises(FileExistsError):
        preflight.run(8, output_json=path)
    assert path.read_text() == 'untouched'


def test_frozen_source_contract():
    from types import SimpleNamespace
    from scripts.preflight_d0b4_diffusion_memory import extract_za, source_gradient_free
    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.projection = SimpleNamespace(anatomy_projector=lambda x: x)
        def encoder(self, x):
            assert not torch.is_grad_enabled()
            return [self.weight * torch.ones(x.shape[0], 256, 12, 12, 12)]
    source = Source()
    batch = torch.empty(1, 4, 96, 96, 96)
    with pytest.raises(ValueError):
        extract_za(source, batch)
    source.eval().requires_grad_(False)
    z = extract_za(source, batch)
    assert z.dtype == torch.float32 and not z.requires_grad
    trainable = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([trainable])
    (trainable * z.mean()).backward()
    optimizer.step()
    assert source_gradient_free(source)
    source.weight.grad = torch.ones_like(source.weight)
    assert not source_gradient_free(source)


def test_every_pass_criterion_and_non_scientific_flags():
    from scripts.preflight_d0b4_diffusion_memory import PASS_FLAGS, NON_SCIENTIFIC, candidate_pass
    assert set(PASS_FLAGS) == {'loss_finite', 'gradient_finite', 'all_za_shapes_valid',
        'all_output_shapes_valid', 'source_gradient_free', 'optimizer_step_performed',
        'optimizer_state_materialized', 'memory_headroom_pass'}
    result = dict.fromkeys(PASS_FLAGS, True)
    result.update(oom=False, effective_batch_size=8, failure_reason=None)
    assert candidate_pass(result)
    for key in PASS_FLAGS:
        assert not candidate_pass({**result, key: False})
    assert not candidate_pass({**result, 'oom': True})
    assert not candidate_pass({**result, 'effective_batch_size': 7})
    assert not candidate_pass({**result, 'failure_reason': 'failure'})
    assert NON_SCIENTIFIC == dict(scientific_training=False, validation_used=False,
                                 model_selection=False, scientific_interpretation=False)


def test_frozen_decision_manifest_verification(tmp_path, monkeypatch):
    import scripts.preflight_d0b4_diffusion_memory as preflight
    root = tmp_path / 'frozen'
    decision = root / 'decision' / 'record.json'
    decision.parent.mkdir(parents=True)
    decision.write_text('{"decision": {"selected_operator": "local_empirical_resample"}}')
    digest = preflight.file_sha256(decision)
    monkeypatch.setattr(preflight, 'D0B3_DECISION_SHA256', digest)
    manifest = root / 'MANIFEST.sha256'
    manifest.write_text(f'{digest}  ./decision/record.json\n')
    assert preflight.verify_d0b3(decision)['d0b3_manifest_decision_verified']
    manifest.write_text(f'{"0" * 64}  ./decision/record.json\n')
    with pytest.raises(ValueError, match='MANIFEST'):
        preflight.verify_d0b3(decision)


@pytest.mark.parametrize('microbatch', [8, 4, 2, 1])
def test_candidate_lifecycle_cpu_mock(tmp_path, monkeypatch, microbatch):
    """Exercise the real accumulation/update/report flow without initializing CUDA."""
    import contextlib
    import json
    from types import SimpleNamespace
    import scripts.preflight_d0b4_diffusion_memory as pf
    ids, records = synthetic_manifest()
    monkeypatch.setattr(pf, 'runtime_git_provenance', lambda: {'runtime_git_branch': pf.EXPECTED_BRANCH})
    monkeypatch.setattr(pf, 'read_config', lambda _: {'train_cases': 'unused', 'root': 'unused'})
    monkeypatch.setattr(pf, 'file_sha256', lambda path: {
        'patch_manifest.csv': pf.PATCH_MANIFEST_SHA256,
        'best.pt': pf.CHECKPOINT_SHA256,
        'unused': pf.TRAIN_SHA256,
    }[path.name])
    monkeypatch.setattr(pf, 'verify_d0b3', lambda _: {})
    monkeypatch.setattr(pf, 'read_case_ids', lambda _: ids)
    monkeypatch.setattr(pf, 'read_manifest', lambda *a: records)
    monkeypatch.setattr(pf, '_load_contract', lambda _: (torch.zeros(256), torch.ones(256)))
    original_to = torch.Tensor.to
    def cpu_to(self, *args, **kwargs):
        if args and args[0] == 'cuda':
            args = ('cpu', *args[1:])
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    for name in ['empty_cache', 'reset_peak_memory_stats', 'manual_seed_all', 'synchronize']:
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(total_memory=1000))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'mock CUDA; no device used')
    for name in ['max_memory_allocated', 'max_memory_reserved', 'memory_allocated', 'memory_reserved']:
        monkeypatch.setattr(torch.cuda, name, lambda: 100)
    monkeypatch.setattr(torch.cuda.amp, 'autocast', lambda **kw: contextlib.nullcontext())
    class Scaler:
        def __init__(self, **kw): pass
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass
    monkeypatch.setattr(torch.cuda.amp, 'GradScaler', Scaler)
    events = []
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def to(self, device): return self
        def forward(self, z, t):
            events.append('forward')
            out = z * self.weight
            out.register_hook(lambda grad: events.append('backward'))
            return out
    monkeypatch.setattr(pf, 'LatentDiffusionUNet3D', Model)
    source = Model()
    monkeypatch.setattr(pf, 'load_source', lambda *a, **kw: (source, {}))
    seen = []
    def prepare(rows, root, frozen, means, stds, device):
        assert pf.source_gradient_free(frozen)
        assert len(rows) == microbatch
        seen.extend(rows)
        events.append('extract')
        return torch.ones(len(rows), 256, 12, 12, 12), torch.ones(len(rows), 1, 12, 12, 12)
    monkeypatch.setattr(pf, 'prepare_microbatch', prepare)
    original_randomness = pf.sample_randomness
    monkeypatch.setattr(pf, 'sample_randomness', lambda rows, shape, device: original_randomness(rows, shape, 'cpu'))
    output = tmp_path / 'candidate.json'
    result = pf.run(microbatch, output_json=output)
    assert result['pass'], result['failure_reason']
    assert all(result[k] for k in pf.PASS_FLAGS)
    assert events == ['extract', 'forward', 'backward'] * (8 // microbatch)
    assert seen == pf.select_effective_batch(ids, records)
    assert len(result['step_shapes']) == 8 // microbatch
    assert result == json.loads(output.read_text())
    from pathlib import Path
    assert result['patch_manifest_sha256'] == '9c4ba67fc195537a5500229722ce55e4772ea5b7d135b351b3a2fc4b857f44b9'
    assert result['source_checkpoint_sha256'] == pf.E1C_SOURCE_SHA256
    assert result['patch_manifest_path'] == str(Path('outputs/frozen/d0b_support_mask_audit/patch_manifest.csv').resolve())
    assert result['source_checkpoint_path'] == str(Path('outputs/frozen/e1c_anatomy_constraint/training/best.pt').resolve())
    assert pf.source_gradient_free(source)
    # Allocation OOM is caught before any extraction and produces a failure JSON.
    def oom(*a, **kw):
        raise torch.cuda.OutOfMemoryError('mock allocation OOM')
    monkeypatch.setattr(pf, 'load_source', oom)
    failed = pf.run(microbatch, output_json=tmp_path / 'oom.json')
    assert failed['oom'] and not failed['pass']
    assert failed['failure_stage'] == 'source_allocation'
    assert 'mock allocation OOM' in failed['failure_reason']


@pytest.mark.parametrize('invalid_source,label', [
    ('manifest', 'D0-B0 patch manifest'), ('checkpoint', 'E1-C checkpoint'),
])
def test_source_hash_mismatch_precedes_cuda(tmp_path, monkeypatch, invalid_source, label):
    from unittest.mock import Mock
    import scripts.preflight_d0b4_diffusion_memory as pf
    manifest = tmp_path / 'manifest.csv'
    checkpoint = tmp_path / 'source.pt'
    manifest.write_bytes(b'test manifest')
    checkpoint.write_bytes(b'test checkpoint')
    # Synthetic locked inputs exercise real file hashing, then tamper one source.
    monkeypatch.setattr(pf, 'PATCH_MANIFEST_SHA256', pf.file_sha256(manifest))
    monkeypatch.setattr(pf, 'CHECKPOINT_SHA256', pf.file_sha256(checkpoint))
    (manifest if invalid_source == 'manifest' else checkpoint).write_bytes(b'tampered')
    monkeypatch.setattr(pf, 'runtime_git_provenance', lambda: {'runtime_git_branch': pf.EXPECTED_BRANCH})
    guards = []
    for owner, name in [(torch.cuda, 'is_available'), (torch.cuda, 'empty_cache'),
                        (torch.cuda, 'synchronize'), (torch.cuda, 'manual_seed_all'),
                        (pf, 'load_source'), (pf, 'prepare_microbatch')]:
        guard = Mock(side_effect=AssertionError('CUDA processing reached before source validation'))
        monkeypatch.setattr(owner, name, guard)
        guards.append(guard)
    output = tmp_path / 'candidate.json'
    with pytest.raises(ValueError, match=f'{label} SHA256 mismatch:'):
        pf.run(8, output_json=output, patch_manifest=manifest, checkpoint=checkpoint)
    for guard in guards:
        guard.assert_not_called()
    assert not output.exists()


def test_prepare_microbatch_uses_frozen_crop_raw_support(monkeypatch):
    from unittest.mock import Mock
    import scripts.preflight_d0b4_diffusion_memory as pf
    image = torch.zeros(4, 100, 104, 108)
    # Constant raw foreground becomes zero after normalization, so using
    # normalized MRI for support would give a different answer.
    image[2, 2:98, 3:99, 5:53] = 7
    raw_mask = torch.zeros(100, 104, 108, dtype=torch.int64)
    raw_mask[2, 3, 5:11] = torch.tensor([0, 1, 2, 3, 4, 5])
    row = dict(case_id='case_support', patch_index=17,
               crop_start_d=2, crop_stop_d=98, crop_start_h=3, crop_stop_h=99,
               crop_start_w=5, crop_stop_w=101)
    original_row = row.copy()
    expected_patch = image[:, 2:98, 3:99, 5:101].clone()
    expected_abnormal = torch.isin(raw_mask[2:98, 3:99, 5:101], torch.tensor([1, 2, 3, 4]))
    load = Mock(return_value=(image, raw_mask))
    monkeypatch.setattr(pf, 'load_case', load)
    normalize = Mock(wraps=pf.normalize_nonzero)
    monkeypatch.setattr(pf, 'normalize_nonzero', normalize)
    frozen_support = pf.support_mask
    observed = {}
    def support(patch):
        normalize.assert_called_once()
        assert normalize.call_args.args[0] is image
        assert patch.shape == (4, 96, 96, 96)
        assert torch.equal(patch, expected_patch)
        observed['support'] = frozen_support(patch)
        return observed['support']
    support_spy = Mock(side_effect=support)
    monkeypatch.setattr(pf, 'support_mask', support_spy)
    pools = object()
    def donors(patch, abnormal, raw_support):
        assert torch.equal(patch, expected_patch)
        assert torch.equal(abnormal, expected_abnormal)
        assert raw_support is observed['support']
        return pools, None
    donor_spy = Mock(side_effect=donors)
    monkeypatch.setattr(pf, 'donor_pools', donor_spy)
    def corrupt(patch, abnormal, donor_values, operator, **kwargs):
        assert patch.shape == (4, 96, 96, 96)
        assert torch.count_nonzero(patch) == 0
        assert torch.equal(abnormal, expected_abnormal)
        assert donor_values is pools
        assert operator == 'local_empirical_resample'
        assert kwargs == dict(global_seed=pf.PREFLIGHT_SEED, case_id=row['case_id'], patch_index=17)
        return patch
    corrupt_spy = Mock(side_effect=corrupt)
    monkeypatch.setattr(pf, 'corrupt_input', corrupt_spy)
    frozen_occupancy = pf.block_occupancy
    def occupancy(mask):
        assert mask is observed['support']
        assert mask.shape == (96, 96, 96)
        return frozen_occupancy(mask)
    occupancy_spy = Mock(side_effect=occupancy)
    monkeypatch.setattr(pf, 'block_occupancy', occupancy_spy)
    monkeypatch.setattr(pf, 'extract_za', lambda source, batch: torch.zeros(batch.shape[0], 256, 12, 12, 12))
    z, latent_support = pf.prepare_microbatch([row], 'dataset', object(), torch.zeros(256), torch.ones(256), 'cpu')
    load.assert_called_once_with('dataset', 'case_support')
    for spy in (support_spy, donor_spy, corrupt_spy, occupancy_spy):
        spy.assert_called_once()
    assert row == original_row
    assert z.shape == (1, 256, 12, 12, 12)
    assert latent_support[0].shape == (1, 12, 12, 12)
    expected_support = torch.zeros(1, 1, 12, 12, 12)
    expected_support[..., :6] = 1
    assert torch.equal(latent_support, expected_support)


def test_prepare_microbatch_rejects_incomplete_raw_patch(monkeypatch):
    from unittest.mock import Mock
    import scripts.preflight_d0b4_diffusion_memory as pf
    monkeypatch.setattr(pf, 'load_case', lambda *a: (torch.zeros(4, 96, 96, 96), torch.zeros(96, 96, 96)))
    support = Mock(side_effect=AssertionError('Malformed patch reached support helper'))
    monkeypatch.setattr(pf, 'support_mask', support)
    row = dict(case_id='case', crop_start_d=1, crop_stop_d=97,
               crop_start_h=0, crop_stop_h=96, crop_start_w=0, crop_stop_w=96)
    with pytest.raises(ValueError, match='Expected raw MRI patch'):
        pf.prepare_microbatch([row], 'dataset', object(), torch.zeros(256), torch.ones(256), 'cpu')
    support.assert_not_called()
