import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_d0c3_deltaz_inference.py'
SPEC = importlib.util.spec_from_file_location('run_d0c3_deltaz_inference', SCRIPT)
inf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inf)


def manifest_row(case='Case_001', window=0, start=(0, 0, 0)):
    return dict(case_id=case, split='validation', window_index=window,
        start_d=start[0], start_h=start[1], start_w=start[2],
        stop_d=start[0] + 96, stop_h=start[1] + 96, stop_w=start[2] + 96,
        latent_start_d=start[0] // 8, latent_start_h=start[1] // 8, latent_start_w=start[2] // 8,
        latent_stop_d=start[0] // 8 + 12, latent_stop_h=start[1] // 8 + 12, latent_stop_w=start[2] // 8 + 12)


class ZeroEpsilonModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.seen = []

    def forward(self, x, t):
        self.seen.append(int(t.item()))
        return torch.zeros_like(x)


def test_contract_and_frozen_hashes_are_exact():
    assert inf.sha256_file(inf.B4B_PATH) == '29cf659528f6257cd9f8f218e9f4eaf8fabdc6d8a0e4905505aa8548ae68bde4'
    assert inf.sha256_file(inf.SPATIAL_MANIFEST_PATH) == '58fac2ebef778d4d19d6da9f21451c25175c61bb4100c03387e8d3c859c05d53'
    assert inf.sha256_file(inf.D0A_PATH) == 'bca5c0f8f1617d73fc4d0850fe5b6f45e4b2bcbacbb4018f1d5d825de49ad792'
    deps = inf.verify_frozen_inputs()
    assert deps['d0c2_best_checkpoint_expected_sha256'] == inf.D0C2_SHA256


def test_primary_frozen_settings_and_schedule():
    b4b = json.loads(inf.B4B_PATH.read_text())
    assert b4b['partial_noise']['primary_timestep'] == 495
    assert b4b['ddim']['eta'] == 0
    assert b4b['ddim']['model_evaluations'] == 50
    assert b4b['randomness']['primary_noise_replicates'] == 4
    assert b4b['randomness']['inference_global_seed'] == 20260923
    schedule = inf.cosine_schedule()
    assert float(schedule['alpha_bar'][495]) == pytest.approx(0.5000776052474976, abs=1e-7)
    assert [v['timestep'] for v in b4b['partial_noise']['points'] if v['secondary']] == [327, 663]


def test_exact_b4b_seed_payload_and_paired_noise():
    row = manifest_row(start=(48, 96, 48))
    identity = ['Case_001', 48, 96, 48, 144, 192, 144]
    payload = [20260923, identity, 2, 'partial_noise']
    expected = int(hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()[:16], 16) & ((1 << 63) - 1)
    assert inf.deterministic_noise_seed('Case_001', row, 2) == expected
    s0, e0 = inf.make_partial_noise('Case_001', row, 0)
    s0_again, e0_again = inf.make_partial_noise('Case_001', row, 0)
    s1, e1 = inf.make_partial_noise('Case_001', row, 1)
    assert s0 == s0_again and torch.equal(e0, e0_again)
    assert s0 != s1 and not torch.equal(e0, e1)
    assert e0.shape == (1, 256, 12, 12, 12) and e0.dtype == torch.float32
    assert s0 == inf.deterministic_noise_seed('Case_001', row, 0)  # t is absent from identity


def test_ddim_uses_frozen_sequence_eta_zero_and_exactly_50_model_calls():
    model = ZeroEpsilonModel()
    z0 = torch.randn((1, 256, 12, 12, 12))
    support = torch.ones((1, 1, 12, 12, 12), dtype=torch.bool)
    epsilon = torch.randn_like(z0)
    prior, count = inf.ddim_reconstruct(model, z0, support, epsilon, 495)
    sequence = json.loads(inf.B4B_PATH.read_text())['ddim']['resolved_timesteps']
    assert count == len(model.seen) == 50
    assert model.seen == sequence and model.seen[0] == 495 and model.seen[-1] == 0
    assert prior.shape == z0.shape and torch.isfinite(prior).all()


@pytest.mark.parametrize('timestep,alpha', [(327, 0.7501165866851807), (663, 0.25003883242607117)])
def test_sensitivity_ddim_is_frozen_secondary_sequence(timestep, alpha):
    model = ZeroEpsilonModel()
    z = torch.ones((1, 256, 12, 12, 12))
    mask = torch.ones((1, 1, 12, 12, 12), dtype=torch.bool)
    prior, count = inf.ddim_reconstruct(model, z, mask, torch.zeros_like(z), timestep)
    assert count == len(model.seen) == 50
    assert model.seen[0] == timestep and model.seen[-1] == 0
    assert float(inf.cosine_schedule()['alpha_bar'][timestep]) == pytest.approx(alpha, abs=1e-7)


def test_support_uses_raw_occupancy_threshold_and_shape():
    raw = torch.zeros((4, 96, 96, 96), dtype=torch.float32)
    raw[0, :8, :8, :4] = 1  # exactly 50% of one 8^3 cell
    raw[1, 8:16, :8, :8] = (torch.arange(512).reshape(8, 8, 8) < 255).float()
    support = inf.support_from_raw_window(raw)
    assert support.shape == (1, 1, 12, 12, 12)
    assert support[0, 0, 0, 0, 0]
    assert not support[0, 0, 1, 0, 0]
    with pytest.raises(ValueError, match='zero valid latent support'):
        inf.support_from_raw_window(torch.zeros_like(raw))


def test_exact_d0a_per_channel_standardization():
    means = torch.arange(256, dtype=torch.float32).view(1, 256, 1, 1, 1)
    stds = torch.full_like(means, 2.0)
    za = (means + 6.0).expand(1, 256, 12, 12, 12)
    result = inf.standardize_za(za, means, stds)
    assert result.shape == (1, 256, 12, 12, 12)
    assert torch.equal(result, torch.full_like(result, 3.0))


def test_delta_z_abs_then_channel_mean_then_replicate_mean_and_mask():
    z0 = torch.zeros((1, 256, 12, 12, 12))
    support = torch.zeros((1, 1, 12, 12, 12), dtype=torch.bool)
    support[..., 0, 0, 0] = True
    p1 = torch.zeros_like(z0); p2 = torch.zeros_like(z0)
    p1[:, 0, 0, 0, 0] = 1; p2[:, 0, 0, 0, 0] = -1
    prior_sets = [p1, p2, p1, p2]
    delta, replicates = inf.local_deltaz(z0, prior_sets, support)
    assert delta.shape == (1, 1, 12, 12, 12)
    assert delta[0, 0, 0, 0, 0].item() == pytest.approx(1 / 256)
    assert torch.count_nonzero(delta.masked_select(~support)) == 0
    assert replicates[0][0, 0, 0, 0, 0].item() == pytest.approx(1 / 256)
    # This differs from abs(z0 - mean(priors)) because absolute value precedes averaging.
    assert not torch.equal(delta, (z0 - torch.stack(prior_sets).mean(0)).abs().mean(1, keepdim=True) * support)


def test_manifest_exact_frozen_usage_no_regenerated_windows():
    rows, by_case = inf.parse_manifest()
    assert len(rows) == 9720 and len(by_case) == 270
    assert all(len(case_rows) == 36 for case_rows in by_case.values())
    assert all(row['split'] == 'validation' for row in rows)
    assert all(set(row) == set(inf.MANIFEST_FIELDS) for row in rows)
    assert [(r['case_id'], r['window_index']) for r in rows] == sorted(
        (r['case_id'], r['window_index']) for r in rows)


def test_local_to_global_exact_placement_and_uniform_valid_overlap():
    row0 = manifest_row(window=0, start=(0, 0, 0))
    row1 = manifest_row(window=1, start=(0, 0, 48))
    local0 = torch.full((1, 1, 12, 12, 12), 2.0)
    local1 = torch.full((1, 1, 12, 12, 12), 6.0)
    support0 = torch.zeros((1, 1, 12, 12, 12), dtype=torch.bool)
    support1 = torch.zeros_like(support0)
    support0[..., 0, 0, 6] = True
    support1[..., 0, 0, 0] = True
    # These two local cells map to the same global coordinate (0,0,6).
    mean, valid, counts = inf.place_and_aggregate([local0, local1], [support0, support1], [row0, row1])
    assert mean.shape == valid.shape == counts.shape == (1, 23, 28, 23)
    assert mean[0, 0, 0, 6].item() == 4.0 and counts[0, 0, 0, 6].item() == 2
    assert valid[0, 0, 0, 6]
    assert not valid[0, 0, 0, 7] and counts[0, 0, 0, 7] == 0 and mean[0, 0, 0, 7] == 0


def test_ema_state_is_loaded_raw_state_is_never_loaded(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / 'trusted.pt'
    checkpoint_path.write_bytes(b'checkpoint')
    monkeypatch.setattr(inf, 'verify_checkpoint', lambda path, expected, label: (Path(path), expected))

    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)

    monkeypatch.setattr(inf, 'load_source', lambda *a, **k: (Source(), {'sha256': inf.E1C_SHA256}))
    marker = {'used': None}

    class Diffusion(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.p = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)

        def load_state_dict(self, state, strict=True):
            marker['used'] = state
            self.p.data.fill_(state['ema_marker'].item())
            return None

    monkeypatch.setattr(inf, 'LatentDiffusionUNet3D', Diffusion)
    monkeypatch.setattr(inf, 'EXPECTED_PARAMETER_COUNT', 1)
    ema = {'ema_marker': torch.tensor(4.), **{f'e{i}': torch.tensor(float(i)) for i in range(141)}}
    raw = {'raw_marker': torch.tensor(9.), **{f'r{i}': torch.tensor(float(i)) for i in range(141)}}
    metadata = dict(epoch_index=99, human_epoch=100, successful_optimizer_step=13499,
        primary_validation_metric=.7276276773324719, best_primary_validation_metric=.7276276773324719,
        best_epoch=99, next_sample_cursor=1080)
    monkeypatch.setattr(inf.torch, 'load', lambda *a, **k: {**metadata,
        'architecture_provenance': {'model': 'LatentDiffusionUNet3D', 'parameter_count': 1},
        'ema_state': ema, 'raw_model_state': raw})
    source, model, info = inf.load_inference_models(checkpoint_path, checkpoint_path)
    assert marker['used'] is ema
    assert model.p.item() == 4 and info['diffusion_state_key'] == 'ema_state'
    assert info['raw_state_key_loaded'] is False
    assert all(not p.requires_grad for p in source.parameters())
    monkeypatch.setattr(inf.torch, 'load', lambda *a, **k: {**metadata,
        'architecture_provenance': {'model': 'LatentDiffusionUNet3D', 'parameter_count': 1},
        'raw_model_state': raw})
    with pytest.raises(ValueError, match='missing ema_state'):
        inf.load_inference_models(checkpoint_path, checkpoint_path)


def test_checkpoint_hash_enforcement(tmp_path):
    wrong = tmp_path / 'wrong.pt'; wrong.write_bytes(b'wrong')
    for expected, label in ((inf.E1C_SHA256, 'E1-C'), (inf.D0C2_SHA256, 'D0-C2 best')):
        with pytest.raises(ValueError, match='SHA256 mismatch'):
            inf.verify_checkpoint(wrong, expected, label)
    with pytest.raises(ValueError, match='does not exist'):
        inf.verify_checkpoint(tmp_path / 'missing.pt', inf.E1C_SHA256, 'E1-C')


def test_generation_mri_loader_never_opens_segmentation(tmp_path, monkeypatch):
    case = 'Case_001'; folder = tmp_path / case; folder.mkdir()
    loaded = []

    class Header:
        def get_zooms(self): return (1., 1., 1.)

    class Volume:
        shape = (4, 5, 6)
        affine = np.eye(4)
        header = Header()
        def get_fdata(self, dtype): return np.ones(self.shape, dtype=dtype)

    def fake_load(path):
        loaded.append(Path(path).name)
        return Volume()
    monkeypatch.setattr(inf.nib, 'load', fake_load)
    result = inf._load_original_mri(tmp_path, case)
    assert result.shape == (4, 4, 5, 6)
    assert loaded == [f'{case}-{name}.nii.gz' for name in ('t1n', 't1c', 't2w', 't2f')]
    assert not any('seg' in name for name in loaded)


def test_full_volume_normalizes_before_padding_and_raw_pad_is_zero(tmp_path, monkeypatch):
    raw = torch.ones((4, 10, 11, 12), dtype=torch.float32)
    observed = []
    monkeypatch.setattr(inf, '_load_original_mri', lambda root, case: raw)
    monkeypatch.setattr(inf, 'normalize_nonzero', lambda value: observed.append(tuple(value.shape)) or value * 3)
    row = dict(original_d=10, original_h=11, original_w=12,
               padded_d=16, padded_h=16, padded_w=16)
    original, raw_padded, normalized_padded = inf.prepare_case_mri(tmp_path, 'Case_001', [row])
    assert observed == [(4, 10, 11, 12)]
    assert original.shape == (4, 10, 11, 12)
    assert torch.equal(normalized_padded[:, :10, :11, :12], raw * 3)
    assert torch.count_nonzero(normalized_padded[:, 10:]) == 0
    assert torch.count_nonzero(raw_padded[:, 10:]) == 0
    edge = raw_padded[:, 0:10, 0:11, 0:12]
    assert torch.equal(edge, raw)


def test_smoke_scientific_path_isolation():
    assert inf.validate_output_mode_path(inf.DEFAULT_SCIENTIFIC_OUTPUT, scientific=True) == inf.DEFAULT_SCIENTIFIC_OUTPUT.resolve()
    assert inf.validate_output_mode_path(inf.DEFAULT_SMOKE_OUTPUT, scientific=False) == inf.DEFAULT_SMOKE_OUTPUT.resolve()
    with pytest.raises(ValueError, match='Scientific output'):
        inf.validate_output_mode_path(inf.DEFAULT_SMOKE_OUTPUT, scientific=True)
    with pytest.raises(ValueError, match='Smoke output'):
        inf.validate_output_mode_path(inf.DEFAULT_SCIENTIFIC_OUTPUT, scientific=False)


def test_atomic_case_completion_and_verified_resume(tmp_path):
    case_dir = tmp_path / 'Case_001'
    identity = {'run': 'frozen'}
    tensors = {'DeltaZ_global': torch.zeros((1, 23, 28, 23))}
    provenance = {'run_identity': identity, 'ground_truth_used': False}
    row = dict(case_id='Case_001', window_index=0, start_d=0, start_h=0, start_w=0,
        stop_d=96, stop_h=96, stop_w=96, replicate_index=0, seed=12, timestep=495,
        noise_identity='[]', support_fraction=.5, local_deltaz_mean=1., local_deltaz_std=0.,
        local_deltaz_min=1., local_deltaz_max=1.)
    inf.atomic_save_case(case_dir, tensors, provenance, [row])
    assert inf.verify_completed_case(case_dir, identity)['run_identity'] == identity
    with pytest.raises(ValueError, match='does not match'):
        inf.verify_completed_case(case_dir, {'run': 'different contract/checkpoint/code'})
    with pytest.raises(FileExistsError):
        inf.atomic_save_case(case_dir, tensors, provenance, [row])
    with (case_dir / 'case_deltaz.pt').open('ab') as stream:
        stream.write(b'bad')
    with pytest.raises(ValueError, match='checksum mismatch'):
        inf.verify_completed_case(case_dir, identity)


def test_source_fingerprint_detects_mutation():
    module = torch.nn.Linear(2, 2)
    before = inf.model_fingerprint(module)
    with torch.no_grad(): module.weight.add_(1)
    assert inf.model_fingerprint(module) != before
