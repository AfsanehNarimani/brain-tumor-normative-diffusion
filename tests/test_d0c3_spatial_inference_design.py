import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/freeze_d0c3_spatial_inference_design.py'
SPEC = importlib.util.spec_from_file_location('freeze_d0c3_spatial_inference_design', SCRIPT)
freeze_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze_mod)


def test_known_padding_and_normalization_precedes_padding():
    assert freeze_mod.padded_shape((182, 218, 182)) == (184, 224, 184)
    source = torch.ones((4, 10, 11, 12), dtype=torch.float32)
    seen = []

    def normalize(value):
        seen.append(tuple(value.shape))
        return value * 2

    normalized, padded, original, final = freeze_mod.normalize_full_then_pad(
        source, normalizer=normalize, window_shape=(8, 8, 8), encoder_stride=8)
    assert seen == [(4, 10, 11, 12)]
    assert original == (10, 11, 12)
    assert final == (16, 16, 16)
    assert normalized.shape == (4, 10, 11, 12)
    assert padded.shape == (4, 16, 16, 16)
    assert torch.equal(padded[:, :10, :11, :12], normalized)
    assert torch.count_nonzero(padded[:, 10:, :, :]) == 0
    assert torch.count_nonzero(padded[:, :, 11:, :]) == 0
    assert torch.count_nonzero(padded[:, :, :, 12:]) == 0


def test_frozen_window_geometry_and_alignment():
    padded, starts = freeze_mod.window_geometry()
    assert padded == (184, 224, 184)
    assert starts == [[0, 48, 88], [0, 48, 96, 128], [0, 48, 88]]
    assert all(value % 8 == 0 for axis in starts for value in axis)
    assert len(list(__import__('itertools').product(*starts))) == 36


def test_manifest_has_9720_canonical_geometry_only_rows():
    case_ids = [f'case-{index:03d}' for index in reversed(range(270))]
    rows = freeze_mod.build_manifest_rows(case_ids)
    assert len(rows) == 9720
    assert [(r['case_id'], r['window_index']) for r in rows] == sorted(
        (r['case_id'], r['window_index']) for r in rows)
    assert all(row['split'] == 'validation' for row in rows)
    assert set(rows[0]) == set(freeze_mod.MANIFEST_FIELDS)
    assert not any('gt' in key.lower() or 'seg' in key.lower() or 'deltaz' in key.lower()
                   for key in freeze_mod.MANIFEST_FIELDS)
    assert rows[0]['start_d'] == 0 and rows[0]['start_h'] == 0 and rows[0]['start_w'] == 0
    assert rows[1]['start_w'] == 48
    assert rows[0]['stop_d'] == 96 and rows[0]['latent_start_d'] == 0
    assert rows[0]['latent_stop_d'] == 12
    assert freeze_mod.manifest_bytes(rows) == freeze_mod.manifest_bytes(
        freeze_mod.build_manifest_rows(case_ids))


def test_manifest_rejects_duplicates_and_non_validation_split():
    with pytest.raises(ValueError, match='unique'):
        freeze_mod.build_manifest_rows(['same', 'same'])
    with pytest.raises(ValueError, match='validation split'):
        freeze_mod.build_manifest_rows(['case'], split='train')


def test_local_to_global_mapping_is_integer_exact():
    region = freeze_mod.local_to_global_slices((48, 96, 48))
    assert [(s.start, s.stop) for s in region] == [(6, 18), (12, 24), (6, 18)]
    assert tuple(s.start + local for s, local in zip(region, (2, 3, 4))) == (8, 15, 10)
    with pytest.raises(ValueError, match='divisible by 8'):
        freeze_mod.local_to_global_slices((49, 0, 0))


def test_uniform_valid_support_only_overlap_and_zero_count_invalid():
    first = torch.full((1, 12, 12, 12), 2.0)
    second = torch.full((1, 12, 12, 12), 6.0)
    first_mask = torch.zeros_like(first, dtype=torch.bool)
    second_mask = torch.zeros_like(second, dtype=torch.bool)
    first_mask[:, 0, 0, 0] = True
    second_mask[:, 0, 0, 0] = True
    # A supported cell present in one window contributes that value alone.
    first_mask[:, 0, 0, 1] = True
    out, valid, count = freeze_mod.aggregate_valid_windows(
        [first, second], [first_mask, second_mask], [(0, 0, 0), (0, 0, 0)])
    assert out.shape == valid.shape == count.shape == (1, 23, 28, 23)
    assert out[0, 0, 0, 0].item() == 4.0
    assert count[0, 0, 0, 0].item() == 2
    assert out[0, 0, 0, 1].item() == 2.0
    assert count[0, 0, 0, 1].item() == 1
    assert valid[0, 0, 0, 2].item() is False
    assert out[0, 0, 0, 2].item() == 0.0  # placeholder only; validity is false
    assert count[0, 0, 0, 2].item() == 0


def test_paired_noise_seed_matches_b4b_and_is_replicate_specific():
    seed0 = freeze_mod.partial_noise_seed('BraTS-001', (48, 96, 48), 0)
    payload = [20260923, ['BraTS-001', 48, 96, 48, 144, 192, 144], 0, 'partial_noise']
    payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    expected = int(hashlib.sha256(payload_bytes).hexdigest()[:16], 16) & ((1 << 63) - 1)
    assert seed0 == expected
    assert seed0 != freeze_mod.partial_noise_seed('BraTS-001', (48, 96, 48), 1)
    _, epsilon_a = freeze_mod.partial_noise('BraTS-001', (48, 96, 48), 0)
    _, epsilon_b = freeze_mod.partial_noise('BraTS-001', (48, 96, 48), 0)
    assert epsilon_a.shape == (256, 12, 12, 12)
    assert torch.equal(epsilon_a, epsilon_b)
    # Timestep is intentionally absent: this same epsilon is used at 327/495/663.
    contract = json.loads((ROOT / 'configs/d0c3_spatial_inference_contract.json').read_text())
    noise = contract['noise_identity']
    assert noise['diffusion_timestep_in_seed_payload'] is False
    assert noise['paired_across_timesteps'] == [327, 495, 663]


def test_contract_states_fixed_shapes_and_new_design_completion():
    contract = json.loads((ROOT / 'configs/d0c3_spatial_inference_contract.json').read_text())
    assert contract['design_completion_note'].startswith('This is a new pre-inference spatial design completion')
    assert contract['tensor_geometry']['diffusion_spatial_shape_flexible'] is False
    assert contract['tensor_geometry']['diffusion_spatial_shape_contract'] == [12, 12, 12]
    assert contract['tensor_geometry']['global_latent_shape_for_known_padded_shape'] == [23, 28, 23]
    assert contract['support']['ground_truth_used'] is False
    assert contract['output_separation']['delta_z_generation_uses_ground_truth'] is False


def test_freeze_and_verify_only_are_deterministic_and_read_only(tmp_path):
    rows = freeze_mod.build_manifest_rows(freeze_mod._load_validation_ids())
    output = tmp_path / 'frozen-design'
    result = freeze_mod.freeze(output)
    assert result['manifest_sha256'] == hashlib.sha256((output / 'inference_window_manifest.csv').read_bytes()).hexdigest()
    saved = {path.name: path.read_bytes() for path in output.iterdir()}
    verified = freeze_mod.freeze(output, verify_only=True)
    assert verified['summary']['manifest_rows'] == 9720
    assert saved == {path.name: path.read_bytes() for path in output.iterdir()}
    assert (output / 'inference_window_manifest.csv').read_bytes() == freeze_mod.manifest_bytes(rows)
    assert set(saved) == {'inference_window_manifest.csv', 'spatial_design_summary.json', 'provenance.json', 'MANIFEST.sha256'}
    with pytest.raises(FileExistsError):
        freeze_mod.freeze(output)


def test_actual_dependency_hashes_and_checkpoint_bytes_optional():
    deps = freeze_mod.verify_dependencies()
    assert deps['d0b4b_contract_sha256'] == freeze_mod.EXPECTED['d0b4b']
    assert deps['d0b4_final_freeze_sha256'] == freeze_mod.EXPECTED['d0b4_final']
    assert deps['d0a_normalization_contract_sha256'] == freeze_mod.EXPECTED['d0a']
    assert deps['e1c_checkpoint_sha256'] == freeze_mod.EXPECTED['e1c']
    assert deps['validation_split_sha256'] == freeze_mod.EXPECTED['validation_split']
    assert deps['d0c2_best_checkpoint_sha256'] == freeze_mod.EXPECTED['c2_best']
    assert deps['d0c2_best_checkpoint_bytes_verified_by_design_freeze'] is False
