"""Synthetic CPU tests only: never generate a real BraTS validation manifest."""
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from scripts import freeze_d0b4d_validation_manifest as fm


@pytest.fixture(scope='module')
def contract():
    return json.loads(fm.CONTRACT.read_bytes())


def synthetic_audit(row, count=1728):
    return dict(case_id=row['case_id'], validation_patch_index=row['validation_patch_index'],
                raw_patch_shape='[4,96,96,96]', valid_latent_support_cells=count,
                total_latent_cells=1728, valid_support_fraction=count / 1728,
                support_nonzero=count > 0, corruption_shape_valid=True, corruption_finite=True,
                donor_fallback_modality_count=0)


def test_stable_hash_serialization_and_full_digest_integer():
    payload = [20260923, 'café', 2, 'd', 'validation_crop']
    expected = '[20260923,"café",2,"d","validation_crop"]'.encode('utf-8')
    assert fm.payload_bytes(payload) == expected
    digest = hashlib.sha256(expected).digest()
    assert fm.hash_integer(payload) == int.from_bytes(digest, 'big', signed=False)
    assert fm.generator_seed(payload) == int.from_bytes(digest[:8], 'big', signed=False) & ((1 << 63) - 1)


def test_actual_locked_sources(contract):
    loaded, ids, sources = fm.load_sources()
    assert loaded == contract and len(ids) == len(set(ids)) == 270
    assert sources['d0b4c_contract_sha256'] == '861ea070f7bfa2677b9345403905e70400190012e56efac722ccbb892af21f75'
    assert sources['d0b4c_locked_commit'] == 'cb4d830a6ce777fab7935ae77eeb75ca321db5de'
    assert sources['validation_split_sha256'] == 'ec5fe6b0bbcf190fe8b7c032f8da4ff1d495969c2d868561e7a3351c572cc5ca'
    assert sources['d0b3_decision_sha256'] == '095e1f9671411eca0d12a2f834833d78477a14664504bde3e6e1b46766c6cc2f'
    assert sources['train_validation_disjoint'] is True


@pytest.mark.parametrize('keyword', ['contract_path', 'val_split', 'train_split', 'decision_path'])
def test_source_hash_mismatch_fails(keyword, tmp_path):
    path = tmp_path / 'tampered'
    path.write_bytes(b'wrong source')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        fm.load_sources(**{keyword: path})


def test_exact_coordinates_timestep_and_noise_identity(contract):
    dims = [111, 123, 105]
    rows = fm.case_rows('case_é', dims, contract)
    assert len(rows) == 4
    for index, row in enumerate(rows):
        assert row['validation_patch_index'] == index
        for axis, size in zip('dhw', dims):
            payload = [20260923, 'case_é', index, axis, 'validation_crop']
            digest = hashlib.sha256(json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).digest()
            expected = int.from_bytes(digest, 'big') % (size - 96 + 1)
            assert row[f'crop_start_{axis}'] == expected
            assert row[f'crop_stop_{axis}'] == expected + 96 <= size
        identity = ['case_é', index, *[row[key] for key in fm.COORDS]]
        seed = fm.generator_seed([20260923, identity, 'validation_timestep'])
        expected_t = torch.randint(0, 1000, (1,), generator=torch.Generator(device='cpu').manual_seed(seed)).item()
        assert row['timestep_seed'] == seed
        assert row['timestep'] == expected_t and 0 <= expected_t <= 999
        assert row['epsilon_seed'] == fm.generator_seed([20260923, identity, 'validation_epsilon'])
        assert row['epsilon_seed'] != row['timestep_seed']
        assert row['corruption_operator'] == 'local_empirical_resample'
        assert row['corruption_global_seed'] == 20260923
        assert row['corruption_patch_index'] == index
    assert rows == fm.case_rows('case_é', dims, contract)


@pytest.mark.parametrize('dimensions', [[95, 96, 96], [96, 20, 96], [96, 96, 0], [96, 96]])
def test_small_or_invalid_dimensions_hard_error(contract, dimensions):
    with pytest.raises(ValueError, match='>=96'):
        fm.case_rows('case', dimensions, contract)


def test_canonical_1080_rows_and_exact_columns(contract):
    dims = {f'case_{i:03d}': [100 + i % 5, 105, 110] for i in reversed(range(270))}
    rows = fm.manifest_rows(dims, contract)
    assert len(rows) == 1080
    assert Counter(row['case_id'] for row in rows) == dict.fromkeys(dims, 4)
    assert [(r['case_id'], r['validation_patch_index']) for r in rows] == sorted((r['case_id'], r['validation_patch_index']) for r in rows)
    assert fm.COLUMNS == ('case_id', 'validation_patch_index', 'crop_start_d', 'crop_start_h', 'crop_start_w',
        'crop_stop_d', 'crop_stop_h', 'crop_stop_w', 'coordinate_generation_version', 'seed', 'split_sha256',
        'timestep', 'timestep_seed', 'epsilon_seed', 'corruption_operator', 'corruption_global_seed', 'corruption_patch_index')
    data = fm.canonical_csv(rows, fm.COLUMNS)
    assert data == fm.canonical_csv(list(reversed(rows)), fm.COLUMNS)
    assert data == fm.canonical_csv(fm.manifest_rows(dict(reversed(list(dims.items()))), contract), fm.COLUMNS)
    assert b'\r' not in data and data.endswith(b'\n')
    assert data.splitlines()[0].decode() == ','.join(fm.COLUMNS)
    assert len(list(csv.DictReader(io.StringIO(data.decode())))) == 1080
    assert 'epsilon' not in fm.COLUMNS  # Only its seed is stored.


def test_case_loader_dimensions_full_normalization_and_no_gt_coordinates(contract, monkeypatch):
    image = torch.ones(4, 97, 98, 99)
    normalize = Mock(wraps=fm.normalize_nonzero)
    monkeypatch.setattr(fm, 'normalize_nonzero', normalize)
    def audit(image_arg, mask_arg, normalized, row, contract_arg):
        assert image_arg is image
        assert normalized.shape == image.shape
        assert torch.count_nonzero(normalized) == 0
        return synthetic_audit(row)
    monkeypatch.setattr(fm, 'audit_patch', audit)
    monkeypatch.setattr(fm, 'load_case', lambda *a: (image, torch.zeros(97, 98, 99)))
    dims, rows, _ = fm.audit_case('unused', 'case', contract)
    assert dims == [97, 98, 99]
    assert rows == fm.case_rows('case', dims, contract)
    assert normalize.call_args.args[0] is image
    monkeypatch.setattr(fm, 'load_case', lambda *a: (image, torch.ones(97, 98, 99)))
    assert fm.audit_case('unused', 'case', contract)[1] == rows


def test_real_frozen_support_and_corruption_helpers_cpu(contract, monkeypatch):
    image = torch.ones(4, 96, 96, 96)
    image[:, :, :, 48:] = 0
    mask = torch.zeros(96, 96, 96)
    row = fm.case_rows('case', [96, 96, 96], contract)[0]
    real_support, real_occupancy = fm.support_mask, fm.block_occupancy
    observed = {}
    def support(raw):
        assert raw.shape == (4, 96, 96, 96)
        assert torch.equal(raw, image)
        observed['support'] = real_support(raw)
        return observed['support']
    def occupancy(support_arg):
        assert support_arg is observed['support']
        return real_occupancy(support_arg)
    monkeypatch.setattr(fm, 'support_mask', support)
    monkeypatch.setattr(fm, 'block_occupancy', occupancy)
    donors = Mock(wraps=fm.donor_pools)
    corruption = Mock(wraps=fm.corrupt_input)
    monkeypatch.setattr(fm, 'donor_pools', donors)
    monkeypatch.setattr(fm, 'corrupt_input', corruption)
    result = fm.audit_patch(image, mask, fm.normalize_nonzero(image), row, contract)
    assert result['valid_latent_support_cells'] == 864
    assert result['valid_support_fraction'] == 0.5
    assert result['corruption_finite'] and result['corruption_shape_valid']
    assert donors.call_args.args[2] is observed['support']
    assert corruption.call_args.args[3] == 'local_empirical_resample'
    assert corruption.call_args.kwargs == dict(global_seed=20260923, case_id='case', patch_index=0)


def test_zero_support_is_hard_error_before_corruption(contract, monkeypatch):
    image = torch.zeros(4, 96, 96, 96)
    row = fm.case_rows('case', [96, 96, 96], contract)[0]
    corrupt = Mock(side_effect=AssertionError('must not corrupt or resample'))
    monkeypatch.setattr(fm, 'corrupt_input', corrupt)
    with pytest.raises(ValueError, match='Zero valid latent support'):
        fm.audit_patch(image, torch.zeros(96, 96, 96), image, row, contract)
    corrupt.assert_not_called()
    assert row == fm.case_rows('case', [96, 96, 96], contract)[0]


def test_audit_rejects_out_of_bounds(contract):
    image = torch.zeros(4, 96, 96, 96)
    row = fm.case_rows('case', [96, 96, 96], contract)[0]
    row['crop_stop_d'] = 97
    with pytest.raises(ValueError, match='out of bounds'):
        fm.audit_patch(image, torch.zeros(96, 96, 96), image, row, contract)


@pytest.mark.parametrize('bad_output', ['shape', 'nonfinite'])
def test_corruption_executability_failures(contract, monkeypatch, bad_output):
    image = torch.ones(4, 96, 96, 96)
    row = fm.case_rows('case', [96, 96, 96], contract)[0]
    monkeypatch.setattr(fm, 'donor_pools', lambda *a: ([], [False] * 4))
    monkeypatch.setattr(fm, 'corrupt_input', lambda *a, **kw:
                        torch.zeros(1) if bad_output == 'shape' else torch.full_like(image, float('nan')))
    with pytest.raises(ValueError, match='corrupted patch'):
        fm.audit_patch(image, torch.zeros(96, 96, 96), image, row, contract)


@pytest.fixture
def synthetic_freeze(tmp_path, monkeypatch):
    # Real locked IDs/hashes, synthetic directories and audits; no BraTS data loaded.
    contract, ids, _ = fm.load_sources()
    root = tmp_path / 'synthetic_data'
    for case in ids:
        (root / case).mkdir(parents=True)
    def synthetic_case(dataset_root, case, contract_arg):
        dims = [100, 104, 108]
        rows = fm.case_rows(case, dims, contract_arg)
        audits = [synthetic_audit(row, count=864) for row in rows]
        audits[0]['donor_fallback_modality_count'] = 2
        return dims, rows, audits
    monkeypatch.setattr(fm, 'audit_case', synthetic_case)
    return root, tmp_path / 'frozen'


def test_synthetic_freeze_summary_verify_only_and_byte_stability(synthetic_freeze, monkeypatch):
    root, output = synthetic_freeze
    summary = fm.run(dataset_root=root, output_dir=output)
    assert summary['stage'] == 'D0-B4d'
    assert summary['status'] == 'CLOSED_FROZEN_PRE_TRAINING'
    assert summary['validation_role'] == 'DEVELOPMENT_VALIDATION'
    assert all(summary[key] is False for key in fm.FLAGS)
    assert (summary['case_count'], summary['patches_per_case'], summary['total_patches']) == (270, 4, 1080)
    assert summary['zero_support_patch_count'] == summary['required_zero_support_patch_count'] == 0
    assert summary['support_fraction_statistics'] == dict(mean=0.5, std=0.0, min=0.5, max=0.5,
        p01=0.5, p05=0.5, p25=0.5, p50=0.5, p75=0.5, p95=0.5, p99=0.5)
    assert summary['donor_fallback_modality_count'] == 540
    assert summary['donor_fallback_patch_count'] == 270
    second = output.parent / 'second'
    fm.run(dataset_root=root, output_dir=second)
    assert (output / 'validation_manifest.csv').read_bytes() == (second / 'validation_manifest.csv').read_bytes()
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    monkeypatch.setattr(fm, 'audit_case', lambda *a: pytest.fail('verify-only loaded data'))
    monkeypatch.setattr(fm, 'read_config', lambda *a: pytest.fail('verify-only read dataset config'))
    assert fm.run(output_dir=output, verify_only=True) == summary
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}


def test_refuse_overwrite_even_empty_directory(tmp_path):
    with pytest.raises(FileExistsError, match='overwrite'):
        fm.run(output_dir=tmp_path)


def test_audit_failure_creates_no_output(synthetic_freeze, monkeypatch):
    root, output = synthetic_freeze
    def fail(*args):
        raise ValueError('Zero valid latent support')
    monkeypatch.setattr(fm, 'audit_case', fail)
    with pytest.raises(ValueError, match='Zero valid'):
        fm.run(dataset_root=root, output_dir=output)
    assert not output.exists()


def test_missing_case_creates_no_output(synthetic_freeze):
    root, output = synthetic_freeze
    next(root.iterdir()).rmdir()
    with pytest.raises(ValueError, match='Missing validation case'):
        fm.run(dataset_root=root, output_dir=output)
    assert not output.exists()


def rehash(output):
    files = {name: (output / name).read_bytes() for name in fm.FILES}
    (output / 'MANIFEST.sha256').write_bytes(fm.manifest_bytes(files))


@pytest.mark.parametrize('mutation', ['hash', 'order', 'coordinate', 'timestep', 'epsilon', 'corruption',
    'audit_alignment', 'audit_zero', 'audit_flag', 'summary', 'provenance', 'dimensions', 'extra', 'missing'])
def test_verify_detects_tampering(synthetic_freeze, mutation):
    root, output = synthetic_freeze
    fm.run(dataset_root=root, output_dir=output)
    manifest = output / 'validation_manifest.csv'
    audit_path = output / 'validation_manifest_audit.csv'
    if mutation == 'hash':
        manifest.write_bytes(manifest.read_bytes() + b'\n')
    elif mutation == 'order':
        lines = manifest.read_bytes().splitlines(keepends=True)
        lines[1], lines[2] = lines[2], lines[1]
        manifest.write_bytes(b''.join(lines))
        rehash(output)
    elif mutation in ('coordinate', 'timestep', 'epsilon', 'corruption'):
        rows = list(csv.DictReader(io.StringIO(manifest.read_text())))
        key = dict(coordinate='crop_start_d', timestep='timestep', epsilon='epsilon_seed', corruption='corruption_patch_index')[mutation]
        rows[0][key] = str(int(rows[0][key]) + 1)
        manifest.write_bytes(fm.canonical_csv(rows, fm.COLUMNS))
        rehash(output)
    elif mutation.startswith('audit_'):
        rows = list(csv.DictReader(io.StringIO(audit_path.read_text())))
        key, value = {'audit_alignment': ('case_id', 'wrong_case'), 'audit_zero': ('valid_latent_support_cells', '0'),
                      'audit_flag': ('support_nonzero', 'false')}[mutation]
        rows[0][key] = value
        audit_path.write_bytes(fm.canonical_csv(rows, fm.AUDIT_COLUMNS))
        rehash(output)
    elif mutation in ('summary', 'provenance', 'dimensions'):
        path = output / ('validation_manifest_summary.json' if mutation == 'summary' else 'provenance.json')
        payload = json.loads(path.read_bytes())
        if mutation == 'summary':
            payload['validation_role'] = 'FINAL_TEST'
        elif mutation == 'provenance':
            payload['d0b4c_locked_commit'] = 'wrong'
        else:
            payload['case_dimensions'][next(iter(payload['case_dimensions']))] = [95, 96, 96]
        path.write_bytes(fm.json_bytes(payload))
        rehash(output)
    elif mutation == 'extra':
        (output / 'extra').write_text('unexpected')
    else:
        manifest.unlink()
    with pytest.raises(ValueError):
        fm.run(output_dir=output, verify_only=True)
