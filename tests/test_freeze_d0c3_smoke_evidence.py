import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/freeze_d0c3_smoke_evidence.py'
SPEC = importlib.util.spec_from_file_location('freeze_d0c3_smoke_evidence', SCRIPT)
fm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fm)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + '\n', encoding='utf-8')


def make_smoke_run(root, name, *, status='PASS', scientific_analysis=False,
                   scientific_result=False, ground_truth_opened=False,
                   corruption=False, source_unchanged=True, complete=True,
                   delta_value=0.125, support_value=True, indices=(0, 1),
                   replicate_bytes=b'replicate provenance\n'):
    run_dir = Path(root) / name
    case_dir = run_dir / 'smoke_cases' / fm.CASE_ID
    case_dir.mkdir(parents=True)
    identity = dict(mode='smoke', scientific_analysis=False, scientific_result=False,
        dataset_root='/data/brats', b4b_sha256=fm.EXPECTED['d0b4b_contract_sha256'],
        spatial_contract_sha256=fm.EXPECTED['spatial_contract_sha256'],
        spatial_manifest_sha256=fm.EXPECTED['spatial_manifest_sha256'],
        d0a_sha256=fm.EXPECTED['d0a_normalization_contract_sha256'],
        validation_split_sha256=fm.EXPECTED['validation_split_sha256'],
        e1c_checkpoint_sha256=fm.EXPECTED['e1c_checkpoint_sha256'],
        d0c2_checkpoint_sha256=fm.EXPECTED['d0c2_checkpoint_sha256'])
    payload = dict(local_DeltaZ=torch.full((2, 1, 12, 12, 12), delta_value, dtype=torch.float32),
        local_support=torch.full((2, 1, 12, 12, 12), support_value, dtype=torch.bool),
        processed_window_indices=torch.tensor(indices, dtype=torch.int64),
        complete_case_aggregation=complete)
    torch.save(payload, case_dir / 'case_deltaz.pt')
    _write_json(case_dir / 'case_provenance.json', dict(case_id=fm.CASE_ID, window_count=2,
        complete_case=complete, ground_truth_used=ground_truth_opened,
        pathology_corruption_invoked=corruption, source_model_unchanged=source_unchanged,
        diffusion_checkpoint_sha256=fm.EXPECTED['d0c2_checkpoint_sha256'],
        source_checkpoint_sha256=fm.EXPECTED['e1c_checkpoint_sha256'],
        b4b_contract_sha256=fm.EXPECTED['d0b4b_contract_sha256'],
        d0a_contract_sha256=fm.EXPECTED['d0a_normalization_contract_sha256'],
        spatial_manifest_sha256=fm.EXPECTED['spatial_manifest_sha256'],
        spatial_contract_sha256=fm.EXPECTED['spatial_contract_sha256'],
        split_sha256=fm.EXPECTED['validation_split_sha256'],
        run_identity=identity))
    (case_dir / 'replicate_provenance.csv').write_bytes(replicate_bytes)
    checksum_lines = []
    for file_name in fm.CASE_FILES:
        checksum_lines.append(f'{fm.sha256_file(case_dir / file_name)}  {file_name}')
    (case_dir / 'CHECKSUMS.sha256').write_text('\n'.join(checksum_lines) + '\n')
    with (run_dir / 'smoke_case_index.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'status', 'window_count'], lineterminator='\n')
        writer.writeheader(); writer.writerow(dict(case_id=fm.CASE_ID, status='completed', window_count=2))
    _write_json(run_dir / 'smoke_run_provenance.json', dict(status=status,
        scientific_analysis=scientific_analysis, scientific_result=scientific_result,
        ground_truth_opened=ground_truth_opened, pathology_corruption_invoked=corruption,
        source_model_unchanged=source_unchanged,
        runtime=dict(runtime_git_commit=fm.RUN_COMMIT, git_status_short=[],
            device_name=fm.EXPECTED_GPU, pytorch_version=fm.EXPECTED_PYTORCH, cuda_version=fm.EXPECTED_CUDA),
        run_identity=identity,
        checkpoint_info=dict(d0c2_checkpoint_sha256=fm.EXPECTED['d0c2_checkpoint_sha256'],
            e1c_checkpoint_sha256=fm.EXPECTED['e1c_checkpoint_sha256'],
            diffusion_state_key='ema_state', raw_state_key_loaded=False)))
    return run_dir


@pytest.fixture
def valid_smoke_root(tmp_path):
    root = tmp_path / 'smoke'
    make_smoke_run(root, 'smoke_02', complete=False)
    make_smoke_run(root, 'smoke_03', complete=False)
    return root


def test_both_runs_must_pass_and_be_non_scientific(valid_smoke_root):
    summary, _, _, _ = fm.audit_smoke_runs(valid_smoke_root)
    assert summary['smoke_runs'] == ['smoke_02', 'smoke_03']
    assert summary['scientific_result'] is False
    assert summary['scientific_analysis'] is False
    for name in fm.RUN_NAMES:
        path = valid_smoke_root / name / 'smoke_run_provenance.json'
        value = json.loads(path.read_text())
        value['status'] = 'FAILED'
        _write_json(path, value)
        with pytest.raises(ValueError, match='did not PASS'):
            fm.audit_smoke_runs(valid_smoke_root)
        value['status'] = 'PASS'
        _write_json(path, value)


@pytest.mark.parametrize('field,value,message', [
    ('scientific_result', True, 'non-scientific'),
    ('scientific_analysis', True, 'non-scientific'),
    ('ground_truth_opened', True, 'GT/corruption'),
    ('pathology_corruption_invoked', True, 'GT/corruption'),
    ('source_model_unchanged', False, 'GT/corruption'),
])
def test_safety_flags_are_hard_requirements(valid_smoke_root, field, value, message):
    path = valid_smoke_root / 'smoke_02' / 'smoke_run_provenance.json'
    provenance = json.loads(path.read_text())
    provenance[field] = value
    _write_json(path, provenance)
    with pytest.raises(ValueError, match=message):
        fm.audit_smoke_runs(valid_smoke_root)


@pytest.mark.parametrize('mutation', ['delta', 'support', 'indices'])
def test_exact_tensor_reproducibility_fails_on_any_changed_tensor(valid_smoke_root, mutation):
    path = valid_smoke_root / 'smoke_03' / 'smoke_cases' / fm.CASE_ID / 'case_deltaz.pt'
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if mutation == 'delta':
        payload['local_DeltaZ'][0, 0, 0, 0, 0] += 0.001
    elif mutation == 'support':
        payload['local_support'][0, 0, 0, 0, 0] = False
    else:
        payload['processed_window_indices'][0] = 2
    torch.save(payload, path)
    case_dir = path.parent
    (case_dir / 'CHECKSUMS.sha256').write_text('\n'.join(
        f'{fm.sha256_file(case_dir / filename)}  {filename}' for filename in fm.CASE_FILES) + '\n')
    with pytest.raises(ValueError, match='differ|different'):
        fm.audit_smoke_runs(valid_smoke_root)


def test_replicate_provenance_must_be_byte_identical(valid_smoke_root):
    path = valid_smoke_root / 'smoke_03' / 'smoke_cases' / fm.CASE_ID / 'replicate_provenance.csv'
    path.write_bytes(b'different bytes\n')
    case_dir = path.parent
    (case_dir / 'CHECKSUMS.sha256').write_text('\n'.join(
        f'{fm.sha256_file(case_dir / filename)}  {filename}' for filename in fm.CASE_FILES) + '\n')
    with pytest.raises(ValueError, match='replicate_provenance.csv differs'):
        fm.audit_smoke_runs(valid_smoke_root)


def test_full_case_flag_is_read_and_recorded_from_artifact(valid_smoke_root):
    # Do not infer this field from rerun equality. The stored value is authoritative.
    for name in fm.RUN_NAMES:
        case_dir = valid_smoke_root / name / 'smoke_cases' / fm.CASE_ID
        payload = torch.load(case_dir / 'case_deltaz.pt', map_location='cpu', weights_only=True)
        payload['complete_case_aggregation'] = True
        torch.save(payload, case_dir / 'case_deltaz.pt')
        case_prov = json.loads((case_dir / 'case_provenance.json').read_text())
        case_prov['complete_case'] = True
        _write_json(case_dir / 'case_provenance.json', case_prov)
        (case_dir / 'CHECKSUMS.sha256').write_text('\n'.join(
            f'{fm.sha256_file(case_dir / filename)}  {filename}' for filename in fm.CASE_FILES) + '\n')
    summary, _, reproducibility, _ = fm.audit_smoke_runs(valid_smoke_root)
    assert summary['complete_case_aggregation'] is True
    assert summary['complete_case_aggregation_equal_across_reruns'] is True
    assert reproducibility['complete_case_aggregation']['run_values'] == {'smoke_02': True, 'smoke_03': True}


def test_freeze_references_hashes_without_copying_tensors_and_verify_only(valid_smoke_root, tmp_path):
    source_before = {str(path): fm.sha256_file(path) for path in valid_smoke_root.rglob('*') if path.is_file()}
    out = tmp_path / 'frozen'
    result = fm.freeze(valid_smoke_root, out)
    assert result['status'] == 'CLOSED_FROZEN_PASS'
    assert {path.name for path in out.iterdir()} == set(fm.FREEZE_FILES) | {'MANIFEST.sha256'}
    assert not list(out.rglob('case_deltaz.pt'))
    artifact_manifest = json.loads((out / 'd0c3_smoke_artifact_manifest.json').read_text())
    for name in fm.RUN_NAMES:
        record = artifact_manifest['smoke_runs'][name]
        assert record['source_run_path'].endswith(name)
        tensor_record = record['files'][f'smoke_cases/{fm.CASE_ID}/case_deltaz.pt']
        assert tensor_record['path'].endswith('case_deltaz.pt')
        assert tensor_record['sha256']
    before_verify = {path.name: fm.sha256_file(path) for path in out.iterdir()}
    verify_result = fm.freeze(valid_smoke_root, out, verify_only=True)
    assert verify_result['status'] == 'CLOSED_FROZEN_PASS'
    assert before_verify == {path.name: fm.sha256_file(path) for path in out.iterdir()}
    assert source_before == {str(path): fm.sha256_file(path) for path in valid_smoke_root.rglob('*') if path.is_file()}
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        fm.freeze(valid_smoke_root, out)


def test_verify_only_rehashes_source_smoke_artifacts(valid_smoke_root, tmp_path):
    out = tmp_path / 'frozen'
    fm.freeze(valid_smoke_root, out)
    tensor_path = valid_smoke_root / 'smoke_02' / 'smoke_cases' / fm.CASE_ID / 'case_deltaz.pt'
    payload = torch.load(tensor_path, map_location='cpu', weights_only=True)
    payload['local_DeltaZ'][0, 0, 0, 0, 0] += 1
    torch.save(payload, tensor_path)
    with pytest.raises(ValueError, match='checksum mismatch'):
        fm.freeze(valid_smoke_root, out, verify_only=True)


def test_case_artifact_sha_mismatch_fails(valid_smoke_root):
    checksum = valid_smoke_root / 'smoke_02' / 'smoke_cases' / fm.CASE_ID / 'CHECKSUMS.sha256'
    checksum.write_text('0' * 64 + '  case_deltaz.pt\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        fm.audit_smoke_runs(valid_smoke_root)
