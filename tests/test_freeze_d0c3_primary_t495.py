import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/freeze_d0c3_primary_t495.py'
spec = importlib.util.spec_from_file_location('freeze_d0c3_primary_t495', SCRIPT)
freeze_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze_mod)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_run(tmp_path, case_ids=('case-a', 'case-b')):
    root = tmp_path / 'source'
    cases = root / 'cases'
    cases.mkdir(parents=True)
    identity = dict(mode='scientific', scientific_analysis=True, scientific_result=True,
        runtime_git_commit=freeze_mod.EXPECTED['runtime_git_commit'],
        b4b_sha256=freeze_mod.EXPECTED['b4b_contract_sha256'],
        spatial_contract_sha256=freeze_mod.EXPECTED['spatial_contract_sha256'],
        spatial_manifest_sha256=freeze_mod.EXPECTED['spatial_manifest_sha256'],
        d0a_sha256=freeze_mod.EXPECTED['d0a_contract_sha256'],
        validation_split_sha256=freeze_mod.EXPECTED['validation_split_sha256'],
        e1c_checkpoint_sha256=freeze_mod.EXPECTED['source_checkpoint_sha256'],
        d0c2_checkpoint_sha256=freeze_mod.EXPECTED['diffusion_checkpoint_sha256'],
        timestep=495, alpha_bar=0.5000776052474976, ddim_evaluations=50, eta=0.0,
        replicates=4, global_seed=20260923, case_count=len(case_ids))
    runtime = dict(runtime_git_branch=freeze_mod.EXPECTED['runtime_git_branch'],
        runtime_git_commit=freeze_mod.EXPECTED['runtime_git_commit'], git_status_short=[],
        pytorch_version='2.14.0+cu130', cuda_version='13.0', device_name='NVIDIA GeForce RTX 3090')
    run = dict(stage='D0-C3a', status='PASS', scientific_analysis=True, scientific_result=True,
        run_identity=identity, selected_source='D0-C2 EMA state_dict key ema_state',
        checkpoint_info=dict(e1c_checkpoint_path='/not-present/e1c.pt',
            e1c_checkpoint_sha256=freeze_mod.EXPECTED['source_checkpoint_sha256'],
            d0c2_checkpoint_path='/not-present/d0c2.pt',
            d0c2_checkpoint_sha256=freeze_mod.EXPECTED['diffusion_checkpoint_sha256'],
            diffusion_state_key='ema_state', raw_state_key_loaded=False),
        runtime=runtime, runtime_final=runtime, cases_processed=len(case_ids),
        windows_processed=len(case_ids) * 36, source_model_unchanged=True)
    (root / 'run_provenance.json').write_text(json.dumps(run), encoding='utf-8')
    with (root / 'case_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'status', 'window_count'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(dict(case_id=cid, status='completed', window_count=36) for cid in case_ids)
    for case_id in case_ids:
        cdir = cases / case_id
        cdir.mkdir()
        tensor = dict(DeltaZ_global=torch.ones((1, 23, 28, 23)),
            valid_support_global=torch.ones((1, 23, 28, 23), dtype=torch.bool),
            contribution_count_global=torch.ones((1, 23, 28, 23), dtype=torch.int32),
            global_latent_shape=torch.tensor([23, 28, 23]), original_shape=torch.tensor([182, 218, 182]),
            padded_shape=torch.tensor([184, 224, 184]))
        torch.save(tensor, cdir / 'case_deltaz.pt')
        provenance = dict(case_id=case_id, split='validation',
            split_sha256=freeze_mod.EXPECTED['validation_split_sha256'], window_count=36,
            timesteps=[495], alpha_bar={'495': 0.5000776052474976}, replicates=4,
            ddim_evaluations=50, eta=0.0, global_seed=20260923,
            spatial_manifest_sha256=freeze_mod.EXPECTED['spatial_manifest_sha256'],
            spatial_contract_sha256=freeze_mod.EXPECTED['spatial_contract_sha256'],
            source_checkpoint_sha256=freeze_mod.EXPECTED['source_checkpoint_sha256'],
            diffusion_checkpoint_sha256=freeze_mod.EXPECTED['diffusion_checkpoint_sha256'],
            b4b_contract_sha256=freeze_mod.EXPECTED['b4b_contract_sha256'],
            d0a_contract_sha256=freeze_mod.EXPECTED['d0a_contract_sha256'],
            source_model_unchanged=True, source_model_fingerprint_before='same',
            source_model_fingerprint_after='same', ground_truth_used=False,
            pathology_corruption_invoked=False, support_fraction_global=0.6,
            runtime=runtime)
        (cdir / 'case_provenance.json').write_text(json.dumps(provenance), encoding='utf-8')
        (cdir / 'replicate_provenance.csv').write_text('replicate_index,seed\n0,1\n', encoding='utf-8')
        checks = {name: digest(cdir / name) for name in freeze_mod.CASE_FILES}
        (cdir / 'CHECKSUMS.sha256').write_text(''.join(f'{checks[n]}  {n}\n' for n in sorted(checks)), encoding='utf-8')
    return root, case_ids


def frozen_case_manifest(tmp_path, case_ids):
    path = tmp_path / 'inference_window_manifest.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'window_index'], lineterminator='\n')
        writer.writeheader()
        for case_id in case_ids:
            for index in range(36):
                writer.writerow(dict(case_id=case_id, window_index=index))
    return path


def rewrite_case_checksums(case_dir):
    checks = {name: digest(case_dir / name) for name in freeze_mod.CASE_FILES}
    (case_dir / 'CHECKSUMS.sha256').write_text(''.join(f'{checks[n]}  {n}\n' for n in sorted(checks)), encoding='utf-8')


def test_reduced_exact_case_set_and_reference_only_freeze(tmp_path):
    source, ids = build_run(tmp_path)
    manifest = frozen_case_manifest(tmp_path, ids)
    output = tmp_path / 'freeze'
    freeze_mod.freeze(source, output, manifest, expected_case_ids=ids, verify_checkpoint_bytes=False)
    assert len(list(csv.DictReader((output / 'd0c3_primary_case_manifest.csv').open()))) == 2
    assert not list(output.rglob('*.pt'))
    assert freeze_mod.verify_freeze_manifest(output)
    assert freeze_mod.read_json(output / 'd0c3_primary_summary.json')['total_windows'] == 72


@pytest.mark.parametrize('case_ids,summary_ids', [
    (('case-a', 'case-b'), ('case-a',)),
    (('case-a',), ('case-a', 'unexpected')),
])
def test_missing_or_unexpected_case_fails(tmp_path, case_ids, summary_ids):
    source, _ = build_run(tmp_path, case_ids)
    with (source / 'case_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'window_count'], lineterminator='\n')
        writer.writeheader(); writer.writerows(dict(case_id=x, window_count=36) for x in summary_ids)
    with pytest.raises(ValueError, match='case set mismatch'):
        freeze_mod.verify_source_run(source, list(case_ids), verify_checkpoint_bytes=False)


def mutate_case(tmp_path, mutate):
    source, ids = build_run(tmp_path, ('case-a',))
    cdir = source / 'cases/case-a'
    mutate(cdir)
    return source, cdir


def test_wrong_window_timestep_split_and_spatial_identity_fail(tmp_path):
    def change(field, value):
        def mutate(cdir):
            p = cdir / 'case_provenance.json'; data = json.loads(p.read_text()); data[field] = value
            p.write_text(json.dumps(data)); rewrite_case_checksums(cdir)
        return mutate
    for field, value, message in (('window_count', 35, 'window/timestep'), ('timesteps', [327], 'window/timestep'),
                                  ('split_sha256', 'bad', 'split identity'),
                                  ('spatial_manifest_sha256', 'bad', 'spatial_manifest_sha256')):
        source, _ = mutate_case(tmp_path / field, change(field, value))
        with pytest.raises(ValueError, match=message):
            freeze_mod.verify_source_run(source, ['case-a'], verify_checkpoint_bytes=False)


def test_source_fingerprint_and_artifact_checksum_fail(tmp_path):
    source, cdir = mutate_case(tmp_path, lambda c: None)
    data = json.loads((cdir / 'case_provenance.json').read_text())
    data['source_model_fingerprint_after'] = 'changed'
    (cdir / 'case_provenance.json').write_text(json.dumps(data)); rewrite_case_checksums(cdir)
    with pytest.raises(ValueError, match='fingerprint changed'):
        freeze_mod.verify_source_run(source, ['case-a'], verify_checkpoint_bytes=False)
    source, cdir = mutate_case(tmp_path / 'artifact', lambda c: None)
    with (cdir / 'replicate_provenance.csv').open('a') as stream:
        stream.write('1,2\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        freeze_mod.verify_source_run(source, ['case-a'], verify_checkpoint_bytes=False)


def test_per_case_checksum_manifest_failure(tmp_path):
    source, cdir = mutate_case(tmp_path, lambda c: None)
    (cdir / 'CHECKSUMS.sha256').write_text('0' * 64 + '  case_deltaz.pt\n', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum inventory'):
        freeze_mod.verify_source_run(source, ['case-a'], verify_checkpoint_bytes=False)


@pytest.mark.parametrize('mutation,match', [
    ('nan', 'Non-finite supported'), ('support', 'Support/count mismatch'),
    ('negative', 'Negative contribution'), ('shape', 'Global tensor shape')])
def test_tensor_integrity_failures(tmp_path, mutation, match):
    def alter(cdir):
        path = cdir / 'case_deltaz.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if mutation == 'nan': payload['DeltaZ_global'][0, 0, 0, 0] = float('nan')
        elif mutation == 'support': payload['valid_support_global'][0, 0, 0, 0] = False
        elif mutation == 'negative': payload['contribution_count_global'][0, 0, 0, 0] = -1
        elif mutation == 'shape': payload['DeltaZ_global'] = payload['DeltaZ_global'][:, :22]
        torch.save(payload, path); rewrite_case_checksums(cdir)
    source, _ = mutate_case(tmp_path, alter)
    with pytest.raises(ValueError, match=match):
        freeze_mod.verify_source_run(source, ['case-a'], verify_checkpoint_bytes=False)


def test_verify_only_is_read_only_and_overwrite_refused(tmp_path):
    source, ids = build_run(tmp_path)
    manifest = frozen_case_manifest(tmp_path, ids)
    output = tmp_path / 'freeze'
    freeze_mod.freeze(source, output, manifest, expected_case_ids=ids, verify_checkpoint_bytes=False)
    before = {p.relative_to(output).as_posix(): digest(p) for p in output.iterdir()}
    freeze_mod.freeze(source, output, manifest, verify_only=True, expected_case_ids=ids, verify_checkpoint_bytes=False)
    after = {p.relative_to(output).as_posix(): digest(p) for p in output.iterdir()}
    assert before == after
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        freeze_mod.freeze(source, output, manifest, expected_case_ids=ids, verify_checkpoint_bytes=False)


def test_source_artifacts_unchanged_by_freeze(tmp_path):
    source, ids = build_run(tmp_path)
    manifest = frozen_case_manifest(tmp_path, ids)
    before = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob('*') if p.is_file()}
    freeze_mod.freeze(source, tmp_path / 'freeze', manifest, expected_case_ids=ids, verify_checkpoint_bytes=False)
    after = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob('*') if p.is_file()}
    assert before == after
