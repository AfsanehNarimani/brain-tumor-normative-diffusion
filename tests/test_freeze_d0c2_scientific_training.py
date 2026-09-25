"""CPU-only tests for the D0-C2 scientific-run freeze verifier."""
import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import freeze_d0c2_scientific_training as freeze


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + '\n')


def synthetic_run(root):
    root.mkdir(parents=True)
    for name in freeze.ARTIFACT_FILES:
        path = root / name
        if name in ('best.pt', 'latest.pt'):
            payload = dict(epoch_index=99, human_epoch=100, successful_optimizer_step=13499,
                best_primary_validation_metric=freeze.EXPECTED_PRIMARY, best_epoch=99,
                next_sample_cursor=1080)
            if name == 'best.pt':
                payload['primary_validation_metric'] = freeze.EXPECTED_PRIMARY
            torch.save(payload, path)
        elif name == 'training_history.csv':
            fields = ('epoch_index', 'human_epoch', 'samples_presented', 'successful_optimizer_steps',
                      'gradscaler_skipped_steps', 'primary_validation_metric',
                      'secondary_global_support_weighted_mse')
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fields, lineterminator='\n')
                writer.writeheader()
                for human in range(1, 101):
                    validation_index = human // 5 - 1
                    validation = human % 5 == 0
                    primary = (freeze.EXPECTED_PRIMARY + (19-validation_index)*0.001) if validation else ''
                    secondary = (freeze.EXPECTED_SECONDARY + (19-validation_index)*0.001) if validation else ''
                    writer.writerow(dict(epoch_index=human-1, human_epoch=human, samples_presented=1080,
                        successful_optimizer_steps=human*135-(1 if human > 90 else 0),
                        gradscaler_skipped_steps=1 if human == 91 else 0,
                        primary_validation_metric=primary, secondary_global_support_weighted_mse=secondary))
        elif name == 'training_crop_manifest.csv':
            fields = ('case_id', 'epoch_index')
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fields, lineterminator='\n')
                writer.writeheader()
                for epoch in range(100):
                    for case in range(1080):
                        writer.writerow(dict(case_id=f'case-{case}', epoch_index=epoch))
        elif name == 'validation_summary.json':
            _write_json(path, dict(scheduled_validation_epochs_completed=freeze.HUMAN_VALIDATION_EPOCHS,
                best_human_epoch=100, best_primary_metric=freeze.EXPECTED_PRIMARY,
                latest_validation_primary_metric=freeze.EXPECTED_PRIMARY,
                latest_secondary_global_support_weighted_mse=freeze.EXPECTED_SECONDARY,
                validation_role='DEVELOPMENT_VALIDATION', independent_final_evaluation='separate/TBD'))
        elif name == 'training_provenance.json':
            event = dict(event_type='epoch_boundary_resume', checkpoint_human_epoch=90,
                checkpoint_epoch_index=89, checkpoint_successful_optimizer_step=12150,
                checkpoint_sha256=freeze.RECOVERY_CHECKPOINT_SHA256,
                checkpoint_path=str(root / 'latest.pt'),
                initial_runtime_git_commit=freeze.ORIGINAL_COMMIT,
                resume_runtime_git_commit=freeze.RESUME_COMMIT,
                dataset_root='/locked/brats', validation_manifest_sha256=freeze.EXPECTED_SHA256['validation_manifest.csv'],
                d0b4_final_freeze_sha256='05d13292994b59f3c45d70c98591c306970269673e475797620c644446aa86d8')
            _write_json(path, dict(runtime_git_commit=freeze.ORIGINAL_COMMIT, dataset_root='/locked/brats',
                validation_manifest_sha256=freeze.EXPECTED_SHA256['validation_manifest.csv'],
                validation_role='DEVELOPMENT_VALIDATION', independent_final_evaluation='separate/TBD',
                resume_events=[event]))
        elif name == 'validation_manifest.csv':
            path.write_bytes((freeze.ROOT/'outputs/frozen/d0b4d_validation_manifest/validation_manifest.csv').read_bytes())
        else:
            path.write_bytes((name + '\n').encode())
    return {name: freeze.sha256_file(root/name) for name in freeze.ARTIFACT_FILES}


@pytest.fixture
def completed_run(tmp_path):
    source = tmp_path / 'scientific_run'
    hashes = synthetic_run(source)
    return source, hashes


def test_exact_expected_source_sha_contracts():
    assert freeze.EXPECTED_SHA256 == {
        'best.pt': 'aaa6258f26a56c852dc751b8028bfb046da5752974d4cbaf446ad2699a0c6574',
        'latest.pt': '695ac64dcafb01c3621eb767abd1c984d3312f300523ce8168e03bdefbc71a4f',
        'training_history.csv': '988388fce7e005a0c28a2f39bee0f3a5754ebbb62aff50355409936addf483db',
        'training_crop_manifest.csv': '6a16dee6b2b7fbac422bebf3756e0dd9dc5ea174f17252ea6cc57c4ba9ba5cfa',
        'validation_per_case.csv': 'b8fa3bbafbfad6d6779ac416abddb8dc5fb25f3c030d6d39a2c88de60c45fc78',
        'validation_summary.json': 'a001d15857813d6fd9913fbbd48c895d4aa84b39c9afdeb68a70d4d1316c394f',
        'training_provenance.json': 'd35c2353a86af3dae2257c236e0ad19fa56c700a320753d795fd5f17c2d6bb4b',
        'validation_manifest.csv': '2a3a10d0d7b6b34bc412cfee5e30f45912161528ff509e69193f3516af3baee4',
        'training.log': '4c02995823112f36f1cbd7bc78e7fabe92b4ff4959068fefb634bda479801c45',
    }


def test_checkpoint_metadata_is_hard_checked(completed_run):
    source, hashes = completed_run
    best = freeze._validate_checkpoint(source/'best.pt', best=True)
    latest = freeze._validate_checkpoint(source/'latest.pt', best=False)
    assert best['epoch_index'] == latest['epoch_index'] == 99
    assert best['human_epoch'] == latest['human_epoch'] == 100
    assert best['successful_optimizer_step'] == latest['successful_optimizer_step'] == 13499
    assert best['next_sample_cursor'] == latest['next_sample_cursor'] == 1080
    bad = torch.load(source/'latest.pt', map_location='cpu', weights_only=False)
    bad['human_epoch'] = 99
    torch.save(bad, source/'bad.pt')
    with pytest.raises(ValueError, match='human_epoch'):
        freeze._validate_checkpoint(source/'bad.pt', best=False)


def test_completed_run_has_expected_counts_and_20_point_trajectory(completed_run):
    source, hashes = completed_run
    audit = freeze.audit_source_run(source, expected_hashes=hashes)
    assert len(audit['history']) == 100
    assert audit['crop_rows'] == 108000
    assert len(audit['trajectory']) == 20
    assert [row['human_epoch'] for row in audit['trajectory']] == list(range(5, 101, 5))
    assert audit['trajectory'][-1]['primary_validation_metric'] == freeze.EXPECTED_PRIMARY
    assert audit['trajectory'][-1]['secondary_global_support_weighted_mse'] == freeze.EXPECTED_SECONDARY
    assert int(audit['history'][-1]['successful_optimizer_steps']) == 13499
    assert sum(int(row['gradscaler_skipped_steps']) for row in audit['history']) == 1


def test_validation_trajectory_must_be_strictly_decreasing(completed_run):
    source, hashes = completed_run
    history = freeze._read_csv(source/'training_history.csv')
    summary = json.loads((source/'validation_summary.json').read_text())
    history[9]['primary_validation_metric'] = history[4]['primary_validation_metric']
    with pytest.raises(ValueError, match='strictly decreasing'):
        freeze._validation_trajectory(history, summary)


def test_recovery_event_is_verified(completed_run):
    source, hashes = completed_run
    assert hashes['latest.pt'] != freeze.RECOVERY_CHECKPOINT_SHA256
    provenance = json.loads((source/'training_provenance.json').read_text())
    event = freeze._validate_recovery(provenance, source=source,
        manifest_sha=freeze.EXPECTED_SHA256['validation_manifest.csv'])
    assert event['checkpoint_human_epoch'] == 90
    assert event['checkpoint_epoch_index'] == 89
    assert event['checkpoint_successful_optimizer_step'] == 12150
    assert event['checkpoint_sha256'] == freeze.RECOVERY_CHECKPOINT_SHA256
    bad = dict(provenance)
    bad['resume_events'] = [dict(event, resume_runtime_git_commit='other')]
    with pytest.raises(ValueError, match='resume_runtime_git_commit'):
        freeze._validate_recovery(bad, source=source,
            manifest_sha=freeze.EXPECTED_SHA256['validation_manifest.csv'])


@pytest.mark.parametrize(('target', 'field', 'value', 'message'), [
    ('event', 'checkpoint_sha256', 'bad-historical-sha', 'checkpoint_sha256'),
    ('event', 'checkpoint_human_epoch', 89, 'checkpoint_human_epoch'),
    ('event', 'checkpoint_epoch_index', 88, 'checkpoint_epoch_index'),
    ('event', 'checkpoint_successful_optimizer_step', 12149, 'checkpoint_successful_optimizer_step'),
    ('event', 'initial_runtime_git_commit', 'wrong-original', 'initial_runtime_git_commit'),
    ('event', 'resume_runtime_git_commit', 'wrong-resume', 'resume_runtime_git_commit'),
    ('event', 'validation_manifest_sha256', 'wrong-manifest', 'validation_manifest_sha256'),
    ('provenance', 'runtime_git_commit', 'wrong-original', 'Original scientific-run implementation commit'),
])
def test_recovery_attestation_fields_are_strictly_checked(completed_run, target, field, value, message):
    source, _ = completed_run
    provenance = json.loads((source/'training_provenance.json').read_text())
    if target == 'event':
        provenance['resume_events'][0][field] = value
    else:
        provenance[field] = value
    with pytest.raises(ValueError, match=message):
        freeze._validate_recovery(provenance, source=source,
            manifest_sha=freeze.EXPECTED_SHA256['validation_manifest.csv'])


def test_current_latest_sha_remains_exact_and_hard_fails(completed_run):
    source, hashes = completed_run
    assert freeze.EXPECTED_SHA256['latest.pt'] == (
        '695ac64dcafb01c3621eb767abd1c984d3312f300523ce8168e03bdefbc71a4f')
    wrong = dict(hashes, **{'latest.pt': '0'*64})
    with pytest.raises(ValueError, match='SHA256 mismatch for latest.pt'):
        freeze.audit_source_run(source, expected_hashes=wrong)


def test_freeze_references_original_artifacts_and_verify_only_is_read_only(completed_run, tmp_path):
    source, hashes = completed_run
    output = tmp_path/'frozen'
    source_before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    summary = freeze.freeze(source, output, expected_hashes=hashes,
        runtime=dict(runtime_git_branch='test', runtime_git_commit='abc', git_status_short=[]))
    assert summary['stage'] == 'D0-C2' and summary['status'] == 'CLOSED_FROZEN'
    assert summary['epochs'] == 100 and summary['sample_presentations'] == 108000
    assert summary['batch_opportunities'] == 13500
    assert summary['successful_optimizer_steps'] == 13499
    assert summary['gradscaler_skipped_updates'] == 1
    assert summary['best_human_epoch'] == 100
    assert summary['best_primary_validation_metric'] == freeze.EXPECTED_PRIMARY
    assert summary['best_checkpoint_path'] == 'outputs/scientific/d0_latent_diffusion_v1/best.pt'
    assert summary['validation_role'] == 'DEVELOPMENT_VALIDATION'
    assert summary['independent_final_evaluation'] == 'separate/TBD'
    assert summary['interpretation_guards']['za_proven_healthy'] is False
    assert summary['interpretation_guards']['za_proven_pathology_free'] is False
    assert summary['interpretation_guards']['deltaz_or_pathology_analysis_performed'] is False
    historical = summary['historical_recovery_checkpoint']
    assert historical['checkpoint_type'] == 'rolling_latest_checkpoint'
    assert historical['historical_checkpoint_human_epoch'] == 90
    assert historical['historical_checkpoint_sha256'] == freeze.RECOVERY_CHECKPOINT_SHA256
    assert historical['historical_checkpoint_bytes_preserved'] is False
    assert historical['historical_checkpoint_independently_reverified_at_freeze'] is False
    assert historical['verification_basis'] == 'resume-event provenance captured before continuation'
    assert historical['current_path_human_epoch'] == 100
    assert historical['current_path_sha256'] == hashes['latest.pt']
    records = json.loads((output/'d0c2_artifact_manifest.json').read_text())['artifacts']
    assert records['best.pt']['path'] == str(source/'best.pt')
    assert records['best.pt']['sha256'] == hashes['best.pt']
    assert not (output/'best.pt').exists() and not (output/'latest.pt').exists()
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    verified = freeze.freeze(source, output, verify_only=True, expected_hashes=hashes)
    after = {p.name: p.read_bytes() for p in output.iterdir()}
    assert verified['source_audit']['crop_rows'] == 108000
    assert before == after
    assert (output/'MANIFEST.sha256').is_file()
    assert source_before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_refuses_to_overwrite_existing_frozen_output(completed_run, tmp_path):
    source, hashes = completed_run
    output = tmp_path/'existing'
    output.mkdir()
    marker = output/'keep.txt'
    marker.write_text('preserve')
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        freeze.freeze(source, output, expected_hashes=hashes)
    assert marker.read_text() == 'preserve'


def test_verify_only_rejects_frozen_manifest_tampering(completed_run, tmp_path):
    source, hashes = completed_run
    output = tmp_path/'frozen'
    freeze.freeze(source, output, expected_hashes=hashes,
        runtime=dict(runtime_git_branch='test', runtime_git_commit='abc', git_status_short=[]))
    with (output/'d0c2_training_summary.json').open('a') as stream:
        stream.write('tampered')
    with pytest.raises(ValueError, match='MANIFEST hash mismatch'):
        freeze.freeze(source, output, verify_only=True, expected_hashes=hashes)


def test_source_artifact_hash_mismatch_hard_fails(completed_run):
    source, hashes = completed_run
    wrong = dict(hashes, **{'best.pt': '0'*64})
    with pytest.raises(ValueError, match='SHA256 mismatch for best.pt'):
        freeze.audit_source_run(source, expected_hashes=wrong)
