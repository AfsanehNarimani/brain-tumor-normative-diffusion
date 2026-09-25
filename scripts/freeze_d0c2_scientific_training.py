#!/usr/bin/env python3
"""Freeze D0-C2 metadata for a completed scientific diffusion run.

Scientific checkpoints and run artifacts remain in place; this tool writes
only small summary/provenance metadata into a new frozen directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'outputs/scientific/d0_latent_diffusion_v1'
DEFAULT_OUTPUT = ROOT / 'outputs/frozen/d0c2_scientific_diffusion_training'
EXPECTED_SHA256 = {
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
EXPECTED_PRIMARY = 0.7276276773324719
EXPECTED_SECONDARY = 0.7322546858874242
ORIGINAL_COMMIT = '9c5dd790f879c3fdf26328f3cdbad9a610e8971d'
RESUME_COMMIT = '55ae6a755ed82a8ab5dadf969121efa44d054c85'
RECOVERY_CHECKPOINT_SHA256 = '8f9ef4a1c1eb7797d8c3fc9d70f192cec1bc92c48144376f0d22537b88a5f409'
RECOVERY_INTERPRETATION = (
    'Implementation was corrected during an epoch-boundary recovery to honor '
    'the already-frozen GradScaler skipped-step semantics. The frozen scientific '
    'protocol was not changed.'
)
HUMAN_VALIDATION_EPOCHS = list(range(5, 101, 5))
ARTIFACT_FILES = tuple(EXPECTED_SHA256)
FROZEN_FILES = ('d0c2_training_summary.json', 'd0c2_artifact_manifest.json',
                'd0c2_validation_trajectory.csv', 'provenance.json')
TRAJECTORY_FIELDS = ('human_epoch', 'epoch_index', 'primary_validation_metric',
                     'secondary_global_support_weighted_mse')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _check_artifact_hashes(source, expected_hashes):
    actual = {}
    for name in ARTIFACT_FILES:
        path = Path(source) / name
        require(path.is_file(), f'Missing scientific run artifact: {path}')
        actual[name] = sha256_file(path)
        require(actual[name] == expected_hashes[name],
                f'SHA256 mismatch for {name}: {actual[name]} != {expected_hashes[name]}')
    return actual


def _read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _validate_checkpoint(path, *, best, expected_primary=EXPECTED_PRIMARY):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    expected = {'epoch_index': 99, 'human_epoch': 100, 'successful_optimizer_step': 13499,
                'best_epoch': 99, 'next_sample_cursor': 1080}
    for field, value in expected.items():
        require(checkpoint.get(field) == value,
                f'{Path(path).name} metadata mismatch for {field}: {checkpoint.get(field)!r} != {value!r}')
    require(checkpoint.get('best_primary_validation_metric') == expected_primary,
            f'{Path(path).name} best primary validation metric mismatch')
    if best:
        require(checkpoint.get('primary_validation_metric') == expected_primary,
                'best.pt primary validation metric mismatch')
    return checkpoint


def _validate_recovery(provenance, *, source, manifest_sha):
    require(provenance.get('runtime_git_commit') == ORIGINAL_COMMIT,
            'Original scientific-run implementation commit mismatch')
    events = provenance.get('resume_events', [])
    require(len(events) == 1, f'Expected exactly one epoch-boundary resume event; found {len(events)}')
    event = events[0]
    expected = {
        'event_type': 'epoch_boundary_resume',
        'checkpoint_human_epoch': 90,
        'checkpoint_epoch_index': 89,
        'checkpoint_successful_optimizer_step': 12150,
        'checkpoint_sha256': RECOVERY_CHECKPOINT_SHA256,
        'resume_runtime_git_commit': RESUME_COMMIT,
        'initial_runtime_git_commit': ORIGINAL_COMMIT,
        'dataset_root': provenance.get('dataset_root'),
        'validation_manifest_sha256': manifest_sha,
        'd0b4_final_freeze_sha256': '05d13292994b59f3c45d70c98591c306970269673e475797620c644446aa86d8',
    }
    for field, value in expected.items():
        require(event.get(field) == value, f'Recovery event mismatch for {field}')
    require(event.get('checkpoint_path'), 'Recovery event is missing checkpoint path')
    recovery_path = Path(event['checkpoint_path'])
    if recovery_path.is_file():
        require(sha256_file(recovery_path) == RECOVERY_CHECKPOINT_SHA256,
                'Recovery checkpoint bytes do not match the recorded checkpoint SHA256')
    require(provenance.get('validation_manifest_sha256') == manifest_sha,
            'Run provenance validation-manifest SHA mismatch')
    return event


def _validation_trajectory(history, summary):
    require(len(history) == 100, f'Expected 100 training history rows; found {len(history)}')
    require([int(row['human_epoch']) for row in history] == list(range(1, 101)),
            'Training history must contain human epochs 1..100 exactly once in order')
    validation_rows = []
    for row in history:
        primary = row.get('primary_validation_metric', '')
        if primary not in ('', 'None', 'null'):
            validation_rows.append(dict(human_epoch=int(row['human_epoch']),
                epoch_index=int(row['epoch_index']), primary_validation_metric=float(primary),
                secondary_global_support_weighted_mse=float(row['secondary_global_support_weighted_mse'])))
    require([row['human_epoch'] for row in validation_rows] == HUMAN_VALIDATION_EPOCHS,
            'Expected exactly 20 scheduled validation points')
    primaries = [row['primary_validation_metric'] for row in validation_rows]
    require(all(math.isfinite(value) for value in primaries), 'Non-finite primary validation trajectory')
    require(all(a > b for a, b in zip(primaries, primaries[1:])),
            'Observed primary validation trajectory is not strictly decreasing')
    require(primaries[-1] == EXPECTED_PRIMARY, 'Final primary validation metric mismatch')
    require(validation_rows[-1]['secondary_global_support_weighted_mse'] == EXPECTED_SECONDARY,
            'Final secondary validation metric mismatch')
    require(summary.get('scheduled_validation_epochs_completed') == HUMAN_VALIDATION_EPOCHS,
            'Validation summary schedule mismatch')
    require(summary.get('best_human_epoch') == 100, 'Best human epoch mismatch')
    require(summary.get('best_primary_metric') == EXPECTED_PRIMARY, 'Best primary metric mismatch')
    require(summary.get('latest_validation_primary_metric') == EXPECTED_PRIMARY,
            'Latest primary validation metric mismatch')
    require(summary.get('latest_secondary_global_support_weighted_mse') == EXPECTED_SECONDARY,
            'Latest secondary validation metric mismatch')
    require(summary.get('validation_role') == 'DEVELOPMENT_VALIDATION', 'Validation role mismatch')
    require(summary.get('independent_final_evaluation') == 'separate/TBD',
            'Independent final evaluation interpretation mismatch')
    return validation_rows


def audit_source_run(source, *, expected_hashes=EXPECTED_SHA256):
    source = Path(source).expanduser().resolve()
    actual_hashes = _check_artifact_hashes(source, expected_hashes)
    best = _validate_checkpoint(source / 'best.pt', best=True)
    latest = _validate_checkpoint(source / 'latest.pt', best=False)
    history = _read_csv(source / 'training_history.csv')
    crop_rows = _read_csv(source / 'training_crop_manifest.csv')
    require(len(crop_rows) == 108000, f'Expected 108000 training crop rows; found {len(crop_rows)}')
    require(set(int(row['epoch_index']) for row in crop_rows) == set(range(100)),
            'Training crop manifest epoch coverage mismatch')
    counts = {}
    for row in crop_rows:
        epoch = int(row['epoch_index'])
        counts[epoch] = counts.get(epoch, 0) + 1
    require(counts == dict.fromkeys(range(100), 1080), 'Training crop rows per epoch mismatch')
    require(sum(int(row['samples_presented']) for row in history) == 108000,
            'Training sample presentation count mismatch')
    require(len(history) * 135 == 13500, 'Batch opportunity count mismatch')
    require(int(history[-1]['successful_optimizer_steps']) == 13499,
            'Successful optimizer-step count mismatch')
    require(sum(int(row['gradscaler_skipped_steps']) for row in history) == 1,
            'GradScaler skipped-update count mismatch')
    summary = _json(source / 'validation_summary.json')
    trajectory = _validation_trajectory(history, summary)
    provenance = _json(source / 'training_provenance.json')
    require(provenance.get('validation_role') == 'DEVELOPMENT_VALIDATION',
            'Run provenance validation role mismatch')
    require(provenance.get('independent_final_evaluation') == 'separate/TBD',
            'Run provenance independent evaluation mismatch')
    event = _validate_recovery(provenance, source=source, manifest_sha=actual_hashes['validation_manifest.csv'])
    require(best['epoch_index'] == 99 and latest['epoch_index'] == 99,
            'Final checkpoints are not from epoch index 99')
    return dict(source=source, hashes=actual_hashes, best=best, latest=latest,
                history=history, crop_rows=len(crop_rows), summary=summary,
                provenance=provenance, recovery_event=event, trajectory=trajectory)


def _trajectory_bytes(rows):
    import io
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=TRAJECTORY_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode('utf-8')


def _manifest_bytes(directory):
    entries = []
    for path in sorted(Path(directory).rglob('*')):
        if path.is_file() and path.name != 'MANIFEST.sha256':
            entries.append(f'{sha256_file(path)}  {path.relative_to(directory).as_posix()}')
    return ('\n'.join(entries) + '\n').encode('utf-8')


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _freeze_metadata(audit, *, runtime=None):
    source = audit['source']
    refs = {name: {'path': str(source / name), 'sha256': digest}
            for name, digest in audit['hashes'].items()}
    artifact_manifest = dict(stage='D0-C2', artifacts=refs,
        checkpoints_copied=False,
        note='Scientific artifacts remain at their original run paths; no checkpoint copies are stored here.')
    summary = dict(stage='D0-C2', status='CLOSED_FROZEN', scientific_training_completed=True,
        epochs=100, sample_presentations=108000, batch_opportunities=13500,
        successful_optimizer_steps=13499, gradscaler_skipped_updates=1,
        best_human_epoch=100, best_primary_validation_metric=EXPECTED_PRIMARY,
        latest_secondary_global_support_weighted_mse=EXPECTED_SECONDARY,
        validation_role='DEVELOPMENT_VALIDATION', independent_final_evaluation='separate/TBD',
        best_checkpoint_path='outputs/scientific/d0_latent_diffusion_v1/best.pt',
        best_checkpoint_sha256=audit['hashes']['best.pt'],
        training_crop_manifest_rows=108000,
        training_crop_manifest_sha256=audit['hashes']['training_crop_manifest.csv'],
        scheduled_validation_epochs=HUMAN_VALIDATION_EPOCHS,
        validation_point_count=len(audit['trajectory']),
        observed_primary_validation_strictly_decreasing=True,
        monotonicity_is_descriptive_only=True,
        selection_rule='Frozen lowest-primary-validation-metric checkpoint rule; monotonicity is not a selection criterion.',
        recovery_interpretation=RECOVERY_INTERPRETATION,
        interpretation_guards=dict(validation_is_development=True,
            epoch_100_selected_by_frozen_checkpoint_rule=True,
            independent_test_performance_established=False,
            epsilon_mse_improvement_establishes_pathology_detection=False,
            za_representation='anatomy/background representation',
            za_proven_healthy=False, za_proven_pathology_free=False,
            deltaz_or_pathology_analysis_performed=False))
    if runtime is None:
        git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
        runtime = dict(runtime_git_branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
            runtime_git_commit=git('rev-parse', 'HEAD'), git_status_short=git('status', '--short').splitlines())
    provenance = dict(stage='D0-C2', frozen_at=datetime.now(timezone.utc).isoformat(),
        source_run_path=str(source), source_artifact_hashes=audit['hashes'],
        recovery_event=audit['recovery_event'], recovery_interpretation=RECOVERY_INTERPRETATION,
        training_runtime_git_commit=audit['provenance'].get('runtime_git_commit'),
        training_initial_runtime_git_commit=audit['provenance'].get('initial_runtime_git_commit',
            audit['provenance'].get('runtime_git_commit')),
        training_resume_runtime_git_commit=audit['recovery_event']['resume_runtime_git_commit'],
        training_dataset_root=audit['provenance']['dataset_root'],
        validation_manifest_sha256=audit['hashes']['validation_manifest.csv'],
        validation_role='DEVELOPMENT_VALIDATION', independent_final_evaluation='separate/TBD',
        scientific_training_completed=True, **runtime)
    return summary, artifact_manifest, _trajectory_bytes(audit['trajectory']), provenance


def verify_frozen_directory(output, *, source=DEFAULT_SOURCE, expected_hashes=EXPECTED_SHA256):
    output = Path(output).expanduser().resolve()
    manifest_path = output / 'MANIFEST.sha256'
    require(manifest_path.is_file(), 'Frozen D0-C2 MANIFEST.sha256 is missing')
    recorded = {}
    for line in manifest_path.read_text(encoding='utf-8').splitlines():
        digest, relpath = line.split('  ', 1)
        require(relpath not in recorded, f'Duplicate MANIFEST path: {relpath}')
        recorded[relpath] = digest
    actual_paths = {path.relative_to(output).as_posix() for path in output.rglob('*')
                    if path.is_file() and path.name != 'MANIFEST.sha256'}
    require(set(recorded) == actual_paths, 'Frozen MANIFEST file set mismatch')
    require(actual_paths == set(FROZEN_FILES), 'Frozen directory contains unexpected or missing metadata files')
    for relpath, digest in recorded.items():
        require(sha256_file(output / relpath) == digest, f'Frozen MANIFEST hash mismatch: {relpath}')
    audit = audit_source_run(source, expected_hashes=expected_hashes)
    summary = _json(output / 'd0c2_training_summary.json')
    artifacts = _json(output / 'd0c2_artifact_manifest.json')
    provenance = _json(output / 'provenance.json')
    require(summary.get('stage') == 'D0-C2' and summary.get('status') == 'CLOSED_FROZEN',
            'Frozen summary status mismatch')
    require(summary.get('scientific_training_completed') is True
            and summary.get('epochs') == 100
            and summary.get('sample_presentations') == 108000
            and summary.get('batch_opportunities') == 13500
            and summary.get('successful_optimizer_steps') == 13499
            and summary.get('gradscaler_skipped_updates') == 1
            and summary.get('best_human_epoch') == 100
            and summary.get('best_primary_validation_metric') == EXPECTED_PRIMARY
            and summary.get('latest_secondary_global_support_weighted_mse') == EXPECTED_SECONDARY
            and summary.get('validation_role') == 'DEVELOPMENT_VALIDATION'
            and summary.get('independent_final_evaluation') == 'separate/TBD'
            and summary.get('training_crop_manifest_rows') == 108000
            and summary.get('training_crop_manifest_sha256') == audit['hashes']['training_crop_manifest.csv'],
            'Frozen summary values mismatch')
    for name, record in artifacts.get('artifacts', {}).items():
        require(record['sha256'] == audit['hashes'].get(name), f'Artifact reference SHA mismatch: {name}')
        require(Path(record['path']).resolve() == (audit['source'] / name).resolve(),
                f'Artifact reference path mismatch: {name}')
    require(len(artifacts.get('artifacts', {})) == len(ARTIFACT_FILES),
            'Artifact manifest must reference all nine run artifacts')
    require(not any((output / name).exists() for name in ('best.pt', 'latest.pt')),
            'Frozen metadata directory must not contain scientific checkpoints')
    require(provenance.get('recovery_event') == audit['recovery_event'],
            'Frozen recovery event does not match source provenance')
    require((output / 'd0c2_validation_trajectory.csv').read_bytes()
            == _trajectory_bytes(audit['trajectory']), 'Frozen validation trajectory mismatch')
    return dict(source_audit=audit, frozen_manifest=recorded)


def freeze(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT, *, verify_only=False,
           expected_hashes=EXPECTED_SHA256, runtime=None):
    output = Path(output).expanduser().resolve()
    if verify_only:
        return verify_frozen_directory(output, source=source, expected_hashes=expected_hashes)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing frozen output: {output}')
    audit = audit_source_run(source, expected_hashes=expected_hashes)
    summary, artifacts, trajectory, provenance = _freeze_metadata(audit, runtime=runtime)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'd0c2_training_summary.json', summary)
    _write_json(output / 'd0c2_artifact_manifest.json', artifacts)
    (output / 'd0c2_validation_trajectory.csv').write_bytes(trajectory)
    _write_json(output / 'provenance.json', provenance)
    (output / 'MANIFEST.sha256').write_bytes(_manifest_bytes(output))
    verify_frozen_directory(output, source=source, expected_hashes=expected_hashes)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    result = freeze(args.source_run, args.output_dir, verify_only=args.verify_only)
    print(json.dumps(result if isinstance(result, dict) else result, indent=2, sort_keys=True, default=str))


if __name__ == '__main__':
    main()
