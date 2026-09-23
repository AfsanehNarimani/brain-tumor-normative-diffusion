import json

import pytest

from scripts import freeze_d0b4a_diffusion_preflight as freeze


@pytest.fixture
def candidates():
    return {mb: dict(microbatch=mb, gradient_accumulation_steps=8 // mb,
                     effective_batch_size=8, model_parameter_count=22825280,
                     gpu_name=freeze.GPU, **dict.fromkeys(freeze.FLAGS, True),
                     oom=False, failure_reason=None, **freeze.NON_SCIENTIFIC,
                     **freeze.FINGERPRINTS, **freeze.SELECTED_MEMORY,
                     elapsed_seconds=mb * 10, torch_version='test-only', cuda_version='test-only',
                     seed=20260923, patch_manifest_sha256='9c4ba67fc195537a5500229722ce55e4772ea5b7d135b351b3a2fc4b857f44b9')
            for mb in freeze.MICROBATCHES}


@pytest.fixture
def inputs(tmp_path, monkeypatch, candidates):
    source = tmp_path / 'inputs'
    source.mkdir()
    for mb, payload in candidates.items():
        (source / f'microbatch_{mb}_retry1.json').write_text(json.dumps(payload) + '\n\n')
    monkeypatch.setattr(freeze, 'runtime_git_provenance', lambda: dict(
        runtime_git_branch=freeze.BRANCH, runtime_git_commit='test', git_status_short=[]))
    return source, tmp_path / 'frozen'


def test_selection_and_elapsed_time_independence(candidates):
    assert freeze.validate_candidates(candidates) == 8
    reference = freeze.decision_record(candidates)
    assert reference['selection_rule'] == freeze.RULE
    assert reference['gradient_accumulation_steps'] == 1
    assert reference['effective_batch_size'] == 8
    for mb, payload in candidates.items():
        payload['elapsed_seconds'] = 1e20 if mb == 8 else 0
    assert freeze.decision_record(candidates) == reference
    for payload in candidates.values():
        del payload['elapsed_seconds']
    assert freeze.decision_record(candidates) == reference


@pytest.mark.parametrize('field,value', [
    *[(key, False) for key in freeze.FLAGS], ('oom', True), ('failure_reason', 'failure'),
    ('effective_batch_size', 4), ('model_parameter_count', 1), ('gpu_name', 'other GPU'),
    ('gradient_accumulation_steps', 1), ('microbatch', 8),
    ('effective_batch_fingerprint', 'different'), ('randomness_fingerprint', 'different'),
    ('peak_reserved_fraction', 0.900001), ('peak_reserved_fraction', float('nan')),
    ('peak_reserved_fraction', float('inf')), ('scientific_training', True),
])
def test_reject_invalid_candidate(candidates, field, value):
    candidates[2][field] = value
    with pytest.raises(ValueError, match=field):
        freeze.validate_candidates(candidates)


def test_exact_headroom_boundary(candidates):
    candidates[1]['peak_reserved_fraction'] = 0.90
    assert freeze.validate_candidates(candidates) == 8


def test_locked_selected_memory(candidates):
    candidates[8]['peak_memory_reserved_bytes'] += 1
    with pytest.raises(ValueError, match='locked'):
        freeze.validate_candidates(candidates)


def test_missing_candidate_does_not_create_output(inputs):
    source, output = inputs
    (source / 'microbatch_1_retry1.json').unlink()
    with pytest.raises(FileNotFoundError):
        freeze.run(source, output)
    assert not output.exists()


def test_byte_preservation_provenance_and_read_only_verification(inputs, monkeypatch):
    source, output = inputs
    failed = b'{"pass": false, "failure_reason": "earlier failure"}\n'
    (source / 'microbatch_8.json').write_bytes(failed)
    originals = {p.name: p.read_bytes() for p in source.iterdir()}
    decision = freeze.run(source, output)
    assert (output / 'failed_attempts/microbatch_8.json').read_bytes() == failed
    for mb in freeze.MICROBATCHES:
        assert (output / f'candidates/microbatch_{mb}.json').read_bytes() == originals[f'microbatch_{mb}_retry1.json']
    assert {p.name: p.read_bytes() for p in source.iterdir()} == originals
    provenance = json.loads((output / 'provenance.json').read_bytes())
    assert provenance['candidate_provenance']['8']['seed'] == 20260923
    for record in provenance['source_files']:
        assert record['source_sha256'] == record['frozen_copy_sha256']
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob('*') if p.is_file()}
    # Verify-only must work independently of the source directory or git state.
    monkeypatch.setattr(freeze, 'runtime_git_provenance', lambda: pytest.fail('git consulted during verification'))
    assert freeze.run(source / 'absent', output, verify_only=True) == decision
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    lines = (output / 'MANIFEST.sha256').read_text().splitlines()
    assert [line.split('  ')[1] for line in lines] == sorted(
        p.relative_to(output).as_posix() for p in before if p.name != 'MANIFEST.sha256')


def test_failed_attempts_do_not_affect_selection(inputs):
    source, output = inputs
    first = freeze.run(source, output)
    (source / 'microbatch_16.json').write_text('{"pass": false, "elapsed_seconds": 0}')
    second = freeze.run(source, output.parent / 'second')
    assert first == second


def test_refuse_overwrite(inputs):
    source, output = inputs
    freeze.run(source, output)
    before = (output / 'MANIFEST.sha256').read_bytes()
    with pytest.raises(FileExistsError, match='overwrite'):
        freeze.run(source, output)
    assert (output / 'MANIFEST.sha256').read_bytes() == before


@pytest.mark.parametrize('mutation', ['changed', 'missing', 'extra', 'manifest'])
def test_manifest_verification(inputs, mutation):
    source, output = inputs
    freeze.run(source, output)
    if mutation == 'changed':
        (output / 'd0b4a_summary.json').write_text('{}')
    elif mutation == 'missing':
        (output / 'candidates/microbatch_1.json').unlink()
    elif mutation == 'extra':
        (output / 'extra.json').write_text('{}')
    else:
        with (output / 'MANIFEST.sha256').open('a') as stream:
            stream.write('0' * 64 + '  ../outside.json\n')
    with pytest.raises(ValueError, match='MANIFEST'):
        freeze.verify(output)


def rewrite_manifest(output):
    files = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256')
    (output / 'MANIFEST.sha256').write_text(''.join(
        f'{freeze.sha256(p.read_bytes())}  {p.relative_to(output).as_posix()}\n' for p in files))


@pytest.mark.parametrize('field,value', [('selected_physical_microbatch', 4),
    ('gradient_accumulation_steps', 2), ('selection_rule', 'fastest elapsed time'),
    ('selected_peak_memory_allocated_bytes', 0), ('scientific_training', True)])
def test_decision_consistency_even_with_rehashed_manifest(inputs, field, value):
    source, output = inputs
    freeze.run(source, output)
    path = output / 'decision/microbatch_decision.json'
    payload = json.loads(path.read_bytes())
    payload[field] = value
    path.write_bytes(freeze.encode(payload))
    rewrite_manifest(output)
    with pytest.raises(ValueError, match='Decision consistency'):
        freeze.verify(output)


def test_verify_rechecks_fingerprints(inputs):
    source, output = inputs
    freeze.run(source, output)
    path = output / 'candidates/microbatch_4.json'
    payload = json.loads(path.read_bytes())
    payload['randomness_fingerprint'] = 'changed'
    path.write_bytes(freeze.encode(payload))
    rewrite_manifest(output)
    with pytest.raises(ValueError, match='randomness_fingerprint'):
        freeze.verify(output)
