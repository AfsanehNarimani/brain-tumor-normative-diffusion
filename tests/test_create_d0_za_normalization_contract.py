"""Synthetic-artifact-only tests for the fixed D0-B normalization contract."""
import csv
from decimal import Decimal
import hashlib
import json
import sys
from unittest.mock import MagicMock

import pytest

from scripts import create_d0_za_normalization_contract as script


@pytest.fixture
def inputs(tmp_path):
    stats, provenance, output = (tmp_path / name for name in ('channels.csv', 'provenance.json', 'contract.json'))
    rows = [dict(channel_index=i, mean='-0.12345678901234567890123456789',
                 std='1.2345678901234567890123456789e-2') for i in reversed(range(256))]
    metadata = dict(validation_data_used=False, statistics_convention='population standard deviation, ddof=0',
                    source_model_identity='anatomy_constrained_joint_reconstruction_unet3d',
                    source_checkpoint_sha256='a'*64, train_split_sha256='b'*64, git_commit_sha='audit-commit')
    def write():
        with stats.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['channel_index', 'mean', 'std'])
            writer.writeheader()
            writer.writerows(rows)
        provenance.write_text(json.dumps(metadata))
    write()
    return stats, provenance, output, rows, metadata, write


def test_exact_values_schema_hashes_and_determinism(inputs):
    stats, provenance, output, rows, _, _ = inputs
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (stats, provenance)]
    assert script.create_contract(stats, provenance, output, *hashes) == output
    first = output.read_bytes()
    data = json.loads(first, parse_float=Decimal)
    assert data['schema_version'] == 1 and data['channel_count'] == 256
    assert data['normalization_type'] == 'fixed_per_channel_standardization'
    assert data['formula'] == '(Za[c] - mean[c]) / population_std[c]'
    assert data['source_checkpoint_sha256'] == 'a'*64 and data['train_split_sha256'] == 'b'*64
    assert data['d0a_audit_commit'] == 'audit-commit'
    assert 'without recomputation' in data['reuse_policy']
    assert 'Train-derived only' in data['statistics_origin']
    assert all(data[key] is False for key in ('clipping', 'per_case_normalization', 'per_patch_normalization'))
    for label, path, digest in zip(('latent_channel_statistics', 'analysis_provenance'), (stats, provenance), hashes):
        assert data['sources'][label] == dict(filename=path.name, path=str(path), sha256=digest)
    assert [row['channel_index'] for row in data['channels']] == list(range(256))
    for channel in data['channels']:
        assert channel['mean'] == Decimal(rows[0]['mean'])
        assert channel['population_std'] == Decimal(rows[0]['std'])
    assert first.count(rows[0]['std'].encode()) == 256
    script.create_contract(stats, provenance, output, overwrite=True)
    assert output.read_bytes() == first
    assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (stats, provenance)]


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'outside'])
def test_invalid_channel_sets(inputs, kind):
    stats, provenance, output, rows, _, write = inputs
    if kind == 'missing':
        rows.pop()
    else:
        rows[0]['channel_index'] = 0 if kind == 'duplicate' else 256
    write()
    with pytest.raises(ValueError, match='channels|Duplicate|Unexpected'):
        script.create_contract(stats, provenance, output)
    assert not output.exists()


@pytest.mark.parametrize('column,value', [('mean', 'nan'), ('mean', 'inf'), ('std', '-inf'),
                                        ('std', '0'), ('std', '-0.1'), ('std', 'NaN'), ('mean', 'invalid')])
def test_invalid_numbers(inputs, column, value):
    stats, provenance, output, rows, _, write = inputs
    rows[0][column] = value
    write()
    with pytest.raises(ValueError):
        script.create_contract(stats, provenance, output)
    assert not output.exists()


@pytest.mark.parametrize('side', ['statistics', 'provenance'])
def test_expected_hash_rejected(inputs, side):
    stats, provenance, output, *_ = inputs
    with pytest.raises(ValueError, match='SHA256'):
        script.create_contract(stats, provenance, output, **{f'expected_{side}_sha256': '0'*64})
    assert not output.exists()


@pytest.mark.parametrize('field,value', [('validation_data_used', True), ('validation_data_used', None),
    ('statistics_convention', 'sample standard deviation, ddof=1'), ('source_model_identity', 'wrong'),
    ('source_checkpoint_sha256', None), ('train_split_sha256', 'bad')])
def test_provenance_guards(inputs, field, value):
    stats, provenance, output, _, metadata, write = inputs
    metadata[field] = value
    write()
    with pytest.raises(ValueError):
        script.create_contract(stats, provenance, output)


def test_safety_and_optional_commit(inputs):
    stats, provenance, output, _, metadata, write = inputs
    del metadata['git_commit_sha']
    write()
    output.write_text('keep')
    with pytest.raises(FileExistsError):
        script.create_contract(stats, provenance, output)
    assert output.read_text() == 'keep'
    with pytest.raises(ValueError, match='source artifact'):
        script.create_contract(stats, provenance, stats, overwrite=True)
    with pytest.raises(ValueError, match='outputs/frozen'):
        script.create_contract(stats, provenance, 'outputs/frozen/forbidden.json', overwrite=True)
    script.create_contract(stats, provenance, output, overwrite=True)
    assert json.loads(output.read_text())['d0a_audit_commit'] is None
    assert not list(output.parent.glob('.normalization-contract-*'))


def test_invalid_input_preserves_existing_contract(inputs):
    stats, provenance, output, rows, _, write = inputs
    output.write_text('keep')
    rows[0]['std'] = '0'
    write()
    with pytest.raises(ValueError):
        script.create_contract(stats, provenance, output, overwrite=True)
    assert output.read_text() == 'keep'


def test_cli(monkeypatch):
    build = MagicMock()
    monkeypatch.setattr(script, 'create_contract', build)
    monkeypatch.setattr(sys, 'argv', ['contract', '--output', 'contract.json'])
    script.main()
    assert build.call_args.kwargs['output'] == 'contract.json'
    assert build.call_args.kwargs['overwrite'] is False
