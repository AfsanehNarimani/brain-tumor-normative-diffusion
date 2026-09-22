"""Create a fixed D0-B Z_a normalization contract from completed D0-A artifacts only."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

DEFAULT_STATISTICS = 'outputs/d0_latent_distribution_audit/latent_channel_statistics.csv'
DEFAULT_PROVENANCE = 'outputs/d0_latent_distribution_audit/analysis_provenance.json'
DEFAULT_OUTPUT = 'outputs/d0_za_normalization_contract.json'
JSON_NUMBER = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z')
SHA256 = re.compile(r'[0-9a-fA-F]{64}\Z')


def read_source(path, expected_hash):
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_hash is not None and digest != expected_hash.lower():
        raise ValueError(f'Unexpected source SHA256: {path}')
    return data, dict(filename=path.name, path=str(path), sha256=digest)


def number_token(value, field):
    """Validate but never round/reformat the original CSV number token."""
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError) as error:
        raise ValueError(f'Invalid {field}') from error
    if not number.is_finite() or not JSON_NUMBER.fullmatch(value):
        raise ValueError(f'Non-finite or invalid JSON number in {field}')
    if field == 'std' and number <= 0:
        raise ValueError('Channel std must be > 0')
    return value


def build_contract(statistics_bytes, provenance_bytes, sources):
    provenance = json.loads(provenance_bytes)
    if provenance.get('validation_data_used') is not False:
        raise ValueError('D0-A provenance must confirm validation data were not used')
    if provenance.get('statistics_convention') != 'population standard deviation, ddof=0':
        raise ValueError('D0-A provenance must confirm population standard deviation, ddof=0')
    if provenance.get('source_model_identity') != 'anatomy_constrained_joint_reconstruction_unet3d':
        raise ValueError('Expected E1-C source model identity')
    for field in ('source_checkpoint_sha256', 'train_split_sha256'):
        if not isinstance(provenance.get(field), str) or not SHA256.fullmatch(provenance[field]):
            raise ValueError(f'Missing or invalid provenance {field}')
    reader = csv.DictReader(io.StringIO(statistics_bytes.decode('utf-8'), newline=''))
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)) or not {'channel_index', 'mean', 'std'} <= set(fields):
        raise ValueError('Missing or duplicate channel statistics columns')
    channels = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Malformed channel statistics row')
        try:
            index = int(row['channel_index'])
        except ValueError as error:
            raise ValueError('Invalid channel index') from error
        if index in channels:
            raise ValueError(f'Duplicate channel index: {index}')
        if index not in range(256):
            raise ValueError(f'Unexpected channel index: {index}')
        channels[index] = (number_token(row['mean'], 'mean'), number_token(row['std'], 'std'))
    if set(channels) != set(range(256)):
        raise ValueError('Expected exactly 256 unique channels 0..255')
    contract = dict(
        schema='d0_za_normalization_contract', schema_version=1,
        normalization_type='fixed_per_channel_standardization',
        formula='(Za[c] - mean[c]) / population_std[c]', channel_count=256,
        sources=sources, source_checkpoint_sha256=provenance['source_checkpoint_sha256'],
        train_split_sha256=provenance['train_split_sha256'],
        d0a_audit_commit=provenance.get('git_commit_sha'),
        statistics_convention='population standard deviation, ddof=0',
        statistics_origin='Train-derived only; copied from completed D0-A channel statistics without recomputation.',
        reuse_policy='Validation and inference must reuse these fixed values without recomputation.',
        clipping=False, per_case_normalization=False, per_patch_normalization=False,
        numeric_encoding='JSON numeric tokens copied verbatim from CSV mean/std; no precision conversion or rounding.')
    # Standard json.dumps has no Decimal support. Emit validated JSON number
    # tokens directly to preserve every source decimal digit, including exponents.
    header = json.dumps(contract, indent=2, allow_nan=False)
    rows = [f'    {{"channel_index": {index}, "mean": {channels[index][0]}, '
            f'"population_std": {channels[index][1]}}}' for index in range(256)]
    return header[:-2] + ',\n  "channels": [\n' + ',\n'.join(rows) + '\n  ]\n}\n'


def create_contract(channel_statistics=DEFAULT_STATISTICS, analysis_provenance=DEFAULT_PROVENANCE,
                    output=DEFAULT_OUTPUT, expected_statistics_sha256=None,
                    expected_provenance_sha256=None, overwrite=False):
    requested = Path(output).expanduser()
    if requested.is_symlink():
        raise ValueError('Output must not be a symlink')
    output = requested.resolve()
    frozen = (Path(__file__).resolve().parents[1] / 'outputs' / 'frozen').resolve()
    if output == frozen or frozen in output.parents:
        raise ValueError('Do not write directly into outputs/frozen')
    if output in {Path(p).expanduser().resolve() for p in (channel_statistics, analysis_provenance)}:
        raise ValueError('Output must not replace a source artifact')
    if output.exists() and not overwrite:
        raise FileExistsError(f'Contract already exists: {output}; use --overwrite explicitly')
    statistics, stat_source = read_source(channel_statistics, expected_statistics_sha256)
    provenance, prov_source = read_source(analysis_provenance, expected_provenance_sha256)
    text = build_contract(statistics, provenance, dict(latent_channel_statistics=stat_source,
                                                     analysis_provenance=prov_source))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                         prefix='.normalization-contract-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            # Atomic no-clobber publication, including a competing writer.
            os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel-statistics', default=DEFAULT_STATISTICS)
    parser.add_argument('--analysis-provenance', default=DEFAULT_PROVENANCE)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--expected-statistics-sha256')
    parser.add_argument('--expected-provenance-sha256')
    parser.add_argument('--overwrite', action='store_true')
    print(create_contract(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
