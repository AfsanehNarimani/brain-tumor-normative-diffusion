"""Compare frozen E1-C probe case Dice: disease (Zd) minus anatomy/background (Za)."""

import argparse
import hashlib
import json
from pathlib import Path

from analysis import e1a_probe_paired_statistics as core


ANATOMY_CSV = 'outputs/frozen/e1c_probes/anatomy/evaluation/per_case_metrics.csv'
DISEASE_CSV = 'outputs/frozen/e1c_probes/disease/evaluation/per_case_metrics.csv'
OUTPUT_DIR = 'outputs/e1c_probe_paired_statistics'
ANATOMY_SHA256 = 'd1328d3b800490cf56ee44532d185c9fd2e76cb9ca470fa713df809607212880'
DISEASE_SHA256 = 'f68053782d926051c06ea2b6619652c3ee95f9f366f0065b163764d32515185b'


def analyze(anatomy_csv=ANATOMY_CSV, disease_csv=DISEASE_CSV, output_dir=OUTPUT_DIR):
    """Pin inputs and protocol, delegating all calculations to the unchanged core."""
    inputs = [Path(path).expanduser().resolve() for path in (anatomy_csv, disease_csv)]
    requested_output = Path(output_dir).expanduser()
    if requested_output.is_symlink():
        raise ValueError('Analysis output must not be a symlink')
    output = requested_output.resolve()
    frozen = Path(__file__).resolve().parents[1] / 'outputs' / 'frozen'
    if output == frozen or frozen in output.parents or any(
            output == path.parent or output in path.parents or path.parent in output.parents
            for path in inputs):
        raise ValueError('Analysis output must be separate from frozen inputs')
    if output.exists():
        raise FileExistsError(f'Analysis output directory already exists: {output}')
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs]
    for label, actual, expected in zip(('Za', 'Zd'), hashes, (ANATOMY_SHA256, DISEASE_SHA256)):
        if actual != expected:
            raise ValueError(f'{label} input SHA256 does not match the frozen input')
    core.analyze(*inputs, output, expected_cases=270,
                 expected_positive_counts=core.EXPECTED_POSITIVE_COUNTS,
                 seed=20260917, resamples=10000, tolerance=1e-8)
    provenance_path = output / 'analysis_provenance.json'
    provenance = json.loads(provenance_path.read_text())
    if hashes != [provenance['anatomy_input_sha256'], provenance['disease_input_sha256']]:
        raise ValueError('Input SHA256 changed during analysis; discard this output')
    provenance.update(analysis_name='E1-C latent-probe paired comparison',
                      delta_direction='Zd - Za', confidence_level=0.95)
    core._write_json(provenance_path, provenance)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anatomy-csv', default=ANATOMY_CSV)
    parser.add_argument('--disease-csv', default=DISEASE_CSV)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        print(analyze(args.anatomy_csv, args.disease_csv, args.output_dir))
    except (ValueError, OSError) as error:
        parser.exit(1, f'Paired probe analysis failed: {error}\n')


if __name__ == '__main__':
    main()
