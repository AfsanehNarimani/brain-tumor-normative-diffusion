"""Generate read-only post-hoc baseline reports; no GPU or checkpoints needed."""
import argparse
import json
from analysis.baseline_analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.experiment_dir)
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.exit(1, f'Analysis failed: {error}\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
