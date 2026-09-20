"""Analyze paired, presence-aware Dice from frozen E1-B latent probe evaluations."""

import argparse

from analysis.e1a_probe_paired_statistics import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anatomy-csv',
                        default='outputs/frozen/e1b_probes/anatomy/evaluation/per_case_metrics.csv')
    parser.add_argument('--disease-csv',
                        default='outputs/frozen/e1b_probes/disease/evaluation/per_case_metrics.csv')
    parser.add_argument('--output-dir', default='outputs/e1b_probe_paired_statistics')
    args = parser.parse_args()
    try:
        print(analyze(args.anatomy_csv, args.disease_csv, args.output_dir))
    except (ValueError, OSError) as error:
        parser.exit(1, f'Paired probe analysis failed: {error}\n')


if __name__ == '__main__':
    main()
