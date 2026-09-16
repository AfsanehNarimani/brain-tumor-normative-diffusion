"""Deterministic post-hoc reports for frozen Experiment 0.1 evaluation."""
import argparse
from analysis.brats2024_evaluation_analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', default='outputs/baseline_unet3d_full/evaluation/brats2024')
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()
    try:
        print(analyze(args.evaluation_dir, args.top_k))
    except (ValueError, OSError) as error:
        parser.exit(1, f'Evaluation analysis failed: {error}\n')


if __name__ == '__main__':
    main()
