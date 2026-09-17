"""Compare frozen E0 and E1-A BraTS-compatible evaluations post hoc."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.paired_evaluation_comparison import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e0-csv', default='outputs/baseline_unet3d_full/evaluation/brats2024/official_metrics.csv')
    parser.add_argument('--e1a-csv', default='outputs/e1a_latent_split/evaluation/brats2024/official_metrics.csv')
    parser.add_argument('--e0-overlap', default='outputs/baseline_unet3d_full/evaluation/brats2024/overlap_summary.json')
    parser.add_argument('--e1a-overlap', default='outputs/e1a_latent_split/evaluation/brats2024/overlap_summary.json')
    parser.add_argument('--output-dir', default='outputs/comparisons/e0_vs_e1a')
    args = parser.parse_args()
    try:
        print(compare(args.e0_csv, args.e1a_csv, args.e0_overlap, args.e1a_overlap, args.output_dir))
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Paired comparison failed: {exc}\n')


if __name__ == '__main__':
    main()
