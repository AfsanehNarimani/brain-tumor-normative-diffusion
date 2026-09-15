"""Evaluate a best checkpoint on the complete saved validation split."""
import argparse
from pathlib import Path
import torch
from evaluation.artifacts import evaluate_artifacts
from scripts.train_baseline import build_model, make_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/baseline_unet3d')
    parser.add_argument('--checkpoint', help='Defaults to <output-dir>/best.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save-predictions', nargs='+', default=[], metavar='CASE_ID')
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.output_dir) / 'best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = state['config']
    dataset = make_dataset(config['data'], 'val')  # No short-run case limits.
    model = build_model(config['model']).to(args.device)
    model.load_state_dict(state['model'])
    result = evaluate_artifacts(model, dataset, checkpoint.parent / 'evaluation', args.device,
                               config['train'].get('threshold', 0.5),
                               tuple(config['data'].get('patch_size', (96, 96, 96))), args.save_predictions)
    print(result)


if __name__ == '__main__':
    main()
