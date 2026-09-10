"""Evaluate a baseline checkpoint using its saved model configuration."""
import argparse
import torch
from torch.utils.data import DataLoader
from models.baseline_unet3d import BaselineUNet3D
from evaluation.evaluator import evaluate
from scripts.train_baseline import make_dataset, read_config


def main() -> None:
    """Print subject-averaged WT/TC/ET Dice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    data = read_config(args.data_config)
    for key in ("release", "known_labels", "region_labels", "normalize"):
        if data.get(key) != state["config"]["data"].get(key):
            raise ValueError(f"Evaluation configuration differs from checkpoint: {key}")
    dataset = make_dataset(data, "val")
    model = BaselineUNet3D(**state["config"]["model"]).to(args.device)
    model.load_state_dict(state["model"])
    print(evaluate(model, DataLoader(dataset, batch_size=args.batch_size), args.device, args.threshold))


if __name__ == "__main__":
    main()
