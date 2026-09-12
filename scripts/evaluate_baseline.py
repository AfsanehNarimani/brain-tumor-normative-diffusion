"""Evaluate a baseline checkpoint using its saved model configuration."""
import argparse
import torch
from torch.utils.data import DataLoader
from evaluation.tiled import validate
from losses.segmentation_loss import SegmentationLoss
from scripts.train_baseline import build_model, make_dataset, read_config


def main() -> None:
    """Print subject-averaged tissue and composite Dice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    data = read_config(args.data_config)
    for key in ("release", "normalize"):
        if data.get(key) != state["config"]["data"].get(key):
            raise ValueError(f"Evaluation configuration differs from checkpoint: {key}")
    dataset = make_dataset(data, "val")
    model = build_model(state["config"]["model"]).to(args.device)
    model.load_state_dict(state["model"])
    print(validate(model, DataLoader(dataset, batch_size=1), SegmentationLoss(), args.device,
                   tuple(data.get("patch_size", (96, 96, 96))), args.threshold))


if __name__ == "__main__":
    main()
