"""Train the Milestone 1 baseline from explicit YAML configuration."""
import argparse
from pathlib import Path
from typing import Any
import torch
import yaml
from torch.utils.data import DataLoader
from datasets.brats_dataset import BraTSDataset
from datasets.label_mapping import LabelMapping
from models.baseline_unet3d import BaselineUNet3D
from losses.segmentation_loss import SegmentationLoss
from training.trainer import train_epoch
from training.checkpointing import save_checkpoint
from evaluation.evaluator import evaluate


def read_config(path: str) -> dict[str, Any]:
    """Read a YAML mapping."""
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return config


def make_dataset(config: dict[str, Any], split: str) -> BraTSDataset:
    """Require release-specific configuration before accessing data."""
    required = ("release", "root", f"{split}_manifest", "known_labels", "region_labels")
    if any(not config.get(key) for key in required):
        raise ValueError(f"Complete release-specific TODOs in data config: {required}")
    return BraTSDataset(config[f"{split}_manifest"], config["root"],
                        LabelMapping(config["region_labels"], config["known_labels"]),
                        normalize=config.get("normalize", True))


def main() -> None:
    """Run training and validation, saving the latest epoch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    args = parser.parse_args()
    data, model_config, run = (read_config(path) for path in
                               (args.data_config, args.model_config, args.train_config))
    torch.manual_seed(run["seed"])
    train_data, val_data = make_dataset(data, "train"), make_dataset(data, "val")
    train_ids = {record["subject_id"] for record in train_data.records}
    if train_ids & {record["subject_id"] for record in val_data.records}:
        raise ValueError("Training and validation subject IDs must be disjoint.")
    train_loader = DataLoader(train_data, batch_size=run["batch_size"], shuffle=True,
                              num_workers=run["num_workers"])
    val_loader = DataLoader(val_data, batch_size=run["batch_size"], num_workers=run["num_workers"])
    model = BaselineUNet3D(**model_config).to(run["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=run["learning_rate"])
    criterion = SegmentationLoss(run["dice_weight"], run["bce_weight"])
    for epoch in range(1, run["epochs"] + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, run["device"])
        metrics = evaluate(model, val_loader, run["device"], run["threshold"])
        print(f"epoch={epoch} loss={loss:.5f} dice={metrics}")
        save_checkpoint(Path(run["output_dir"]) / "last.pt", model, optimizer, epoch,
                        {"data": data, "model": model_config, "train": run})


if __name__ == "__main__":
    main()
