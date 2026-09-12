"""Train the Milestone 1 baseline from explicit YAML configuration."""
import argparse
import csv
import random
import time
import numpy as np
from pathlib import Path
from typing import Any
import torch
import yaml
from torch.utils.data import DataLoader
from datasets.brats_dataset import BraTSDataset
from models.baseline_unet3d import BaselineUNet3D
from losses.segmentation_loss import SegmentationLoss
from training.trainer import train_epoch
from training.checkpointing import save_checkpoint, load_checkpoint
from evaluation.tiled import validate
from evaluation.metrics import VALIDATION_METRIC_FIELDS, best_validation_metric


def read_config(path: str) -> dict[str, Any]:
    """Read a YAML mapping."""
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return config


def read_case_ids(path):
    ids = Path(path).expanduser().read_text().splitlines()
    ids = [case.strip() for case in ids if case.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Split must be nonempty with unique IDs: {path}")
    return ids


def make_dataset(config: dict[str, Any], split: str) -> BraTSDataset:
    dataset = BraTSDataset(root=config["root"], training=split == "train",
                           normalize=config.get("normalize", True),
                           patch_size=config.get("patch_size", (96, 96, 96)),
                           foreground_sampling_probability=config.get("foreground_sampling_probability", 0.7),
                           augmentation=config.get("augmentation"))
    ids = read_case_ids(config[f"{split}_cases"])
    records = {r["subject_id"]: r for r in dataset.records}
    missing = set(ids) - records.keys()
    if missing:
        raise ValueError(f"Unknown split case IDs: {sorted(missing)}")
    dataset.records = [records[case] for case in ids]
    return dataset


def build_model(config: dict[str, Any]) -> BaselineUNet3D:
    """Separate the configuration model name from constructor parameters."""
    parameters = dict(config)
    if parameters.pop("name", "baseline_unet3d") != "baseline_unet3d":
        raise ValueError("Unsupported model name.")
    return BaselineUNet3D(**parameters)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def limit_cases(dataset, maximum):
    """Keep the first N records in split-file order, without reshuffling."""
    if maximum is not None:
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise ValueError("Case limits must be positive integers.")
        dataset.records = dataset.records[:maximum]


def synchronize(device):
    """Include outstanding accelerator work in wall-clock measurements."""
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "mps":
        torch.mps.synchronize()


def validate_resume(state, data, model_config, run, debug):
    saved = state["config"]
    if saved["model"] != model_config or saved.get("debug", False) != debug:
        raise ValueError("Incompatible resume: model architecture/config or debug mode differs.")
    for key in ("patch_size", "augmentation", "normalize", "foreground_sampling_probability", "train_cases", "val_cases"):
        if saved["data"].get(key) != data.get(key):
            raise ValueError(f"Incompatible resume: data {key} differs.")
    for key in ("batch_size", "max_train_cases", "max_val_cases", "seed", "learning_rate", "weight_decay", "threshold"):
        if saved["train"].get(key) != run.get(key):
            raise ValueError(f"Incompatible resume: training {key} differs.")
    if "target-positive" not in saved.get("selection_metric", ""):
        raise ValueError("Incompatible resume: checkpoint uses a different validation selection metric.")


def run_training(data, model_config, run, debug=False):
    run_started = time.perf_counter()
    run, model_config = dict(run), dict(model_config)
    if debug:
        run.update(epochs=1, num_workers=0, validation_frequency=1)
        model_config["base_channels"] = 2
    state = None
    start_epoch = 1
    if run.get("resume"):
        state = torch.load(run["resume"], map_location="cpu", weights_only=True)
        validate_resume(state, data, model_config, run, debug)
        start_epoch = state["epoch"] + 1
        if state["epoch"] >= run["epochs"]:
            print(f"Checkpoint epoch {state['epoch']} already meets requested final epoch {run['epochs']}; nothing to do.")
            return
    seed = run["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_data, val_data = make_dataset(data, "train"), make_dataset(data, "val")
    if {r["subject_id"] for r in train_data.records} & {r["subject_id"] for r in val_data.records}:
        raise ValueError("Training and validation split IDs overlap.")
    limit_cases(train_data, run.get("max_train_cases"))
    limit_cases(val_data, run.get("max_val_cases"))
    if debug:
        train_data.records = train_data.records[:run["batch_size"]]
        val_data.records = val_data.records[:1]
    output = Path(run["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    history = output / "history.csv"
    if state is None and (history.exists() or (output / "latest.pt").exists() or (output / "best.pt").exists()):
        raise ValueError(f"Output directory already contains a run: {output}; choose a new output directory.")
    train_loader = DataLoader(train_data, batch_size=run["batch_size"], shuffle=True,
                             num_workers=run["num_workers"], worker_init_fn=seed_worker,
                             generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=run["num_workers"],
                           worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed + 1))
    model = build_model(model_config).to(run["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=run["learning_rate"], weight_decay=run["weight_decay"])
    if state is not None:
        load_checkpoint(run["resume"], model, optimizer)
    criterion = SegmentationLoss()
    best = state.get("best_metric") if state else None
    best_epoch = state.get("best_epoch") if state else None
    durations = []
    print(f"device={run['device']} train_cases={len(train_data)} val_cases={len(val_data)} "
          f"epochs={run['epochs']} patch_size={tuple(data.get('patch_size', (96, 96, 96)))} "
          f"batch_size={run['batch_size']} trainable_parameters="
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} output_dir={output}", flush=True)
    config = {"data": data, "model": model_config, "train": run, "debug": debug,
              "selection_metric": "mean target-positive case Dice over NETC,SNFH,ET,RC,WT,TC; exclude regions with zero positive cases"}
    fields = ["epoch", "training_loss", "validation_loss", *VALIDATION_METRIC_FIELDS, "learning_rate", "epoch_duration_seconds", "training_duration_seconds", "validation_duration_seconds"]
    if state is not None:
        if Path(run["resume"]).resolve().parent != output.resolve():
            raise ValueError("Resume output directory must be the checkpoint directory to preserve history and best.pt.")
        if not history.exists():
            raise ValueError("Cannot resume: history.csv is missing.")
        with history.open(newline="") as previous:
            reader = csv.DictReader(previous)
            rows = list(reader)
            if reader.fieldnames != fields or [int(row["epoch"]) for row in rows] != list(range(1, start_epoch)):
                raise ValueError("Resume history schema/epochs do not match checkpoint; use the latest consistent checkpoint/history pair.")
        if best_epoch is None and best is not None and (output / "best.pt").exists():
            best_epoch = torch.load(output / "best.pt", map_location="cpu", weights_only=True)["epoch"]
    with history.open("a" if state is not None else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if state is None:
            writer.writeheader()
        for epoch in range(start_epoch, run["epochs"] + 1):
            synchronize(run["device"])
            epoch_started = time.perf_counter()
            training_started = epoch_started
            loss = train_epoch(model, train_loader, optimizer, criterion, run["device"])
            synchronize(run["device"])
            training_duration = time.perf_counter() - training_started
            validation_duration = 0.0
            row = dict(epoch=epoch, training_loss=loss, learning_rate=optimizer.param_groups[0]["lr"])
            if epoch % run["validation_frequency"] == 0 or epoch == run["epochs"]:
                validation_started = time.perf_counter()
                val_loss, metrics = validate(model, val_loader, criterion, run["device"],
                                             tuple(data.get("patch_size", (96, 96, 96))), run.get("threshold", 0.5))
                synchronize(run["device"])
                validation_duration = time.perf_counter() - validation_started
                row.update(validation_loss=val_loss, **metrics)
                score = best_validation_metric(metrics)
                if score is not None and (best is None or score > best):
                    best, best_epoch = score, epoch
                    save_checkpoint(output / "best.pt", model, optimizer, epoch, config, best, seed, best_epoch)
            save_checkpoint(output / "latest.pt", model, optimizer, epoch, config, best, seed, best_epoch)
            duration = time.perf_counter() - epoch_started
            durations.append(duration)
            row.update(epoch_duration_seconds=duration, training_duration_seconds=training_duration,
                       validation_duration_seconds=validation_duration)
            writer.writerow(row)
            stream.flush()
            print(row, flush=True)

    print(f"Total run duration: {time.perf_counter() - run_started:.2f}s; "
          f"average epoch duration: {sum(durations) / len(durations):.2f}s; "
          f"best validation metric: {best}; best epoch: {best_epoch}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--train-cases")
    parser.add_argument("--val-cases")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-cases", type=int)
    parser.add_argument("--max-val-cases", type=int)
    parser.add_argument("--resume", help="Checkpoint to continue; epochs is the final desired epoch.")
    args = parser.parse_args()
    data = read_config("configs/data.yaml")
    model = read_config("configs/model.yaml")
    run = read_config(args.config)
    if args.resume:
        checkpoint = torch.load(Path(args.resume).expanduser(), map_location="cpu", weights_only=True)
        data, model = dict(checkpoint["config"]["data"]), dict(checkpoint["config"]["model"])
        desired_epochs = run["epochs"]
        run = dict(checkpoint["config"]["train"])
        run.update(epochs=desired_epochs, resume=str(Path(args.resume).expanduser()),
                   output_dir=str(Path(args.resume).expanduser().parent))
        if args.debug != checkpoint["config"].get("debug", False):
            parser.error("Resume must use the same debug mode as the checkpoint")
    for key in ("epochs", "max_train_cases", "max_val_cases"):
        value = getattr(args, key)
        if value is not None:
            if value < 1:
                parser.error(f"{key.replace('_', '-')} must be positive")
            run[key] = value
    data["train_cases"] = args.train_cases or data.get("train_cases", "splits/train_cases.txt")
    data["val_cases"] = args.val_cases or data.get("val_cases", "splits/val_cases.txt")
    if args.data_root:
        data["root"] = args.data_root
    if args.device:
        run["device"] = args.device
    if args.output_dir:
        run["output_dir"] = args.output_dir
    elif args.debug:
        run["output_dir"] = "outputs/baseline_unet3d_debug"
    if any(run[key] < 1 for key in ("epochs", "batch_size", "validation_frequency")) or run["num_workers"] < 0:
        parser.error("epochs, batch_size, validation_frequency must be positive; num_workers nonnegative")
    run_training(data, model, run, args.debug)


if __name__ == "__main__":
    main()
