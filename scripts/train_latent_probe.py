"""Train one E1-A frozen bottleneck accessibility probe; no E1-A updates."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from evaluation.metrics import VALIDATION_METRIC_FIELDS, best_validation_metric
from evaluation.tiled import validate
from losses.segmentation_loss import SegmentationLoss
from models.latent_probe import (FrozenLatentProbe, LatentProbeDecoder3D,
                                 PROBE_ARCHITECTURE, SOURCE_SHA256, load_source, file_sha256)
from scripts.train_baseline import make_dataset, seed_worker, synchronize
from training.artifacts import environment_metadata, write_json
from training.trainer import train_epoch

SELECTION_METRIC = "mean target-positive case Dice over NETC,SNFH,ET,RC,WT,TC; exclude regions with zero positive cases"
FIELDS = ["epoch", "training_loss", "validation_loss", *VALIDATION_METRIC_FIELDS,
          "learning_rate", "epoch_duration_seconds", "training_duration_seconds", "validation_duration_seconds"]


def read_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Probe config must be a YAML mapping")
    return config


def validate_pair(anatomy="configs/probe_anatomy.yaml", disease="configs/probe_disease.yaml"):
    a, d = read_config(anatomy), read_config(disease)
    for config, source in ((a, "anatomy"), (d, "disease")):
        if config.get("latent_source") != source:
            raise ValueError("Probe config latent_source mismatch")
    ignore = {"experiment_name", "latent_source", "output_dir"}
    if {k: v for k, v in a.items() if k not in ignore} != {k: v for k, v in d.items() if k not in ignore}:
        raise ValueError("Anatomy and disease protocol/config differ")
    if Path(a["output_dir"]).resolve() == Path(d["output_dir"]).resolve():
        raise ValueError("Probe output directories must be isolated")
    return a, d


def save_probe_checkpoint(path, model, optimizer, epoch, best_metric, best_epoch, config, source_info):
    """Keep source weights external; store the strict source identity instead."""
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(dict(probe=model.probe.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                    best_metric=best_metric, best_epoch=best_epoch, config=config,
                    probe_architecture=PROBE_ARCHITECTURE, latent_source=model.latent_source,
                    source_checkpoint={k: source_info[k] for k in ("path", "sha256", "epoch")}), temporary)
    temporary.replace(path)


def load_probe_checkpoint(path, model, optimizer, config, source_info):
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected_source = state["source_checkpoint"]
    if not Path(expected_source["path"]).is_file():
        raise FileNotFoundError(f"Frozen E1-A source missing: {expected_source['path']}")
    if file_sha256(expected_source["path"]) != expected_source["sha256"]:
        raise ValueError("Frozen E1-A source SHA256 mismatch")
    if (state["config"] != config or state["latent_source"] != model.latent_source
            or state["probe_architecture"] != PROBE_ARCHITECTURE
            or state["source_checkpoint"] != {k: source_info[k] for k in ("path", "sha256", "epoch")}):
        raise ValueError("Probe resume config, architecture, latent, or frozen source identity mismatch")
    # Check again at resume/evaluation time, including if the file changed since construction.
    if not Path(source_info["path"]).is_file() or file_sha256(source_info["path"]) != source_info["sha256"]:
        raise ValueError("Frozen E1-A source missing or SHA256 mismatch")
    model.probe.load_state_dict(state["probe"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state


def run(config, *, resume=None, epochs=None, device=None, source_sha256=SOURCE_SHA256,
        source_epoch=15):
    config = dict(config)
    source_path = Path(config["source_checkpoint"]).expanduser().resolve()
    output = Path(config["output_dir"]).expanduser().resolve()
    protected = [Path("outputs/e1a_latent_split").resolve(), Path("outputs/baseline_unet3d_full").resolve()]
    if any(output == p or p in output.parents for p in protected) or output == source_path.parent:
        raise ValueError("Probe output must be separate from frozen source artifacts")
    data = read_config(config["data_config"])
    if len(Path(data["train_cases"]).read_text().splitlines()) != 1080 or len(Path(data["val_cases"]).read_text().splitlines()) != 270:
        raise ValueError("Expected fixed 1080/270 split")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source, source_info = load_source(source_path, expected_sha256=source_sha256, expected_epoch=source_epoch)
    model = FrozenLatentProbe(source, config["latent_source"])
    active_device = device or config["device"]
    model.to(active_device)
    optimizer = torch.optim.AdamW(model.probe.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    desired_epochs = epochs or config["epochs"]
    if desired_epochs < 1:
        raise ValueError("epochs must be positive")
    saved_config = {k: v for k, v in config.items() if k != "epochs"}
    state = load_probe_checkpoint(resume, model, optimizer, saved_config, source_info) if resume else None
    start = state["epoch"] + 1 if state else 1
    if state and desired_epochs <= state["epoch"]:
        return
    train_data, val_data = make_dataset(data, "train"), make_dataset(data, "val")
    if {r["subject_id"] for r in train_data.records} & {r["subject_id"] for r in val_data.records}:
        raise ValueError("Training and validation splits overlap")
    output.mkdir(parents=True, exist_ok=True)
    history = output / "history.csv"
    if not state and any((output / name).exists() for name in ("config.yaml", "history.csv", "best.pt", "latest.pt")):
        raise FileExistsError("Probe output already contains a run")
    if state and (Path(resume).resolve().parent != output or not history.exists()):
        raise ValueError("Resume requires same output directory and history.csv")
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                            num_workers=config["num_workers"], worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(seed + 1))
    count = sum(p.numel() for p in model.probe.parameters())
    environment = environment_metadata(count)
    splits = {f"{part}_split_sha256": hashlib.sha256(Path(data[part + "_cases"]).read_bytes()).hexdigest()
              for part in ("train", "val")}
    metadata = dict(**config, data=data, **splits, source_checkpoint_info=source_info,
                    probe_architecture=PROBE_ARCHITECTURE, probe_parameter_count=count,
                    frozen_source_parameter_count=source_info["parameter_count"],
                    optimizer="AdamW", loss="SegmentationLoss (Dice+BCE)",
                    selection_metric=SELECTION_METRIC, environment=environment)
    if state:
        previous = yaml.safe_load((output / "config.yaml").read_text())
        if {k: v for k, v in previous.items() if k != "environment"} != {k: v for k, v in metadata.items() if k != "environment"}:
            raise ValueError("Resume artifact metadata mismatch")
    else:
        (output / "config.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
        write_json(output / "environment.json", environment)
    best, best_epoch = (state["best_metric"], state["best_epoch"]) if state else (None, None)
    criterion = SegmentationLoss()
    started = time.perf_counter()
    with (output / "train.log").open("a" if state else "w") as log, history.open("a" if state else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not state:
            writer.writeheader()
        for epoch in range(start, desired_epochs + 1):
            synchronize(active_device)
            epoch_start = time.perf_counter()
            loss = train_epoch(model, train_loader, optimizer, criterion, active_device)
            synchronize(active_device)
            training_duration = time.perf_counter() - epoch_start
            row = dict(epoch=epoch, training_loss=loss, learning_rate=optimizer.param_groups[0]["lr"])
            validation_duration = 0.0
            if epoch % config["validation_frequency"] == 0 or epoch == desired_epochs:
                val_start = time.perf_counter()
                val_loss, metrics = validate(model, val_loader, criterion, active_device,
                                             tuple(data["patch_size"]), config["threshold"])
                synchronize(active_device)
                validation_duration = time.perf_counter() - val_start
                row.update(validation_loss=val_loss, **metrics)
                score = best_validation_metric(metrics)
                if score is not None and (best is None or score > best):
                    best, best_epoch = score, epoch
                    save_probe_checkpoint(output / "best.pt", model, optimizer, epoch, best, best_epoch, saved_config, source_info)
            row.update(epoch_duration_seconds=time.perf_counter() - epoch_start,
                       training_duration_seconds=training_duration,
                       validation_duration_seconds=validation_duration)
            save_probe_checkpoint(output / "latest.pt", model, optimizer, epoch, best, best_epoch, saved_config, source_info)
            writer.writerow(row)
            stream.flush()
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)
    write_json(output / "summary.json", dict(latent_source=model.latent_source,
               source_checkpoint=source_info, probe_architecture=PROBE_ARCHITECTURE,
               probe_parameter_count=count, frozen_source_parameter_count=source_info["parameter_count"],
               best_epoch=best_epoch, best_validation_metric=best, total_epochs_completed=desired_epochs,
               duration_seconds=time.perf_counter() - started, **splits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=["configs/probe_anatomy.yaml", "configs/probe_disease.yaml"])
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    anatomy, disease = validate_pair()
    run(anatomy if "anatomy" in args.config else disease,
        resume=args.resume, epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
