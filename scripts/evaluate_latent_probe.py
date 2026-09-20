"""Evaluate one frozen E1-A or E1-B probe on the fixed validation split."""

import argparse
from pathlib import Path
import platform
import subprocess

import nibabel as nib
import numpy as np
import torch

from evaluation.artifacts import evaluate_artifacts
from models.latent_probe import FrozenLatentProbe, PROBE_ARCHITECTURE, file_sha256, load_source
from scripts.train_baseline import make_dataset, read_case_ids
from scripts.train_latent_probe import load_probe_checkpoint, read_config
from training.artifacts import write_json


CONFIG_PROTOCOLS = {
    "configs/probe_anatomy.yaml": ("e1a", "anatomy", "latent_split_unet3d"),
    "configs/probe_disease.yaml": ("e1a", "disease", "latent_split_unet3d"),
    "configs/e1b_probe_anatomy.yaml": ("e1b", "anatomy", "joint_reconstruction_unet3d"),
    "configs/e1b_probe_disease.yaml": ("e1b", "disease", "joint_reconstruction_unet3d"),
}
CONFIG_CHOICES = tuple(CONFIG_PROTOCOLS)
EVALUATION_FILES = ("per_case_metrics.csv", "aggregate_metrics.json", "evaluation_provenance.json")


def resolve_paths(config, checkpoint=None, output_dir=None):
    checkpoint_path = Path(checkpoint or Path(config["output_dir"]) / "best.pt").expanduser().resolve()
    output_path = Path(output_dir or checkpoint_path.parent / "evaluation").expanduser().resolve()
    return checkpoint_path, output_path


def check_output_path(output, config, source_path):
    training_dir = Path(config["output_dir"]).expanduser().resolve()
    source_dir = source_path.parent
    baseline_dir = Path("outputs/baseline_unet3d_full").resolve()
    if output == training_dir or any(output == path or path in output.parents
                                     for path in (source_dir, baseline_dir)):
        raise ValueError("Evaluation output must be separate from training, frozen source, and baseline artifacts")
    for name in EVALUATION_FILES:
        if (output / name).exists():
            raise FileExistsError(f"Evaluation artifact already exists: {output / name}")


def git_provenance():
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    return {
        "git_commit_sha": git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(git("status", "--porcelain")),
        "git_status_short": [line for line in git("status", "--short").splitlines() if line],
    }


def resolve_protocol(config_path):
    """Resolve only an explicitly supported config, without filename inference."""
    resolved = Path(config_path).resolve()
    for supported, protocol in CONFIG_PROTOCOLS.items():
        if resolved == Path(supported).resolve():
            return protocol
    raise ValueError(f"Unsupported probe config: {config_path}")


def evaluate(config_path, *, checkpoint=None, device=None, output_dir=None, save_predictions=()):
    config_path = Path(config_path)
    expected_experiment, expected_latent, expected_model = resolve_protocol(config_path)
    config = read_config(config_path)
    if config.get("latent_source") != expected_latent:
        raise ValueError("Probe config latent_source mismatch")
    resolved_experiment = config.get("source_experiment", "e1a")
    if resolved_experiment != expected_experiment:
        raise ValueError("Probe config source_experiment mismatch")

    checkpoint_path, output = resolve_paths(config, checkpoint, output_dir)
    source_path = Path(config["source_checkpoint"]).expanduser().resolve()
    check_output_path(output, config, source_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Probe checkpoint missing: {checkpoint_path}")

    data_path = Path(config["data_config"]).expanduser().resolve()
    data_config = read_config(data_path)
    split_path = Path(data_config["val_cases"]).expanduser().resolve()
    validation_ids = read_case_ids(split_path)
    if len(validation_ids) != 270:
        raise ValueError(f"Expected exactly 270 validation IDs; found {len(validation_ids)}")

    source, source_info = load_source(source_path, source_experiment=resolved_experiment)
    if source_info.get("experiment") != resolved_experiment:
        raise ValueError("Loaded frozen source experiment identity mismatch")
    if source_info.get("model") != expected_model:
        raise ValueError("Loaded frozen source model identity mismatch")
    model = FrozenLatentProbe(source, config["latent_source"])
    saved_config = {key: value for key, value in config.items() if key != "epochs"}
    state = load_probe_checkpoint(checkpoint_path, model, None, saved_config, source_info)
    if state["latent_source"] != config["latent_source"]:
        raise ValueError("Probe checkpoint latent_source mismatch")
    if state["probe_architecture"] != PROBE_ARCHITECTURE:
        raise ValueError("Probe checkpoint architecture mismatch")

    dataset = make_dataset(data_config, "val")
    dataset_ids = [record["subject_id"] for record in dataset.records]
    if len(dataset) != 270:
        raise ValueError(f"Expected exactly 270 validation cases; found {len(dataset)}")
    seen = set()
    duplicates = set()
    for case_id in dataset_ids:
        if case_id in seen:
            duplicates.add(case_id)
        seen.add(case_id)
    if duplicates:
        raise ValueError(f"Duplicate validation dataset subject IDs: {sorted(duplicates)[:5]}")
    missing = sorted(set(validation_ids) - seen)
    unexpected = sorted(seen - set(validation_ids))
    if missing or unexpected:
        raise ValueError(
            "Validation dataset IDs mismatch: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"unexpected={unexpected[:5]} ({len(unexpected)} total)"
        )

    requested_device = device or config["device"]
    model.to(requested_device)
    threshold = config["threshold"]
    tile_size = tuple(data_config["patch_size"])
    output.mkdir(parents=True, exist_ok=True)
    result = evaluate_artifacts(model, dataset, output, requested_device, threshold,
                                tile_size, save_predictions)

    provenance = {
        "source_experiment": source_info["experiment"],
        "source_model_identity": source_info["model"],
        "latent_source": config["latent_source"],
        "probe_checkpoint": str(checkpoint_path),
        "probe_checkpoint_sha256": file_sha256(checkpoint_path),
        "probe_checkpoint_epoch": state["epoch"],
        "probe_checkpoint_best_epoch": state.get("best_epoch"),
        "probe_checkpoint_best_metric": state.get("best_metric"),
        "frozen_source_checkpoint": source_info["path"],
        "frozen_source_sha256": source_info["sha256"],
        "frozen_source_epoch": source_info["epoch"],
        "validation_split": str(split_path),
        "validation_split_sha256": file_sha256(split_path),
        "validation_case_count": len(dataset),
        "threshold": threshold,
        "tile_size": list(tile_size),
        "data_config": str(data_path),
        "data_config_sha256": file_sha256(data_path),
        **git_provenance(),
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "nibabel_version": nib.__version__,
        "device_requested": requested_device,
        "probe_architecture": PROBE_ARCHITECTURE,
    }
    write_json(output / "evaluation_provenance.json", provenance)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=CONFIG_CHOICES)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--save-predictions", nargs="+", default=[], metavar="CASE_ID")
    args = parser.parse_args()
    result = evaluate(args.config, checkpoint=args.checkpoint, device=args.device,
                      output_dir=args.output_dir, save_predictions=args.save_predictions)
    print(result)


if __name__ == "__main__":
    main()
