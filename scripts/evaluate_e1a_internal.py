"""Evaluate frozen E1-A segmentation with the shared internal validation protocol."""
import argparse
import math
from pathlib import Path
import platform
import shlex
import subprocess

import torch

from evaluation.artifacts import evaluate_artifacts
from models.latent_probe import file_sha256
from models.latent_split_unet3d import LatentSplitUNet3D
from scripts.train_baseline import make_dataset, read_case_ids, read_config
from training.artifacts import write_json


DEFAULT_CHECKPOINT = "outputs/frozen/e1a_latent_split/best.pt"
DEFAULT_DATA_CONFIG = "configs/data.yaml"
DEFAULT_OUTPUT = "outputs/e1a_internal_evaluation"
EXPECTED_CASE_COUNT = 270
EXPECTED_MODEL = "latent_split_unet3d"
OUTPUT_FILES = ("per_case_metrics.csv", "aggregate_metrics.json", "evaluation_provenance.json")


def _git_provenance():
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout
    return {
        "git_commit_sha": git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(git("status", "--porcelain")),
        "git_status_short": [line for line in git("status", "--short").splitlines() if line],
    }


def _validate_case_ids(split_ids, dataset):
    if len(split_ids) != EXPECTED_CASE_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_CASE_COUNT} validation IDs; found {len(split_ids)}")
    if len(set(split_ids)) != len(split_ids):
        raise ValueError("Validation split contains duplicate case IDs")
    if len(dataset) != EXPECTED_CASE_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_CASE_COUNT} validation cases; found {len(dataset)}")
    dataset_ids = [record["subject_id"] for record in dataset.records]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Validation dataset contains duplicate case IDs")
    missing = sorted(set(split_ids) - set(dataset_ids))
    unexpected = sorted(set(dataset_ids) - set(split_ids))
    if missing or unexpected:
        raise ValueError(f"Validation case IDs mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")


def _require_finite(value, path="result"):
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite(child, f"{path}.{key}")
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite evaluation metric: {path}={value}")


def evaluate(checkpoint=DEFAULT_CHECKPOINT, data_config=DEFAULT_DATA_CONFIG, device=None,
             output_dir=DEFAULT_OUTPUT, expected_checkpoint_sha256=None, save_predictions=()):
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"E1-A checkpoint missing: {checkpoint_path}")
    checkpoint_sha = file_sha256(checkpoint_path)
    if (expected_checkpoint_sha256 is not None and
            checkpoint_sha.lower() != expected_checkpoint_sha256.lower()):
        raise ValueError("E1-A checkpoint SHA256 does not match expected value")

    output = Path(output_dir or DEFAULT_OUTPUT).expanduser().resolve()
    frozen = Path("outputs/frozen").resolve()
    if output == checkpoint_path.parent or frozen in (output, *output.parents):
        raise ValueError("Evaluation output must be separate from checkpoint and frozen artifact directories")
    for name in OUTPUT_FILES:
        if (output / name).exists():
            raise FileExistsError(f"Evaluation artifact already exists: {output / name}")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("model", "config", "epoch", "seed"):
        if key not in state:
            raise ValueError(f"E1-A checkpoint is missing required field: {key}")
    config = state["config"]
    if not isinstance(config, dict):
        raise ValueError("E1-A checkpoint config must be a mapping")
    model_config = config.get("model", {})
    if model_config.get("name") != EXPECTED_MODEL:
        raise ValueError(f"Checkpoint model identity is not E1-A {EXPECTED_MODEL}")
    constructor_config = {key: value for key, value in model_config.items() if key != "name"}
    try:
        model = LatentSplitUNet3D(**constructor_config)
        model.load_state_dict(state["model"], strict=True)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"E1-A checkpoint architecture/state is incompatible: {error}") from error

    data_path = Path(data_config).expanduser().resolve()
    data = read_config(data_path)
    split_path = Path(data["val_cases"]).expanduser().resolve()
    split_ids = read_case_ids(split_path)
    dataset = make_dataset(data, "val")
    _validate_case_ids(split_ids, dataset)

    selected_device = device or config.get("train", {}).get("device", "cpu")
    threshold = config.get("train", {}).get("threshold", 0.5)
    if threshold != 0.5:
        raise ValueError(f"E1-A internal evaluation requires threshold 0.5; checkpoint has {threshold}")
    tile_size = tuple(data["patch_size"])
    model.to(selected_device)
    output.mkdir(parents=True, exist_ok=True)
    result = evaluate_artifacts(model, dataset, output, selected_device, threshold,
                                tile_size, save_predictions)
    _require_finite(result)

    command = ["python", "-m", "scripts.evaluate_e1a_internal",
               "--checkpoint", str(checkpoint_path), "--data-config", str(data_path),
               "--device", selected_device, "--output-dir", str(output)]
    if expected_checkpoint_sha256:
        command.extend(("--expected-checkpoint-sha256", expected_checkpoint_sha256))
    if save_predictions:
        command.append("--save-predictions")
        command.extend(save_predictions)
    provenance = {
        "experiment": "E1-A latent-split segmentation",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": state["epoch"],
        "checkpoint_best_epoch": state.get("best_epoch"),
        "checkpoint_best_metric": state.get("best_metric"),
        "seed": state["seed"],
        "threshold": threshold,
        "validation_case_count": len(dataset),
        "validation_split_path": str(split_path),
        "validation_split_sha256": file_sha256(split_path),
        "data_config_path": str(data_path),
        "data_config_sha256": file_sha256(data_path),
        "model_identity": model_config["name"],
        "model_config": model_config,
        "evaluation_settings": {"tile_size": list(tile_size),
                                "segmentation_output": "LatentSplitUNet3D forward raw logits",
                                "protocol": "internal presence-aware Dice"},
        "evaluation_command": shlex.join(command),
        **_git_provenance(),
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "device_requested": selected_device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_name": (torch.cuda.get_device_name(torch.cuda.current_device())
                             if torch.cuda.is_available() else None),
        "per_case_metrics_sha256": file_sha256(output / "per_case_metrics.csv"),
        "aggregate_metrics_sha256": file_sha256(output / "aggregate_metrics.json"),
    }
    write_json(output / "evaluation_provenance.json", provenance)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--save-predictions", nargs="+", default=[], metavar="CASE_ID")
    args = parser.parse_args()
    print(evaluate(**vars(args)))


if __name__ == "__main__":
    main()
