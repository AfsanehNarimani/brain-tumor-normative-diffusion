"""Evaluate E1-C segmentation on the fixed internal 270-case validation split."""
import argparse
import math
from pathlib import Path
import platform
import shlex
import subprocess

import torch
from torch import nn

from evaluation.artifacts import evaluate_artifacts
from models.anatomy_constrained_joint_reconstruction_unet3d import AnatomyConstrainedJointReconstructionUNet3D
from models.latent_probe import file_sha256
from scripts.train_baseline import make_dataset, read_case_ids, read_config
from training.artifacts import write_json


DEFAULT_CHECKPOINT = "outputs/e1c_anatomy_constraint/best.pt"
EXPECTED_CHECKPOINT_SHA256 = "cf5a04a2d845cecb45a0ea87d617abad0afab9ad4264abce8342a6045563ee53"
DEFAULT_DATA_CONFIG = "configs/data.yaml"
EXPECTED_CASE_COUNT = 270
OUTPUT_FILES = ("per_case_metrics.csv", "aggregate_metrics.json", "evaluation_provenance.json")


class SegmentationView(nn.Module):
    """Expose only E1-C raw segmentation logits to the shared evaluator."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.segmentation_only(image)


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
             output_dir=None, expected_checkpoint_sha256=None, save_predictions=()):
    # The optional CLI assertion cannot override the frozen scientific identity.
    if (expected_checkpoint_sha256 is not None and
            expected_checkpoint_sha256.lower() != EXPECTED_CHECKPOINT_SHA256):
        raise ValueError("E1-C expected checkpoint SHA256 must match the frozen checkpoint")
    expected_checkpoint_sha256 = EXPECTED_CHECKPOINT_SHA256
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"E1-C checkpoint missing: {checkpoint_path}")
    checkpoint_sha = file_sha256(checkpoint_path)
    if (expected_checkpoint_sha256 is not None and
            checkpoint_sha.lower() != expected_checkpoint_sha256.lower()):
        raise ValueError("E1-C checkpoint SHA256 does not match expected value")

    output = Path(output_dir).expanduser().resolve() if output_dir else checkpoint_path.parent / "evaluation"
    if output == checkpoint_path.parent or Path("outputs/frozen").resolve() in (output, *output.parents):
        raise ValueError("Evaluation output must be separate from training and frozen artifact directories")
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("model", "config", "epoch", "seed"):
        if key not in state:
            raise ValueError(f"E1-C checkpoint is missing required field: {key}")
    config = state["config"]
    model_config = config.get("model", {})
    if model_config.get("name") != "anatomy_constrained_joint_reconstruction_unet3d":
        raise ValueError("Checkpoint model identity is not E1-C anatomy_constrained_joint_reconstruction_unet3d")
    constructor_config = {key: value for key, value in model_config.items() if key != "name"}
    try:
        model = AnatomyConstrainedJointReconstructionUNet3D(**constructor_config)
        model.load_state_dict(state["model"], strict=True)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"E1-C checkpoint architecture/state is incompatible: {error}") from error

    data_path = Path(data_config).expanduser().resolve()
    data = read_config(data_path)
    split_path = Path(data["val_cases"]).expanduser().resolve()
    split_ids = read_case_ids(split_path)
    dataset = make_dataset(data, "val")
    _validate_case_ids(split_ids, dataset)

    selected_device = device or config.get("train", {}).get("device", "cpu")
    threshold = config.get("train", {}).get("threshold", 0.5)
    if threshold != 0.5:
        raise ValueError(f"E1-C internal evaluation requires threshold 0.5; checkpoint has {threshold}")
    tile_size = tuple(data["patch_size"])
    model.to(selected_device)
    output.mkdir(parents=True, exist_ok=False)
    result = evaluate_artifacts(SegmentationView(model), dataset, output, selected_device,
                                threshold, tile_size, save_predictions)
    _require_finite(result)

    train_config = config.get("train", {})
    reconstruction_config = train_config.get("reconstruction", {})
    command = ["python", "-m", "scripts.evaluate_e1c", "--checkpoint", str(checkpoint_path),
               "--data-config", str(data_path), "--device", selected_device,
               "--output-dir", str(output)]
    if expected_checkpoint_sha256:
        command.extend(("--expected-checkpoint-sha256", expected_checkpoint_sha256))
    if save_predictions:
        command.extend(("--save-predictions", *save_predictions))
    provenance = {
        "experiment": "E1-C anatomy/background reconstruction constraint",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": state["epoch"],
        "checkpoint_best_epoch": state.get("best_epoch"),
        "checkpoint_best_metric": state.get("best_metric"),
        "seed": state["seed"],
        "lambda_rec": reconstruction_config.get("lambda_rec"),
        "threshold": threshold,
        "validation_case_count": len(dataset),
        "validation_split_path": str(split_path),
        "validation_split_sha256": file_sha256(split_path),
        "data_config_path": str(data_path),
        "data_config_sha256": file_sha256(data_path),
        "model_identity": model_config.get("name"),
        "model_config": model_config,
        "evaluation_settings": {"tile_size": list(tile_size),
                                "segmentation_output": "segmentation_only raw logits",
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
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--save-predictions", nargs="+", default=[], metavar="CASE_ID")
    args = parser.parse_args()
    result = evaluate(**vars(args))
    print(result)


if __name__ == "__main__":
    main()
