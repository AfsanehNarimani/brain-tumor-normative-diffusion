"""Focused E1-C evaluator orchestration tests without BraTS data or a GPU."""
import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from evaluation.metrics import EVALUATION_REGIONS
from models.latent_probe import file_sha256
from scripts import evaluate_e1c as script


class FakeJointModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
        self.loaded_strict = None
        self.segmentation_calls = 0
        self.reconstruction_calls = 0

    def load_state_dict(self, state, strict=True):
        self.loaded_strict = strict
        if state == "bad":
            raise RuntimeError("incompatible")

    def segmentation_only(self, image):
        self.segmentation_calls += 1
        return image + 3

    def forward(self, image):
        self.reconstruction_calls += 1
        raise AssertionError("joint reconstruction forward must not be evaluated")


@pytest.fixture
def setup_evaluation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "e1c" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(script, "file_sha256", lambda path: (
        script.EXPECTED_CHECKPOINT_SHA256 if Path(path) == checkpoint else file_sha256(path)))
    split = tmp_path / "val_cases.txt"
    split.write_text("".join(f"case-{index:03d}\n" for index in range(270)))
    data_path = tmp_path / "data.yaml"
    data_path.write_text("data config")
    data = {"val_cases": str(split), "patch_size": [8, 9, 10]}
    config = {
        "model": {"name": "anatomy_constrained_joint_reconstruction_unet3d", "in_channels": 4,
                  "out_channels": 4, "base_channels": 32,
                  "latent_projection": "conv1x1x1", "latent_channels": 256},
        "train": {"device": "cuda", "threshold": 0.5,
                  "reconstruction": {"enabled": True, "loss_type": "L1", "lambda_rec": 1.0}},
    }
    state = {"model": {"weight": torch.tensor(1)}, "config": config, "epoch": 11,
             "best_epoch": 11, "best_metric": 0.6349685445865519, "seed": 42}
    monkeypatch.setattr(script.torch, "load", MagicMock(return_value=state))
    monkeypatch.setattr(script, "read_config", lambda path: data)
    dataset = MagicMock()
    dataset.__len__.return_value = 270
    dataset.records = [{"subject_id": f"case-{index:03d}"} for index in range(270)]
    monkeypatch.setattr(script, "make_dataset", MagicMock(return_value=dataset))
    models = []
    def construct(**kwargs):
        model = FakeJointModel(**kwargs)
        models.append(model)
        return model
    monkeypatch.setattr(script, "AnatomyConstrainedJointReconstructionUNet3D", construct)

    def evaluate_artifacts(model, passed_dataset, output, device, threshold, tile_size, predictions):
        assert passed_dataset is dataset
        logits = model(torch.zeros(1))
        assert torch.equal(logits, torch.tensor([3.]))
        fields = ["case_id", *(f"{region}_{suffix}" for region in EVALUATION_REGIONS
                               for suffix in ("target_present", "dice"))]
        with (output / "per_case_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow({"case_id": "case-000", **{
                f"{region}_{suffix}": 1 if suffix == "target_present" else 0.5
                for region in EVALUATION_REGIONS for suffix in ("target_present", "dice")}})
        result = {"case_count": 270, "threshold": threshold, "mean_validation_metric": 0.5,
                  "regions": {region: {"positive_case_count": 1, "case_dice": 0.5,
                                       "dataset_dice": 0.5} for region in EVALUATION_REGIONS}}
        (output / "aggregate_metrics.json").write_text(json.dumps(result))
        return result
    artifacts = MagicMock(side_effect=evaluate_artifacts)
    monkeypatch.setattr(script, "evaluate_artifacts", artifacts)
    monkeypatch.setattr(script, "_git_provenance", lambda: {
        "git_commit_sha": "abc123", "git_dirty": True,
        "git_status_short": [" M configs/data.yaml"]})
    return locals()


def test_strict_loading_segmentation_only_schema_and_provenance(setup_evaluation):
    values = setup_evaluation
    output = values["tmp_path"] / "evaluation"
    result = script.evaluate(values["checkpoint"], values["data_path"], device="cpu",
                             output_dir=output, expected_checkpoint_sha256=script.EXPECTED_CHECKPOINT_SHA256)
    model = values["models"][0]
    assert model.loaded_strict is True
    assert model.segmentation_calls == 1 and model.reconstruction_calls == 0
    assert result["case_count"] == 270
    with (output / "per_case_metrics.csv").open(newline="") as stream:
        fields = next(csv.DictReader(stream)).keys()
    assert "case_id" in fields
    for region in EVALUATION_REGIONS:
        assert f"{region}_target_present" in fields and f"{region}_dice" in fields
    provenance = json.loads((output / "evaluation_provenance.json").read_text())
    required = {"experiment", "checkpoint_path", "checkpoint_sha256", "checkpoint_epoch",
                "checkpoint_best_epoch", "checkpoint_best_metric", "seed", "lambda_rec",
                "threshold", "validation_case_count", "validation_split_path",
                "validation_split_sha256", "model_identity", "model_config",
                "evaluation_command", "git_commit_sha", "git_dirty", "python_version",
                "pytorch_version", "device_requested", "per_case_metrics_sha256",
                "aggregate_metrics_sha256"}
    assert required <= provenance.keys()
    assert provenance["checkpoint_epoch"] == provenance["checkpoint_best_epoch"] == 11
    assert provenance["experiment"] == "E1-C anatomy/background reconstruction constraint"
    assert provenance["model_identity"] == "anatomy_constrained_joint_reconstruction_unet3d"
    assert provenance["checkpoint_sha256"] == script.EXPECTED_CHECKPOINT_SHA256
    assert provenance["checkpoint_path"] == str(values["checkpoint"])
    assert provenance["validation_split_sha256"] == file_sha256(values["split"])
    assert provenance["validation_case_count"] == 270
    assert provenance["threshold"] == 0.5
    assert "scripts.evaluate_e1c" in provenance["evaluation_command"]
    assert provenance["git_commit_sha"] == "abc123"
    assert provenance["git_dirty"] is True
    assert provenance["git_status_short"] == [" M configs/data.yaml"]
    assert result["mean_validation_metric"] != provenance["checkpoint_best_metric"]
    script.make_dataset.assert_called_once_with(values["data"], "val")
    args = values["artifacts"].call_args.args
    assert args[3:] == ("cpu", 0.5, (8, 9, 10), ())
    script.torch.load.assert_called_once_with(values["checkpoint"], map_location="cpu", weights_only=True)
    assert provenance["lambda_rec"] == 1.0
    assert provenance["per_case_metrics_sha256"] == file_sha256(output / "per_case_metrics.csv")


def test_checkpoint_sha_and_strict_incompatibility_fail(setup_evaluation):
    values = setup_evaluation
    with pytest.raises(ValueError, match="SHA256"):
        script.evaluate(values["checkpoint"], values["data_path"],
                        output_dir=values["tmp_path"] / "sha", expected_checkpoint_sha256="0" * 64)
    values["state"]["model"] = "bad"
    with pytest.raises(ValueError, match="incompatible"):
        script.evaluate(values["checkpoint"], values["data_path"],
                        output_dir=values["tmp_path"] / "bad")


def test_validation_count_and_duplicate_guards(setup_evaluation):
    values = setup_evaluation
    values["dataset"].__len__.return_value = 269
    with pytest.raises(ValueError, match="270 validation cases"):
        script.evaluate(values["checkpoint"], values["data_path"],
                        output_dir=values["tmp_path"] / "count")
    values["dataset"].__len__.return_value = 270
    values["dataset"].records[-1] = {"subject_id": "case-000"}
    with pytest.raises(ValueError, match="duplicate case IDs"):
        script.evaluate(values["checkpoint"], values["data_path"],
                        output_dir=values["tmp_path"] / "duplicate")


def test_existing_artifacts_are_not_overwritten(setup_evaluation):
    values = setup_evaluation
    output = values["tmp_path"] / "evaluation"
    output.mkdir()
    artifact = output / "aggregate_metrics.json"
    artifact.write_text("keep")
    with pytest.raises(FileExistsError, match="already exists"):
        script.evaluate(values["checkpoint"], values["data_path"], output_dir=output)
    assert artifact.read_text() == "keep"


@pytest.mark.parametrize("field", ["model", "config", "epoch", "seed"])
def test_missing_checkpoint_fields_rejected(setup_evaluation, field):
    values = setup_evaluation
    del values["state"][field]
    with pytest.raises(ValueError, match=f"missing required field: {field}"):
        script.evaluate(values["checkpoint"], values["data_path"])


def test_wrong_model_identity_rejected(setup_evaluation):
    values = setup_evaluation
    values["config"]["model"]["name"] = "joint_reconstruction_unet3d"
    with pytest.raises(ValueError, match="model identity"):
        script.evaluate(values["checkpoint"], values["data_path"])


def test_actual_checkpoint_sha_rejected_by_default(setup_evaluation, monkeypatch):
    values = setup_evaluation
    monkeypatch.setattr(script, "file_sha256", file_sha256)
    with pytest.raises(ValueError, match="SHA256"):
        script.evaluate(values["checkpoint"], values["data_path"])
    script.torch.load.assert_not_called()


def test_threshold_rejected(setup_evaluation):
    values = setup_evaluation
    values["config"]["train"]["threshold"] = 0.4
    with pytest.raises(ValueError, match="requires threshold 0.5"):
        script.evaluate(values["checkpoint"], values["data_path"])
    values["artifacts"].assert_not_called()


@pytest.mark.parametrize("kind", ["empty", "training", "frozen"])
def test_output_directory_protection(setup_evaluation, kind):
    values = setup_evaluation
    output = {"empty": values["tmp_path"] / "empty",
              "training": values["checkpoint"].parent,
              "frozen": Path("outputs/frozen/e1c-evaluation")}[kind]
    if kind == "empty":
        output.mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        script.evaluate(values["checkpoint"], values["data_path"], output_dir=output)
    values["artifacts"].assert_not_called()


@pytest.mark.parametrize("kind", ["count", "duplicate", "identity"])
def test_validation_split_guards(setup_evaluation, kind):
    values = setup_evaluation
    ids = values["split"].read_text().splitlines()
    if kind == "count":
        ids.pop()
    elif kind == "duplicate":
        ids[-1] = ids[0]
    else:
        ids[-1] = "unexpected-case"
    values["split"].write_text("\n".join(ids))
    with pytest.raises(ValueError, match="validation IDs|unique IDs|duplicate case IDs|case IDs mismatch"):
        script.evaluate(values["checkpoint"], values["data_path"])
    values["artifacts"].assert_not_called()


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape"])
def test_real_model_strict_state_loading(setup_evaluation, monkeypatch, mutation):
    from models.anatomy_constrained_joint_reconstruction_unet3d import (
        AnatomyConstrainedJointReconstructionUNet3D,
    )
    values = setup_evaluation
    model_config = values["config"]["model"]
    model_config.update(base_channels=2, latent_channels=16)
    model = AnatomyConstrainedJointReconstructionUNet3D(**{
        key: value for key, value in model_config.items() if key != "name"})
    state = model.state_dict()
    key = next(iter(state))
    if mutation == "missing":
        del state[key]
    elif mutation == "unexpected":
        state["unexpected.weight"] = torch.zeros(1)
    else:
        state[key] = torch.zeros(1)
    values["state"]["model"] = state
    monkeypatch.setattr(script, "AnatomyConstrainedJointReconstructionUNet3D",
                        AnatomyConstrainedJointReconstructionUNet3D)
    with pytest.raises(ValueError, match="architecture/state is incompatible"):
        script.evaluate(values["checkpoint"], values["data_path"])
    values["artifacts"].assert_not_called()
