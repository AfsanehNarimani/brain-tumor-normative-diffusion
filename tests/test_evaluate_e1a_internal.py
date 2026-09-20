"""Focused E1-A internal evaluator tests without BraTS data or a GPU."""
import csv
import json
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from evaluation.metrics import EVALUATION_REGIONS
from models.latent_probe import file_sha256
from scripts import evaluate_e1a_internal as script


class FakeE1A(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.loaded_strict = None
        self.forward_calls = 0

    def load_state_dict(self, state, strict=True):
        self.loaded_strict = strict
        if state == "bad":
            raise RuntimeError("incompatible")

    def forward(self, image):
        self.forward_calls += 1
        return image + 2


@pytest.fixture
def setup_evaluation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "source" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"e1a checkpoint")
    split = tmp_path / "val_cases.txt"
    split.write_text("".join(f"case-{index:03d}\n" for index in range(270)))
    data_path = tmp_path / "data.yaml"
    data_path.write_text("data config")
    data = {"val_cases": str(split), "patch_size": [8, 9, 10]}
    config = {
        "model": {"name": "latent_split_unet3d", "in_channels": 4,
                  "out_channels": 4, "base_channels": 32,
                  "latent_projection": "conv1x1x1", "latent_channels": 256},
        "train": {"device": "cuda", "threshold": 0.5},
    }
    state = {"model": {"weight": torch.tensor(1)}, "config": config, "epoch": 15,
             "best_epoch": 15, "best_metric": 0.62, "seed": 42}
    monkeypatch.setattr(script.torch, "load", MagicMock(return_value=state))
    monkeypatch.setattr(script, "read_config", lambda _path: data)
    dataset = MagicMock()
    dataset.__len__.return_value = 270
    dataset.records = [{"subject_id": f"case-{index:03d}"} for index in range(270)]
    make_dataset = MagicMock(return_value=dataset)
    monkeypatch.setattr(script, "make_dataset", make_dataset)
    models = []
    def construct(**kwargs):
        model = FakeE1A(**kwargs)
        models.append(model)
        return model
    monkeypatch.setattr(script, "LatentSplitUNet3D", construct)

    def shared_evaluator(model, passed_dataset, output, device, threshold, tile_size, predictions):
        assert passed_dataset is dataset
        assert torch.equal(model(torch.zeros(1)), torch.tensor([2.]))
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
    artifacts = MagicMock(side_effect=shared_evaluator)
    monkeypatch.setattr(script, "evaluate_artifacts", artifacts)
    monkeypatch.setattr(script, "_git_provenance", lambda: {
        "git_commit_sha": "abc123", "git_dirty": True,
        "git_status_short": [" M configs/data.yaml"]})
    return locals()


def test_strict_loading_shared_evaluator_and_provenance(setup_evaluation):
    values = setup_evaluation
    output = values["tmp_path"] / "evaluation"
    result = script.evaluate(values["checkpoint"], values["data_path"], device="cpu",
                             output_dir=output,
                             expected_checkpoint_sha256=file_sha256(values["checkpoint"]),
                             save_predictions=["case-001"])
    model = values["models"][0]
    assert model.loaded_strict is True and model.forward_calls == 1
    values["artifacts"].assert_called_once_with(
        model, values["dataset"], output.resolve(), "cpu", 0.5, (8, 9, 10), ["case-001"])
    assert result["case_count"] == 270
    provenance = json.loads((output / "evaluation_provenance.json").read_text())
    required = {"experiment", "checkpoint_path", "checkpoint_sha256", "checkpoint_epoch",
                "checkpoint_best_epoch", "checkpoint_best_metric", "seed", "threshold",
                "validation_case_count", "validation_split_path", "validation_split_sha256",
                "data_config_path", "data_config_sha256", "model_identity", "model_config",
                "evaluation_settings", "evaluation_command", "git_commit_sha", "git_dirty",
                "git_status_short", "python_version", "pytorch_version", "device_requested",
                "cuda_available", "cuda_version", "cuda_device_name",
                "per_case_metrics_sha256", "aggregate_metrics_sha256"}
    assert required <= provenance.keys()
    assert provenance["model_identity"] == "latent_split_unet3d"
    assert provenance["evaluation_settings"] == {
        "tile_size": [8, 9, 10],
        "segmentation_output": "LatentSplitUNet3D forward raw logits",
        "protocol": "internal presence-aware Dice",
    }
    assert provenance["per_case_metrics_sha256"] == file_sha256(output / "per_case_metrics.csv")
    assert provenance["aggregate_metrics_sha256"] == file_sha256(output / "aggregate_metrics.json")


def test_checkpoint_sha_model_identity_and_strict_state_guards(setup_evaluation):
    values = setup_evaluation
    with pytest.raises(ValueError, match="SHA256"):
        script.evaluate(values["checkpoint"], values["data_path"],
                        output_dir=values["tmp_path"] / "sha",
                        expected_checkpoint_sha256="0" * 64)
    values["config"]["model"]["name"] = "baseline_unet3d"
    with pytest.raises(ValueError, match="model identity"):
        script.evaluate(values["checkpoint"], values["data_path"],
                        output_dir=values["tmp_path"] / "identity")
    values["config"]["model"]["name"] = "latent_split_unet3d"
    values["state"]["model"] = "bad"
    with pytest.raises(ValueError, match="incompatible"):
        script.evaluate(values["checkpoint"], values["data_path"],
                        output_dir=values["tmp_path"] / "strict")


def test_validation_count_duplicate_and_identity_guards(setup_evaluation):
    values = setup_evaluation
    values["dataset"].__len__.return_value = 269
    with pytest.raises(ValueError, match="270 validation cases"):
        script.evaluate(values["checkpoint"], values["data_path"],
                        output_dir=values["tmp_path"] / "count")
    values["dataset"].__len__.return_value = 270
    values["split"].write_text("".join(f"case-{index:03d}\n" for index in range(269)) + "case-000\n")
    with pytest.raises(ValueError, match="unique IDs"):
        script.evaluate(values["checkpoint"], values["data_path"],
                        output_dir=values["tmp_path"] / "duplicate")
    values["split"].write_text("".join(f"case-{index:03d}\n" for index in range(270)))
    values["dataset"].records[-1] = {"subject_id": "case-999"}
    with pytest.raises(ValueError, match=r"missing=\['case-269'\].*unexpected=\['case-999'\]"):
        script.evaluate(values["checkpoint"], values["data_path"],
                        output_dir=values["tmp_path"] / "identity")


def test_threshold_output_and_nonfinite_guards(setup_evaluation):
    values = setup_evaluation
    values["config"]["train"]["threshold"] = 0.4
    with pytest.raises(ValueError, match="threshold 0.5"):
        script.evaluate(values["checkpoint"], values["data_path"],
                        output_dir=values["tmp_path"] / "threshold")
    values["config"]["train"]["threshold"] = 0.5
    with pytest.raises(ValueError, match="frozen artifact"):
        script.evaluate(values["checkpoint"], values["data_path"],
                        output_dir="outputs/frozen/e1a-evaluation")
    output = values["tmp_path"] / "existing"
    output.mkdir()
    (output / "aggregate_metrics.json").write_text("keep")
    with pytest.raises(FileExistsError, match="already exists"):
        script.evaluate(values["checkpoint"], values["data_path"], output_dir=output)
    assert (output / "aggregate_metrics.json").read_text() == "keep"

    values["artifacts"].side_effect = None
    values["artifacts"].return_value = {"mean_validation_metric": float("nan")}
    nonfinite = values["tmp_path"] / "nonfinite"
    nonfinite.mkdir()
    with pytest.raises(RuntimeError, match="Non-finite evaluation metric"):
        script.evaluate(values["checkpoint"], values["data_path"], output_dir=nonfinite)
