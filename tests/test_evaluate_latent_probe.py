"""Focused orchestration tests for paired frozen-probe evaluation."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.latent_probe import PROBE_ARCHITECTURE, file_sha256
from scripts import evaluate_latent_probe as script


@pytest.fixture
def setup_evaluation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "probe" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"probe checkpoint")
    source = tmp_path / "source" / "best.pt"
    source.parent.mkdir()
    source.write_bytes(b"source checkpoint")
    split = tmp_path / "val_cases.txt"
    split.write_text("".join(f"case-{index:03d}\n" for index in range(270)))
    data_path = tmp_path / "data.yaml"
    data_path.write_text("validation configuration\n")
    config = {
        "latent_source": "anatomy",
        "output_dir": str(checkpoint.parent),
        "source_checkpoint": str(source),
        "data_config": str(data_path),
        "epochs": 30,
        "device": "cpu",
        "threshold": 0.5,
    }
    data_config = {"val_cases": str(split), "patch_size": [8, 9, 10]}
    read_config = MagicMock(side_effect=lambda path: data_config if Path(path) == data_path else config)
    monkeypatch.setattr(script, "read_config", read_config)
    source_info = {"path": str(source.resolve()), "sha256": file_sha256(source), "epoch": 15}
    load_source = MagicMock(return_value=(object(), source_info))
    monkeypatch.setattr(script, "load_source", load_source)
    model = MagicMock()
    model.latent_source = "anatomy"
    construct_model = MagicMock(return_value=model)
    monkeypatch.setattr(script, "FrozenLatentProbe", construct_model)
    state = {"latent_source": "anatomy", "probe_architecture": PROBE_ARCHITECTURE,
             "epoch": 12, "best_epoch": 10, "best_metric": 0.61}
    load_checkpoint = MagicMock(return_value=state)
    monkeypatch.setattr(script, "load_probe_checkpoint", load_checkpoint)
    dataset = MagicMock()
    dataset.__len__.return_value = 270
    dataset.records = [{"subject_id": f"case-{index:03d}"} for index in range(270)]
    make_dataset = MagicMock(return_value=dataset)
    monkeypatch.setattr(script, "make_dataset", make_dataset)
    evaluate_artifacts = MagicMock(return_value={"case_count": 270})
    monkeypatch.setattr(script, "evaluate_artifacts", evaluate_artifacts)
    monkeypatch.setattr(script, "git_provenance", lambda: {
        "git_commit_sha": "abc123", "git_dirty": True, "git_status_short": [" M configs/data.yaml"]
    })
    return locals()


def test_default_paths(setup_evaluation):
    config = setup_evaluation["config"]
    checkpoint, output = script.resolve_paths(config)
    assert checkpoint == setup_evaluation["checkpoint"].resolve()
    assert output == checkpoint.parent / "evaluation"
    alternate = checkpoint.parent / "latest.pt"
    assert script.resolve_paths(config, alternate)[1] == alternate.parent / "evaluation"


def test_latent_source_mismatch_is_rejected(setup_evaluation):
    setup_evaluation["config"]["latent_source"] = "disease"
    with pytest.raises(ValueError, match="latent_source mismatch"):
        script.evaluate("configs/probe_anatomy.yaml")
    setup_evaluation["load_source"].assert_not_called()


def test_validation_split_and_dataset_must_each_have_270_cases(setup_evaluation):
    split = setup_evaluation["split"]
    split.write_text("".join(f"case-{index:03d}\n" for index in range(269)))
    with pytest.raises(ValueError, match="270 validation IDs"):
        script.evaluate("configs/probe_anatomy.yaml")
    setup_evaluation["load_source"].assert_not_called()

    split.write_text("".join(f"case-{index:03d}\n" for index in range(270)))
    setup_evaluation["dataset"].__len__.return_value = 269
    with pytest.raises(ValueError, match="270 validation cases"):
        script.evaluate("configs/probe_anatomy.yaml")
    setup_evaluation["evaluate_artifacts"].assert_not_called()


def test_duplicate_validation_split_ids_are_rejected(setup_evaluation):
    split = setup_evaluation["split"]
    split.write_text("".join(f"case-{index:03d}\n" for index in range(269)) + "case-000\n")
    with pytest.raises(ValueError, match="unique IDs"):
        script.evaluate("configs/probe_anatomy.yaml")
    setup_evaluation["make_dataset"].assert_not_called()


def test_duplicate_dataset_subject_ids_are_rejected(setup_evaluation):
    values = setup_evaluation
    values["dataset"].records[-1] = {"subject_id": "case-000"}
    with pytest.raises(ValueError, match="Duplicate validation dataset subject IDs:.*case-000"):
        script.evaluate("configs/probe_anatomy.yaml")
    values["evaluate_artifacts"].assert_not_called()


def test_same_count_with_mismatched_case_id_is_rejected(setup_evaluation):
    values = setup_evaluation
    values["dataset"].records[-1] = {"subject_id": "case-999"}
    with pytest.raises(ValueError, match=r"missing=\['case-269'\].*unexpected=\['case-999'\]"):
        script.evaluate("configs/probe_anatomy.yaml")
    values["evaluate_artifacts"].assert_not_called()


def test_exact_identity_accepts_different_dataset_order(setup_evaluation):
    values = setup_evaluation
    values["dataset"].records.reverse()
    assert script.evaluate("configs/probe_anatomy.yaml") == {"case_count": 270}
    values["evaluate_artifacts"].assert_called_once()


def test_strict_loading_full_dataset_artifacts_and_provenance(setup_evaluation):
    values = setup_evaluation
    result = script.evaluate("configs/probe_anatomy.yaml", save_predictions=["case-001"])
    assert result == {"case_count": 270}
    values["load_source"].assert_called_once_with(values["source"].resolve())
    values["construct_model"].assert_called_once_with(values["load_source"].return_value[0], "anatomy")
    values["load_checkpoint"].assert_called_once_with(
        values["checkpoint"].resolve(), values["model"], None,
        {key: value for key, value in values["config"].items() if key != "epochs"},
        values["source_info"],
    )
    values["make_dataset"].assert_called_once_with(values["data_config"], "val")
    values["evaluate_artifacts"].assert_called_once_with(
        values["model"], values["dataset"], values["checkpoint"].parent / "evaluation",
        "cpu", 0.5, (8, 9, 10), ["case-001"],
    )
    provenance = json.loads((values["checkpoint"].parent / "evaluation" /
                             "evaluation_provenance.json").read_text())
    assert provenance["latent_source"] == "anatomy"
    assert provenance["probe_checkpoint_sha256"] == file_sha256(values["checkpoint"])
    assert provenance["validation_split_sha256"] == file_sha256(values["split"])
    assert provenance["data_config_sha256"] == file_sha256(values["data_path"])
    assert provenance["frozen_source_sha256"] == values["source_info"]["sha256"]
    assert provenance["probe_checkpoint_best_epoch"] == 10
    assert provenance["probe_architecture"] == PROBE_ARCHITECTURE
    assert provenance["git_dirty"] is True


@pytest.mark.parametrize("name", script.EVALUATION_FILES)
def test_existing_evaluation_artifacts_are_not_overwritten(setup_evaluation, name):
    values = setup_evaluation
    output = values["checkpoint"].parent / "evaluation"
    output.mkdir()
    artifact = output / name
    artifact.write_text("existing result")
    with pytest.raises(FileExistsError, match="already exists"):
        script.evaluate("configs/probe_anatomy.yaml")
    assert artifact.read_text() == "existing result"
    values["evaluate_artifacts"].assert_not_called()


@pytest.mark.parametrize("protected", ["training", "source", "baseline"])
def test_protected_output_directories_are_rejected(setup_evaluation, protected):
    values = setup_evaluation
    output = {
        "training": values["checkpoint"].parent,
        "source": values["source"].parent,
        "baseline": Path("outputs/baseline_unet3d_full"),
    }[protected]
    with pytest.raises(ValueError, match="Evaluation output must be separate"):
        script.evaluate("configs/probe_anatomy.yaml", output_dir=output)
    values["evaluate_artifacts"].assert_not_called()
