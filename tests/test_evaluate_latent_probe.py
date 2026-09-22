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
    source_info = {"path": str(source.resolve()), "sha256": file_sha256(source), "epoch": 15,
                   "experiment": "e1a", "model": "latent_split_unet3d"}
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


def test_source_experiment_mismatch_is_rejected(setup_evaluation):
    setup_evaluation["config"]["source_experiment"] = "e1b"
    with pytest.raises(ValueError, match="source_experiment mismatch"):
        script.evaluate("configs/probe_anatomy.yaml")
    setup_evaluation["load_source"].assert_not_called()


def test_loaded_source_identity_mismatch_is_rejected(setup_evaluation):
    setup_evaluation["source_info"]["experiment"] = "e1b"
    with pytest.raises(ValueError, match="source experiment identity mismatch"):
        script.evaluate("configs/probe_anatomy.yaml")
    setup_evaluation["load_checkpoint"].assert_not_called()

    setup_evaluation["source_info"].update(experiment="e1a", model="joint_reconstruction_unet3d")
    with pytest.raises(ValueError, match="source model identity mismatch"):
        script.evaluate("configs/probe_anatomy.yaml")
    setup_evaluation["load_checkpoint"].assert_not_called()


def test_unsupported_config_is_rejected_even_with_supported_basename(setup_evaluation, tmp_path):
    unsupported = tmp_path / "probe_anatomy.yaml"
    unsupported.write_text("latent_source: anatomy\n")
    with pytest.raises(ValueError, match="Unsupported probe config"):
        script.evaluate(unsupported)
    setup_evaluation["read_config"].assert_not_called()


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
    values["load_source"].assert_called_once_with(values["source"].resolve(), source_experiment="e1a")
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
    assert provenance["source_experiment"] == "e1a"
    assert provenance["source_model_identity"] == "latent_split_unet3d"
    assert provenance["probe_checkpoint_sha256"] == file_sha256(values["checkpoint"])
    assert provenance["validation_split_sha256"] == file_sha256(values["split"])
    assert provenance["data_config_sha256"] == file_sha256(values["data_path"])
    assert provenance["frozen_source_sha256"] == values["source_info"]["sha256"]
    assert provenance["probe_checkpoint_best_epoch"] == 10
    assert provenance["probe_architecture"] == PROBE_ARCHITECTURE
    assert provenance["git_dirty"] is True


@pytest.mark.parametrize(("config_path", "experiment", "latent_source", "model_identity"), [
    ("configs/probe_anatomy.yaml", "e1a", "anatomy", "latent_split_unet3d"),
    ("configs/probe_disease.yaml", "e1a", "disease", "latent_split_unet3d"),
    ("configs/e1b_probe_anatomy.yaml", "e1b", "anatomy", "joint_reconstruction_unet3d"),
    ("configs/e1b_probe_disease.yaml", "e1b", "disease", "joint_reconstruction_unet3d"),
])
def test_explicit_supported_protocols(setup_evaluation, config_path, experiment,
                                      latent_source, model_identity):
    values = setup_evaluation
    values["config"]["latent_source"] = latent_source
    values["model"].latent_source = latent_source
    values["state"]["latent_source"] = latent_source
    if experiment == "e1b":
        values["config"]["source_experiment"] = "e1b"
        values["source_info"].update(experiment="e1b", model=model_identity, epoch=13)
    assert script.evaluate(config_path) == {"case_count": 270}
    values["load_source"].assert_called_once_with(values["source"].resolve(),
                                                   source_experiment=experiment)
    values["construct_model"].assert_called_once_with(
        values["load_source"].return_value[0], latent_source)
    provenance = json.loads((values["checkpoint"].parent / "evaluation" /
                             "evaluation_provenance.json").read_text())
    assert provenance["source_experiment"] == experiment
    assert provenance["source_model_identity"] == model_identity
    assert provenance["latent_source"] == latent_source


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


@pytest.fixture(params=["anatomy", "disease"])
def e1c_evaluation(setup_evaluation, request):
    values = setup_evaluation
    latent = request.param
    values["config"].update(
        latent_source=latent, source_experiment="e1c",
        source_model_identity="anatomy_constrained_joint_reconstruction_unet3d",
        expected_source_checkpoint_sha256=script.E1C_SOURCE_SHA256,
    )
    values["source_info"].update(
        experiment="e1c", model=values["config"]["source_model_identity"],
        sha256=script.E1C_SOURCE_SHA256, epoch=11,
    )
    values["model"].latent_source = latent
    values["state"]["latent_source"] = latent
    values["config_path"] = f"configs/e1c_probe_{latent}.yaml"
    return values


def test_e1c_evaluation_flow_and_provenance(e1c_evaluation):
    values = e1c_evaluation
    assert values["config_path"] in script.CONFIG_CHOICES
    assert script.evaluate(values["config_path"]) == {"case_count": 270}
    values["load_source"].assert_called_once_with(values["source"].resolve(), source_experiment="e1c")
    values["construct_model"].assert_called_once_with(
        values["load_source"].return_value[0], values["config"]["latent_source"])
    values["load_checkpoint"].assert_called_once_with(
        values["checkpoint"].resolve(), values["model"], None,
        {key: value for key, value in values["config"].items() if key != "epochs"},
        values["source_info"],
    )
    values["make_dataset"].assert_called_once_with(values["data_config"], "val")
    output = values["checkpoint"].parent / "evaluation"
    values["evaluate_artifacts"].assert_called_once_with(
        values["model"], values["dataset"], output, "cpu", 0.5, (8, 9, 10), (),
    )
    provenance = json.loads((output / "evaluation_provenance.json").read_text())
    expected = {
        "source_experiment": "e1c",
        "source_model_identity": "anatomy_constrained_joint_reconstruction_unet3d",
        "latent_source": values["config"]["latent_source"],
        "probe_checkpoint": str(values["checkpoint"].resolve()),
        "probe_checkpoint_sha256": file_sha256(values["checkpoint"]),
        "probe_checkpoint_epoch": 12, "probe_checkpoint_best_epoch": 10,
        "probe_checkpoint_best_metric": 0.61,
        "frozen_source_checkpoint": str(values["source"].resolve()),
        "frozen_source_sha256": "cf5a04a2d845cecb45a0ea87d617abad0afab9ad4264abce8342a6045563ee53",
        "frozen_source_epoch": 11,
        "validation_split": str(values["split"].resolve()),
        "validation_split_sha256": file_sha256(values["split"]),
        "validation_case_count": 270, "threshold": 0.5,
        "git_commit_sha": "abc123", "git_dirty": True,
        "git_status_short": [" M configs/data.yaml"], "device_requested": "cpu",
    }
    assert expected.items() <= provenance.items()
    assert {"python_version", "pytorch_version", "numpy_version", "nibabel_version"} <= provenance.keys()
    assert not (output / "predictions").exists()


@pytest.mark.parametrize("key,value,match", [
    ("latent_source", "invalid", "latent_source mismatch"),
    ("source_experiment", "e1b", "source_experiment mismatch"),
    ("source_model_identity", "joint_reconstruction_unet3d", "model identity mismatch"),
    ("source_model_identity", None, "model identity mismatch"),
    ("expected_source_checkpoint_sha256", "0" * 64, "SHA256"),
    ("expected_source_checkpoint_sha256", None, "SHA256"),
])
def test_e1c_config_identity_rejected(e1c_evaluation, key, value, match):
    values = e1c_evaluation
    values["config"][key] = value
    with pytest.raises(ValueError, match=match):
        script.evaluate(values["config_path"])
    values["load_source"].assert_not_called()


@pytest.mark.parametrize("key,value", [
    ("experiment", "e1b"), ("model", "joint_reconstruction_unet3d"),
])
def test_e1c_loaded_source_identity_rejected(e1c_evaluation, key, value):
    values = e1c_evaluation
    values["source_info"][key] = value
    with pytest.raises(ValueError, match="identity mismatch"):
        script.evaluate(values["config_path"])
    values["load_checkpoint"].assert_not_called()


def test_e1c_config_path_must_be_exact(e1c_evaluation, tmp_path):
    values = e1c_evaluation
    with pytest.raises(ValueError, match="Unsupported probe config"):
        script.evaluate(tmp_path / Path(values["config_path"]).name)
    values["read_config"].assert_not_called()


@pytest.mark.parametrize("name", script.EVALUATION_FILES)
def test_e1c_overwrite_protection(e1c_evaluation, name):
    values = e1c_evaluation
    output = values["checkpoint"].parent / "evaluation"
    output.mkdir()
    artifact = output / name
    artifact.write_text("existing result")
    with pytest.raises(FileExistsError, match="already exists"):
        script.evaluate(values["config_path"])
    assert artifact.read_text() == "existing result"
    values["evaluate_artifacts"].assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("config", {}), ("latent_source", "invalid"), ("probe_architecture", {}),
    ("source_checkpoint", {"experiment": "e1b"}),
])
def test_e1c_strict_probe_checkpoint_rejected(e1c_evaluation, monkeypatch, field, value):
    from scripts import train_latent_probe

    values = e1c_evaluation
    # Use the real checkpoint validator; only deserialize a synthetic state.
    state = {
        **values["state"],
        "config": {k: v for k, v in values["config"].items() if k != "epochs"},
        "source_checkpoint": {k: values["source_info"][k]
                              for k in ("path", "sha256", "epoch", "experiment", "model")},
    }
    if field == "source_checkpoint":
        state[field].update(value)
    else:
        state[field] = value
    monkeypatch.setattr(train_latent_probe.torch, "load", lambda *a, **kw: state)
    monkeypatch.setattr(train_latent_probe, "file_sha256", lambda path: script.E1C_SOURCE_SHA256)
    monkeypatch.setattr(script, "load_probe_checkpoint", train_latent_probe.load_probe_checkpoint)
    with pytest.raises(ValueError, match="identity mismatch"):
        script.evaluate(values["config_path"])
    values["evaluate_artifacts"].assert_not_called()
