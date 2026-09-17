"""Sanity entry point tests; production requires the exact E1-A best checkpoint."""
import csv
import json

import pytest
import torch

from models.latent_split_unet3d import LatentSplitUNet3D
from scripts import tiny_overfit_latent_probes as script


def test_fixed_latents_do_not_call_source_decoder_or_track_gradients():
    source = LatentSplitUNet3D().requires_grad_(False).eval()
    source.decoder.register_forward_pre_hook(
        lambda *_: (_ for _ in ()).throw(AssertionError("E1-A decoder called")))
    image = torch.randn(1, 4, 8, 8, 8)
    latents = script.extract_fixed_latents(source, image)
    assert set(latents) == {"anatomy", "disease"}
    assert all(value.shape == (1, 256, 1, 1, 1) and not value.requires_grad
               for value in latents.values())
    assert not torch.equal(latents["anatomy"], latents["disease"])
    assert all(p.grad is None for p in source.parameters())


def test_fixed_latent_optimization_and_audits(tmp_path):
    torch.manual_seed(4)
    source = LatentSplitUNet3D().requires_grad_(False).eval()
    image = torch.randn(1, 4, 8, 8, 8)
    target = torch.zeros(1, 4, 8, 8, 8)
    target[:, 1, 2:6, 2:6, 2:6] = 1
    latents = script.extract_fixed_latents(source, image)
    source_before = script.state_snapshot(source)
    fitted = {}
    for name in ("anatomy", "disease"):
        fitted[name] = script.fit_fixed_latent(latents[name], target, seed=42, steps=3,
                                                learning_rate=0.001, weight_decay=0.00001,
                                                history_path=tmp_path / f"{name}.csv", source=source)
        result = fitted[name]["result"]
        assert result["all_losses_finite"]
        assert not result["source_gradient_present"]
        assert result["probe_gradient_present"] and result["probe_parameters_changed"]
        assert result["final_loss"] < result["initial_loss"]
        assert result["minimum_loss"] <= result["final_loss"]
        assert result["steps"] == 3 and len(result["loss_trajectory"]) == 4
        rows = list(csv.DictReader((tmp_path / f"{name}.csv").open()))
        assert len(rows) == 3
        assert float(rows[0]["loss_before_update"]) == pytest.approx(result["initial_loss"])
    assert all(torch.equal(value, fitted["disease"]["initial_state"][key])
               for key, value in fitted["anatomy"]["initial_state"].items())
    assert not script.state_changed(source_before, source)


def test_run_artifact_and_output_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "SANITY_ROOT", tmp_path / "sanity")
    monkeypatch.setattr(script, "PATCH_SIZE", (8, 8, 8))
    monkeypatch.setattr(script, "read_config", lambda _: {"root": "unused", "train_cases": str(tmp_path / "train.txt"),
                                                       "patch_size": [8, 8, 8]})
    (tmp_path / "train.txt").write_text("training-case\n")
    target = torch.zeros(1, 4, 8, 8, 8)
    target[:, 2, 2:6, 2:6, 2:6] = 1
    calls = []
    def fake_patch(root, split, seed):
        calls.append((root, split, seed))
        return dict(case_id="training-case", crop_start=[1, 2, 3],
                    image=torch.randn(1, 4, 8, 8, 8), target=target.clone())
    monkeypatch.setattr(script, "cache_patch", fake_patch)
    source = LatentSplitUNet3D().requires_grad_(False).eval()
    def fake_load(path, *, expected_sha256, expected_epoch):
        assert path == script.SOURCE_CHECKPOINT
        assert expected_sha256 == script.SOURCE_SHA256 and expected_epoch == 15
        return source, dict(path=str(path), sha256=expected_sha256, epoch=15,
                            config={"model": {"name": "latent_split_unet3d"}},
                            parameter_count=sum(p.numel() for p in source.parameters()))
    monkeypatch.setattr(script, "load_source", fake_load)
    with pytest.raises(ValueError, match="Sanity artifacts"):
        script.run(device="cpu", steps=1, output_dir=str(tmp_path / "official"))
    summary = script.run(device="cpu", steps=2, output_dir=str(tmp_path / "sanity"))
    assert len(calls) == 1
    assert summary["case_id"] == "training-case" and summary["crop_start"] == [1, 2, 3]
    assert summary["identical_initial_probe_states"]
    assert not summary["source_parameters_changed"]
    assert set(summary["results"]) == {"anatomy", "disease"}
    assert all(not r["source_parameters_changed"] and not r["source_gradient_present"]
               for r in summary["results"].values())
    assert {p.name for p in (tmp_path / "sanity").iterdir()} == {
        "anatomy_history.csv", "disease_history.csv", "summary.json"}
    assert json.loads((tmp_path / "sanity/summary.json").read_text()) == summary
    with pytest.raises(FileExistsError, match="overwrite"):
        script.run(device="cpu", steps=1, output_dir=str(tmp_path / "sanity"))


def test_requires_detached_latent(tmp_path):
    with pytest.raises(ValueError, match="detached"):
        script.fit_fixed_latent(torch.randn(1, 256, 1, 1, 1, requires_grad=True),
                                torch.zeros(1, 4, 8, 8, 8), seed=1, steps=1,
                                learning_rate=0.001, weight_decay=0.00001,
                                history_path=tmp_path / "invalid.csv", source=LatentSplitUNet3D())
