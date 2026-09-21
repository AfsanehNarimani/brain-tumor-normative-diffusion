"""Contract tests for the E1-A pathology-accessibility probes."""
import copy
import hashlib

import pytest
import torch

from losses.segmentation_loss import SegmentationLoss
from models.latent_probe import (FrozenLatentProbe, LatentProbeDecoder3D, PROBE_ARCHITECTURE,
                                 E1B_SOURCE_MODEL, SOURCE_MODEL, load_source)
from models.joint_reconstruction_unet3d import JointReconstructionUNet3D
from models.latent_split_unet3d import LatentSplitUNet3D
from scripts.train_latent_probe import (load_probe_checkpoint, read_config,
                                        CONFIG_CHOICES, save_probe_checkpoint, select_config,
                                        validate_pair)


@pytest.fixture
def source_checkpoint(tmp_path):
    model = LatentSplitUNet3D()
    path = tmp_path / "source.pt"
    torch.save(dict(model=model.state_dict(), epoch=15, config={"model": SOURCE_MODEL}), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def test_pair_protocol_and_deterministic_initialization():
    anatomy, disease = validate_pair()
    assert anatomy["latent_source"] == "anatomy" and disease["latent_source"] == "disease"
    assert anatomy["output_dir"] != disease["output_dir"]
    torch.manual_seed(anatomy["seed"])
    a = LatentProbeDecoder3D()
    torch.manual_seed(disease["seed"])
    d = LatentProbeDecoder3D()
    assert all(torch.equal(value, d.state_dict()[key]) for key, value in a.state_dict().items())
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in d.parameters())


def test_source_checkpoint_is_strict(source_checkpoint, tmp_path):
    path, digest = source_checkpoint
    model, info = load_source(path, expected_sha256=digest)
    assert info["sha256"] == digest and info["epoch"] == 15
    assert info["config"]["model"] == SOURCE_MODEL
    assert info["parameter_count"] == 5971716
    assert all(not p.requires_grad for p in model.parameters())
    with pytest.raises(FileNotFoundError):
        load_source(tmp_path / "absent.pt", expected_sha256=digest)
    with pytest.raises(ValueError, match="SHA256"):
        load_source(path, expected_sha256="0" * 64)
    state = torch.load(path, weights_only=True)
    for change, message in [({"epoch": 14}, "epoch"),
                            ({"config": {"model": {**SOURCE_MODEL, "name": "other"}}}, "config")]:
        bad = tmp_path / (message + ".pt")
        torch.save({**state, **change}, bad)
        with pytest.raises(ValueError, match=message):
            load_source(bad, expected_sha256=hashlib.sha256(bad.read_bytes()).hexdigest())
    bad = tmp_path / "missing_weight.pt"
    broken = copy.deepcopy(state)
    broken["model"].pop(next(iter(broken["model"])))
    torch.save(broken, bad)
    with pytest.raises(RuntimeError):
        load_source(bad, expected_sha256=hashlib.sha256(bad.read_bytes()).hexdigest())


def test_e1b_source_checkpoint_identity_freezing_and_extraction(tmp_path):
    source = JointReconstructionUNet3D()
    path = tmp_path / "e1b.pt"
    torch.save(dict(model=source.state_dict(), epoch=13,
                    config={"model": E1B_SOURCE_MODEL}), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, info = load_source(path, source_experiment="e1b", expected_sha256=digest)
    assert isinstance(loaded, JointReconstructionUNet3D)
    assert info["experiment"] == "e1b" and info["model"] == "joint_reconstruction_unet3d"
    assert info["epoch"] == 13 and all(not parameter.requires_grad for parameter in loaded.parameters())
    assert not loaded.training and all(not module.training for module in loaded.modules())

    loaded.decoder.register_forward_pre_hook(
        lambda *_: (_ for _ in ()).throw(AssertionError("segmentation decoder called")))
    loaded.reconstruction_decoder.register_forward_pre_hook(
        lambda *_: (_ for _ in ()).throw(AssertionError("reconstruction decoder called")))
    image = torch.randn(1, 4, 8, 8, 8)
    probes = {name: FrozenLatentProbe(loaded, name) for name in ("anatomy", "disease")}
    latents = {name: probe.extract_latent(image) for name, probe in probes.items()}
    assert all(value.shape == (1, 256, 1, 1, 1) for value in latents.values())
    assert not torch.equal(latents["anatomy"], latents["disease"])
    assert all(parameter.requires_grad for probe in probes.values()
               for parameter in probe.probe.parameters())
    assert all(not parameter.requires_grad for parameter in loaded.parameters())

    with pytest.raises(ValueError, match="SHA256"):
        load_source(path, source_experiment="e1b", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="epoch"):
        load_source(path, source_experiment="e1b", expected_sha256=digest, expected_epoch=12)
    bad = tmp_path / "wrong-model.pt"
    torch.save(dict(model=source.state_dict(), epoch=13,
                    config={"model": {**E1B_SOURCE_MODEL, "name": "other"}}), bad)
    with pytest.raises(ValueError, match="architecture/config"):
        load_source(bad, source_experiment="e1b",
                    expected_sha256=hashlib.sha256(bad.read_bytes()).hexdigest())


@pytest.mark.parametrize("latent_source", ["anatomy", "disease"])
def test_freeze_selection_logits_and_fixed_latent_overfit(latent_source):
    torch.manual_seed(7)
    source = LatentSplitUNet3D()
    model = FrozenLatentProbe(source, latent_source)
    assert all(not p.requires_grad for p in source.parameters())
    assert all(p.requires_grad for p in model.probe.parameters())
    model.train()
    assert not source.training and all(not module.training for module in source.modules())
    # The original decoder must never run. A hook also proves it supplies no skips.
    source.decoder.register_forward_pre_hook(lambda *_: (_ for _ in ()).throw(AssertionError("source decoder called")))
    supplied = []
    model.probe.register_forward_pre_hook(lambda _m, args: supplied.append(args))
    image = torch.randn(1, 4, 8, 8, 8)
    with torch.no_grad():
        bottleneck = source.encoder(image)[-1]
        expected = getattr(source.projection, latent_source + "_projector")(bottleneck)
    latent = model.extract_latent(image)
    torch.testing.assert_close(latent, expected)
    logits = model(image)
    assert logits.shape == (1, 4, 8, 8, 8)
    assert len(supplied[-1]) == 2 and supplied[-1][0].shape == (1, 256, 1, 1, 1)
    assert not supplied[-1][0].requires_grad
    target = torch.zeros_like(logits)
    target[:, 1, 2:6, 2:6, 2:6] = 1
    criterion = SegmentationLoss()
    loss = criterion(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.probe.parameters())
    for module in (source.encoder, source.projection.anatomy_projector,
                   source.projection.disease_projector, source.decoder):
        assert all(p.grad is None for p in module.parameters())
    # Fixed frozen latent isolates the trainable decoder's ability to optimize.
    optimizer = torch.optim.AdamW(model.probe.parameters(), lr=0.001, weight_decay=0.00001)
    initial = criterion(model.probe(latent, (8, 8, 8)), target).item()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        criterion(model.probe(latent, (8, 8, 8)), target).backward()
        optimizer.step()
    final = criterion(model.probe(latent, (8, 8, 8)), target).item()
    assert final < initial, (latent_source, initial, final)


def test_only_selected_latent_changes_probe_input():
    torch.manual_seed(8)
    source = LatentSplitUNet3D()
    torch.manual_seed(42)
    a = FrozenLatentProbe(source, "anatomy")
    torch.manual_seed(42)
    d = FrozenLatentProbe(source, "disease")
    assert all(torch.equal(value, d.probe.state_dict()[key]) for key, value in a.probe.state_dict().items())
    image = torch.randn(1, 4, 8, 9, 10)
    assert a.extract_latent(image).shape == d.extract_latent(image).shape
    assert not torch.equal(a.extract_latent(image), d.extract_latent(image))
    assert a(image).shape == d(image).shape == (1, 4, 8, 9, 10)


def test_probe_checkpoint_external_source_and_metadata(source_checkpoint, tmp_path):
    path, digest = source_checkpoint
    source, info = load_source(path, expected_sha256=digest)
    model = FrozenLatentProbe(source, "anatomy")
    optimizer = torch.optim.AdamW(model.probe.parameters())
    config = {"latent_source": "anatomy", "output_dir": str(tmp_path / "anatomy")}
    checkpoint = tmp_path / "probe.pt"
    save_probe_checkpoint(checkpoint, model, optimizer, 1, 0.3, 1, config, info)
    state = torch.load(checkpoint, weights_only=True)
    assert "source" not in state and "model" not in state
    assert state["probe_architecture"] == PROBE_ARCHITECTURE
    assert state["source_checkpoint"]["sha256"] == digest
    assert state["latent_source"] == "anatomy" and state["epoch"] == 1
    assert load_probe_checkpoint(checkpoint, model, optimizer, config, info)["best_metric"] == 0.3
    with pytest.raises(ValueError, match="identity mismatch"):
        load_probe_checkpoint(checkpoint, model, optimizer, {**config, "latent_source": "disease"}, info)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_probe_checkpoint(checkpoint, model, optimizer, config, {**info, "sha256": "0" * 64})
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_probe_checkpoint(checkpoint, model, optimizer, config, info)


def test_arbitrary_geometry_and_raw_logits():
    decoder = LatentProbeDecoder3D()
    with torch.no_grad():
        decoder.head.weight.zero_()
        decoder.head.bias.fill_(2)
        logits = decoder(torch.randn(1, 256, 2, 2, 2), (13, 17, 19))
    assert logits.shape == (1, 4, 13, 17, 19)
    assert torch.all(logits == 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_synthetic_forward():
    model = FrozenLatentProbe(LatentSplitUNet3D(), "disease").cuda().train()
    with torch.no_grad():
        logits = model(torch.randn(1, 4, 16, 16, 16, device="cuda"))
    assert logits.shape == (1, 4, 16, 16, 16)
    assert torch.isfinite(logits).all()
    assert not model.source.training


def test_default_source_absence_is_clear():
    config = read_config("configs/probe_anatomy.yaml")
    assert config["source_checkpoint"] == "outputs/e1a_latent_split/best.pt"
    assert config["threshold"] == 0.5


def test_e1b_pair_protocol_and_explicit_config_selection():
    anatomy, disease = validate_pair("configs/e1b_probe_anatomy.yaml",
                                     "configs/e1b_probe_disease.yaml")
    ignored = {"experiment_name", "latent_source", "output_dir"}
    assert {key: value for key, value in anatomy.items() if key not in ignored} == {
        key: value for key, value in disease.items() if key not in ignored}
    assert anatomy["batch_size"] == disease["batch_size"] == 4
    assert anatomy["source_experiment"] == disease["source_experiment"] == "e1b"
    assert select_config("configs/e1b_probe_anatomy.yaml")["latent_source"] == "anatomy"
    assert select_config("configs/e1b_probe_disease.yaml")["latent_source"] == "disease"
    assert set(CONFIG_CHOICES) == {"configs/probe_anatomy.yaml", "configs/probe_disease.yaml",
                                   "configs/e1b_probe_anatomy.yaml", "configs/e1b_probe_disease.yaml",
                                   "configs/e1c_probe_anatomy.yaml", "configs/e1c_probe_disease.yaml"}
    with pytest.raises(ValueError, match="Unsupported probe config"):
        select_config("configs/anatomy-but-unsupported.yaml")
    with pytest.raises(ValueError, match="same source experiment"):
        validate_pair("configs/probe_anatomy.yaml", "configs/e1b_probe_disease.yaml")


def test_e1a_configs_remain_original_protocol():
    anatomy, disease = validate_pair()
    for config, latent, output in ((anatomy, "anatomy", "outputs/e1a_probes/anatomy"),
                                   (disease, "disease", "outputs/e1a_probes/disease")):
        assert "source_experiment" not in config
        assert config["latent_source"] == latent
        assert config["output_dir"] == output
        assert config["source_checkpoint"] == "outputs/e1a_latent_split/best.pt"
        assert config["batch_size"] == 1
