import pytest
import torch
from models.baseline_unet3d import BaselineUNet3D
from models.latent_split_unet3d import LatentSplitUNet3D, LatentSplitOutput
from scripts.train_baseline import build_model
from losses.segmentation_loss import SegmentationLoss


def test_baseline_factory_unchanged():
    torch.manual_seed(12)
    direct = BaselineUNet3D(base_channels=2)
    torch.manual_seed(12)
    factory = build_model({'base_channels': 2})
    image = torch.randn(1, 4, 8, 9, 10)
    assert type(factory) is BaselineUNet3D
    assert torch.equal(direct(image), factory(image))


def test_forward_and_gradients():
    model = LatentSplitUNet3D(in_channels=3, out_channels=5, base_channels=2)
    image = torch.randn(1, 3, 17, 18, 19)
    output = model.forward_with_latents(image)
    assert isinstance(output, LatentSplitOutput)
    assert output.logits.shape == (1, 5, 17, 18, 19)
    assert output.anatomy_latent.shape == output.disease_latent.shape == (1, 16, 2, 2, 2)
    assert torch.equal(model(image), output.logits)
    anatomy, disease = model.projection.anatomy_projector, model.projection.disease_projector
    assert not {id(p) for p in anatomy.parameters()} & {id(p) for p in disease.parameters()}
    loss = SegmentationLoss()(output.logits, torch.zeros_like(output.logits))
    loss.backward()
    for module in (model.encoder, disease, model.decoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert any(p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in anatomy.parameters())
    with torch.no_grad():
        model.decoder.head.weight.zero_()
        model.decoder.head.bias.fill_(2)
    assert torch.all(model(image) == 2)  # Raw logits, not sigmoid.


def test_default_patch_shape_and_counts():
    # Meta tensors validate full default geometry without 96^3 CPU allocations.
    with torch.device('meta'):
        model = LatentSplitUNet3D()
        output = model.forward_with_latents(torch.empty(1, 4, 96, 96, 96))
    assert output.logits.shape == (1, 4, 96, 96, 96)
    assert output.anatomy_latent.shape == output.disease_latent.shape == (1, 256, 12, 12, 12)
    assert sum(p.numel() for p in model.parameters()) == 5971716
    assert sum(p.numel() for p in BaselineUNet3D().parameters()) == 5840132


@pytest.mark.parametrize('shape', [(1,4,7,8,8), (1,3,8,8,8), (4,8,8,8)])
def test_invalid_input_matches_baseline(shape):
    for model in (BaselineUNet3D(base_channels=2), LatentSplitUNet3D(base_channels=2)):
        with pytest.raises(ValueError, match='Expected'):
            model(torch.zeros(shape))


@pytest.mark.parametrize('kwargs', [{'base_channels':0}, {'in_channels':True}, {'latent_channels':128}, {'latent_projection':'split'}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LatentSplitUNet3D(**kwargs)


def test_single_encoder_and_original_skips():
    model = LatentSplitUNet3D(base_channels=2)
    features, supplied = [], []
    model.encoder.register_forward_hook(lambda m, args, result: features.append(result))
    model.decoder.register_forward_pre_hook(lambda m, args: supplied.append(args[0]))
    model.forward_with_latents(torch.randn(1,4,8,8,8))
    assert len(features) == 1
    assert all(a is b for a,b in zip(features[0][:3], supplied[0][:3]))


def test_smoke_checkpoint_and_factory(tmp_path):
    from scripts.smoke_test_e1a import run
    assert isinstance(build_model({'name':'latent_split_unet3d','base_channels':2}), LatentSplitUNet3D)
    result = run('cpu', str(tmp_path), patch_size=8, base_channels=2)
    assert result['checkpoint_round_trip']
    assert result['parameter_delta'] == 2*(16*16+16)


def test_e1a_rejects_frozen_output():
    from scripts.train_baseline import run_training
    with pytest.raises(ValueError, match='distinct output'):
        run_training({}, {'name':'latent_split_unet3d'}, {'output_dir':'outputs/baseline_unet3d_full'})


def test_e1a_config_selection(monkeypatch):
    from scripts import train_baseline as script
    captured = []
    monkeypatch.setattr('sys.argv', ['train_baseline', '--model-config', 'configs/model_e1a.yaml', '--config', 'configs/train_e1a.yaml'])
    monkeypatch.setattr(script, 'run_training', lambda *args: captured.append(args))
    script.main()
    data, model, train, debug = captured[0]
    assert model['latent_channels'] == 256 and model['name'] == 'latent_split_unet3d'
    assert train['output_dir'] == 'outputs/e1a_latent_split'
    assert data['patch_size'] == [96, 96, 96]
    assert not debug
