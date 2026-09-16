import csv
import json

import pytest
import torch
from losses.segmentation_loss import SegmentationLoss
from models.latent_split_unet3d import LatentSplitUNet3D
from scripts import tiny_overfit_e1a as script
from training.checkpointing import load_checkpoint


def test_cache_first_training_case_once_without_augmentation(tmp_path, monkeypatch):
    split = tmp_path/'train.txt'
    split.write_text('train-b\ntrain-a\n')
    calls = []
    class Dataset:
        def __init__(self, **kwargs):
            assert kwargs['training'] and kwargs['normalize']
            assert kwargs['foreground_sampling_probability'] == 1.0
            assert kwargs['augmentation'] == {'enabled': False}
            assert kwargs['generator'].initial_seed() == 7
            self.records = [{'subject_id': name} for name in ('train-a', 'train-b', 'validation')]
        def __getitem__(self, index):
            calls.append(self.records[index]['subject_id'])
            target = torch.zeros(4,96,96,96)
            target[3,1,2,3] = 1
            return dict(case_id=self.records[index]['subject_id'], image=torch.zeros_like(target),
                        target=target, crop_start=(1,2,3))
    monkeypatch.setattr(script, 'BraTSDataset', Dataset)
    sample = script.cache_patch('unused', str(split), 7)
    assert calls == ['train-b']
    assert sample['image'].shape == (1,4,96,96,96)
    assert sample['target'].sum() == 1


def test_fixed_tensor_identity_and_artifacts(tmp_path, monkeypatch):
    torch.manual_seed(3)
    image = torch.randn(1,4,8,8,8)
    target = torch.zeros_like(image)
    target[:,1,2:6,2:6,2:6] = 1
    originals = image.clone(), target.clone()
    cache_calls = []
    def cached(*args):
        cache_calls.append(args)
        return dict(case_id='train-only', crop_start=[1,2,3], image=image, target=target)
    monkeypatch.setattr(script, 'cache_patch', cached)
    # Exercise the full artifact/optimization path cheaply. Production CLI has
    # no width/patch overrides and separately enforces full-model shapes/count.
    small = LatentSplitUNet3D(base_channels=2)
    monkeypatch.setattr(script, 'EXPECTED_PARAMETERS', sum(p.numel() for p in small.parameters()))
    seen = []
    def model_factory(**kwargs):
        assert kwargs == dict(in_channels=4, out_channels=4, base_channels=32)
        small.register_forward_pre_hook(lambda m, args: seen.append(args[0]))
        return small
    monkeypatch.setattr(script, 'LatentSplitUNet3D', model_factory)
    monkeypatch.setattr(script, 'verify_shapes', lambda result: {name:list(getattr(result,name).shape)
                        for name in ('logits','anatomy_latent','disease_latent')})
    split = tmp_path/'train.txt'; split.write_text('train-only\n')
    output = tmp_path/'sanity'
    summary = script.run('unused', str(split), 'cpu', 3, 42, str(output))
    assert len(cache_calls) == 1 and all(value is image for value in seen)
    assert torch.equal(image, originals[0]) and torch.equal(target, originals[1])
    assert {p.name for p in output.iterdir()} == {'config.json','history.csv','summary.json','initial.pt','final.pt'}
    history = list(csv.DictReader((output/'history.csv').open()))
    assert [int(row['step']) for row in history] == [1,2,3]
    assert all(float(row['learning_rate']) == .001 for row in history)
    assert summary['initial_loss'] == pytest.approx(float(history[0]['loss']))
    assert summary['minimum_loss'] == min(summary['initial_loss'], summary['final_loss'], *(float(row['loss']) for row in history))
    assert summary['relative_loss_reduction'] == pytest.approx((summary['initial_loss']-summary['final_loss'])/summary['initial_loss'])
    assert all(summary['gradient_checks'].values())
    assert summary['optimizer'] == 'AdamW' and summary['weight_decay'] == 1e-5
    initial = torch.load(output/'initial.pt', weights_only=True)
    final = torch.load(output/'final.pt', weights_only=True)
    for name, value in initial['model'].items():
        if 'anatomy_projector' in name:
            assert torch.equal(value, final['model'][name])
    restored = LatentSplitUNet3D(base_channels=2)
    load_checkpoint(output/'final.pt', restored)
    with torch.no_grad():
        assert SegmentationLoss()(restored(image), target).item() == pytest.approx(summary['final_loss'])
    assert json.loads((output/'summary.json').read_text()) == summary


def test_gradient_structure_rejects_disconnected_disease():
    model = LatentSplitUNet3D(base_channels=2)
    with pytest.raises(RuntimeError, match='gradient structure'):
        script.check_gradients(model)
    SegmentationLoss()(model(torch.randn(1,4,8,8,8)), torch.zeros(1,4,8,8,8)).backward()
    assert all(script.check_gradients(model).values())
    model.projection.anatomy_projector.weight.grad = torch.zeros_like(model.projection.anatomy_projector.weight)
    with pytest.raises(RuntimeError, match='gradient structure'):
        script.check_gradients(model)


def test_real_default_shape_validation():
    with torch.device('meta'):
        result = LatentSplitUNet3D().forward_with_latents(torch.empty(1,4,96,96,96))
    assert script.verify_shapes(result)['disease_latent'] == [1,256,12,12,12]
    with torch.device('meta'):
        wrong = LatentSplitUNet3D(base_channels=2).forward_with_latents(torch.empty(1,4,8,8,8))
    with pytest.raises(ValueError, match='tensor shapes'):
        script.verify_shapes(wrong)


def test_failures_before_loading(tmp_path):
    with pytest.raises(ValueError, match='positive'):
        script.run('unused','unused','cpu',0,42,str(tmp_path))
    with pytest.raises(ValueError, match='separate'):
        script.run('unused','unused','cpu',1,42,'outputs/baseline_unet3d_full')
    (tmp_path/'existing').write_text('protected')
    with pytest.raises(FileExistsError, match='overwrite'):
        script.run('unused','unused','cpu',1,42,str(tmp_path))
