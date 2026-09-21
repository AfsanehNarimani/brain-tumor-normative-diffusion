"""Synthetic E1-C probe contracts; no scientific checkpoint or data required."""
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import yaml

from models import latent_probe as models
from models.anatomy_constrained_joint_reconstruction_unet3d import AnatomyConstrainedJointReconstructionUNet3D
from scripts import train_latent_probe as script


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    # Small genuine E1-C checkpoint exercises strict loading without a 119 MB file.
    config = {**models.E1C_SOURCE_MODEL, 'base_channels': 2, 'latent_channels': 16}
    spec = {**models.SOURCE_SPECS['e1c'], 'model': config}
    monkeypatch.setitem(models.SOURCE_SPECS, 'e1c', spec)
    source = AnatomyConstrainedJointReconstructionUNet3D(base_channels=2, latent_channels=16)
    state = dict(model=source.state_dict(), epoch=11,
                 config=dict(model=config, experiment_name=models.E1C_SOURCE_EXPERIMENT))
    path = tmp_path / 'source' / 'source.pt'
    path.parent.mkdir()
    torch.save(state, path)
    # Only the byte digest is substituted. Production pinning stays enabled.
    monkeypatch.setattr(models, 'file_sha256', lambda path: models.E1C_SOURCE_SHA256)
    return path, state


def test_registry_and_exact_scientific_identity():
    spec = models.SOURCE_SPECS['e1c']
    assert spec['constructor'] is AnatomyConstrainedJointReconstructionUNet3D
    assert spec['epoch'] == 11
    assert spec['model'] == models.E1C_SOURCE_MODEL
    assert spec['model']['name'] == 'anatomy_constrained_joint_reconstruction_unet3d'
    assert spec['model']['latent_channels'] == 256
    assert spec['sha256'] == 'cf5a04a2d845cecb45a0ea87d617abad0afab9ad4264abce8342a6045563ee53'
    assert set(models.SOURCE_SPECS) == {'e1a', 'e1b', 'e1c'}


def test_source_loads_strictly_and_frozen(checkpoint):
    path, _ = checkpoint
    source, info = models.load_source(path, source_experiment='e1c')
    assert isinstance(source, AnatomyConstrainedJointReconstructionUNet3D)
    assert info['experiment'] == 'e1c' and info['epoch'] == 11
    assert info['model'] == models.E1C_SOURCE_MODEL['name']
    assert info['sha256'] == models.E1C_SOURCE_SHA256
    assert info['path'] == str(path)
    assert all(not p.requires_grad for p in source.parameters())
    assert all(not module.training for module in source.modules())


@pytest.mark.parametrize('override', [None, models.E1C_SOURCE_SHA256, '0' * 64])
def test_wrong_digest_and_override_rejected_before_loading(checkpoint, monkeypatch, override):
    path, _ = checkpoint
    monkeypatch.setattr(models, 'file_sha256', lambda path: '1' * 64)
    loader = MagicMock()
    monkeypatch.setattr(models.torch, 'load', loader)
    with pytest.raises(ValueError, match='SHA256'):
        models.load_source(path, source_experiment='e1c', expected_sha256=override)
    loader.assert_not_called()


@pytest.mark.parametrize('mutation,match', [
    ('experiment', 'experiment identity'), ('identity', 'architecture/config'),
    ('epoch', 'epoch'), ('missing', 'Missing key'), ('unexpected', 'Unexpected key'),
    ('shape', 'size mismatch'),
])
def test_invalid_checkpoint_rejected(checkpoint, mutation, match):
    path, state = checkpoint
    if mutation == 'experiment':
        state['config']['experiment_name'] = 'E1-B joint reconstruction'
    elif mutation == 'identity':
        state['config']['model'] = {**state['config']['model'], 'name': 'joint_reconstruction_unet3d'}
    elif mutation == 'epoch':
        state['epoch'] = 13
    elif mutation == 'missing':
        del state['model'][next(iter(state['model']))]
    elif mutation == 'unexpected':
        state['model']['unexpected.weight'] = torch.zeros(1)
    else:
        state['model'][next(iter(state['model']))] = torch.zeros(1)
    torch.save(state, path)
    with pytest.raises((ValueError, RuntimeError), match=match):
        models.load_source(path, source_experiment='e1c')


def test_wrong_requested_experiment(checkpoint):
    path, _ = checkpoint
    with pytest.raises(ValueError, match='Unsupported frozen source experiment'):
        models.load_source(path, source_experiment='unknown')
    with pytest.raises(ValueError, match='epoch|architecture/config'):
        models.load_source(path, source_experiment='e1b', expected_sha256=models.E1C_SOURCE_SHA256)


@pytest.mark.parametrize('latent', ['anatomy', 'disease'])
def test_exact_latent_selection_and_no_decoder_information(latent):
    source = AnatomyConstrainedJointReconstructionUNet3D()
    model = models.FrozenLatentProbe(source, latent).train()
    for decoder in (source.decoder, source.reconstruction_decoder, source.anatomy_decoder):
        decoder.register_forward_pre_hook(lambda *_: pytest.fail('Source decoder was called'))
    image = torch.randn(1, 4, 8, 8, 8)
    with torch.no_grad():
        bottleneck = source.encoder(image)[-1]
        expected = getattr(source.projection, latent + '_projector')(bottleneck)
    torch.testing.assert_close(model.extract_latent(image), expected)
    assert expected.shape == (1, 256, 1, 1, 1)
    assert all(not module.training for module in source.modules())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1742852
    assert all(p.requires_grad for p in model.probe.parameters())
    logits = model(image)
    assert logits.shape == (1, 4, 8, 8, 8)
    logits.mean().backward()
    assert any(p.grad is not None for p in model.probe.parameters())
    assert all(p.grad is None and not p.requires_grad for p in source.parameters())
    next(source.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match='unexpected trainable'):
        model.extract_latent(image)
    source.requires_grad_(False).train()
    with pytest.raises(ValueError, match='eval mode'):
        model.extract_latent(image)


def test_configs_match_e1b_protocol_and_fixed_split():
    a, d = script.validate_pair(*script.CONFIG_PAIRS['e1c'])
    ignored = {'experiment_name', 'latent_source', 'output_dir'}
    assert {k: v for k, v in a.items() if k not in ignored} == {k: v for k, v in d.items() if k not in ignored}
    b = script.read_config('configs/e1b_probe_anatomy.yaml')
    for latent, config in zip(('anatomy', 'disease'), (a, d)):
        assert script.select_config(f'configs/e1c_probe_{latent}.yaml') == config
        assert config['source_experiment'] == 'e1c'
        assert config['source_model_identity'] == models.E1C_SOURCE_MODEL['name']
        assert config['source_checkpoint'] == 'outputs/frozen/e1c_anatomy_constraint/training/best.pt'
        assert config['expected_source_checkpoint_sha256'] == models.E1C_SOURCE_SHA256
        for key in ('epochs', 'batch_size', 'learning_rate', 'weight_decay', 'seed',
                    'num_workers', 'validation_frequency', 'threshold', 'data_config'):
            assert config[key] == b[key]
        assert (config['batch_size'], config['epochs'], config['threshold']) == (4, 30, 0.5)
    data = script.read_config(a['data_config'])
    assert data['train_cases'] == 'splits/train_cases.txt'
    assert data['val_cases'] == 'splits/val_cases.txt'
    assert data['patch_size'] == [96, 96, 96]
    assert len(Path(data['train_cases']).read_text().splitlines()) == 1080
    assert len(Path(data['val_cases']).read_text().splitlines()) == 270
    assert script.select_config('configs/probe_anatomy.yaml')['latent_source'] == 'anatomy'
    assert script.select_config('configs/e1b_probe_disease.yaml')['source_experiment'] == 'e1b'


@pytest.mark.parametrize('key,value,match', [
    ('source_experiment', 'e1b', 'identity'), ('source_experiment', 'unknown', 'experiment'),
    ('source_model_identity', 'joint_reconstruction_unet3d', 'identity'),
    ('expected_source_checkpoint_sha256', '0' * 64, 'SHA256'),
    ('latent_source', 'unsupported', 'latent_source'), ('batch_size', 1, 'batch_size'),
    ('epochs', 1, 'epochs'), ('threshold', 0.4, 'threshold'),
])
def test_invalid_probe_config_rejected(key, value, match):
    config = script.read_config('configs/e1c_probe_anatomy.yaml')
    config[key] = value
    with pytest.raises(ValueError, match=match):
        script.validate_source_config(config)


@pytest.mark.parametrize('kind', ['empty', 'existing_run', 'frozen', 'source_child'])
def test_output_protection_before_loading(tmp_path, monkeypatch, kind):
    config = script.read_config('configs/e1c_probe_anatomy.yaml')
    output = tmp_path / 'run'
    if kind in ('empty', 'existing_run'):
        output.mkdir()
        if kind == 'existing_run':
            (output / 'train.log').write_text('keep')
    elif kind == 'frozen':
        output = Path('outputs/frozen/forbidden-probe')
    else:
        output = Path(config['source_checkpoint']).parent / 'child'
    config['output_dir'] = str(output)
    loader = MagicMock()
    monkeypatch.setattr(script, 'load_source', loader)
    with pytest.raises((ValueError, FileExistsError)):
        script.run(config)
    loader.assert_not_called()
    if kind == 'existing_run':
        assert (output / 'train.log').read_text() == 'keep'


@pytest.fixture
def orchestration(tmp_path, monkeypatch, checkpoint):
    path, _ = checkpoint
    source, source_info = models.load_source(path, source_experiment='e1c')
    config = script.read_config('configs/e1c_probe_anatomy.yaml')
    config.update(source_checkpoint=str(path), output_dir=str(tmp_path / 'run'))
    loader = MagicMock(return_value=(source, source_info))
    monkeypatch.setattr(script, 'load_source', loader)
    train_data, val_data = MagicMock(), MagicMock()
    train_data.records = [{'subject_id': 'train'}]
    val_data.records = [{'subject_id': 'val'}]
    monkeypatch.setattr(script, 'make_dataset', lambda data, split: train_data if split == 'train' else val_data)
    monkeypatch.setattr(script, 'DataLoader', MagicMock())
    train = MagicMock(return_value=0.2)
    monkeypatch.setattr(script, 'train_epoch', train)
    validation = MagicMock(return_value=(0.3, {}))
    monkeypatch.setattr(script, 'validate', validation)
    monkeypatch.setattr(script, 'best_validation_metric', lambda metrics: 0.4)
    save = MagicMock()
    monkeypatch.setattr(script, 'save_probe_checkpoint', save)
    return config, source, source_info, loader, train, validation, save


def test_training_orchestration_and_provenance_without_training(orchestration):
    config, source, info, loader, train, validation, save = orchestration
    script.run(config, device='cpu')
    loader.assert_called_once_with(Path(config['source_checkpoint']), source_experiment='e1c',
                                   expected_sha256=models.E1C_SOURCE_SHA256, expected_epoch=None)
    assert train.call_count == validation.call_count == 30
    assert all(call.args[-1] == 0.5 for call in validation.call_args_list)
    optimizer = train.call_args.args[2]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == 0.001 and optimizer.defaults['weight_decay'] == 0.00001
    source_ids = {id(p) for p in source.parameters()}
    assert not source_ids & {id(p) for group in optimizer.param_groups for p in group['params']}
    output = Path(config['output_dir'])
    metadata = yaml.safe_load((output / 'config.yaml').read_text())
    assert metadata['source_experiment'] == 'e1c'
    assert metadata['source_model_identity'] == models.E1C_SOURCE_MODEL['name']
    assert metadata['source_checkpoint_info'] == info
    assert metadata['probe_parameter_count'] == 1742852
    assert metadata['probe_architecture'] == models.PROBE_ARCHITECTURE
    assert metadata['optimizer'] == 'AdamW'
    for split in ('train', 'val'):
        assert metadata[split + '_split_sha256'] == hashlib.sha256(
            Path(metadata['data'][split + '_cases']).read_bytes()).hexdigest()
    env = metadata['environment']
    assert {'git_commit', 'git_dirty', 'git_status_short', 'python_version', 'pytorch_version',
            'cuda_version', 'cuda_available', 'device_requested'} <= env.keys()
    assert env['device_requested'] == 'cpu'
    events = [json.loads(line) for line in (output / 'run_events.jsonl').read_text().splitlines()]
    assert events[0]['resume_checkpoint'] is None and events[0]['start_epoch'] == 1
    assert events[-1] == dict(event='completed', total_epochs_completed=30)
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['source_checkpoint']['sha256'] == models.E1C_SOURCE_SHA256


@pytest.mark.parametrize('mutation,match', [('trainable', 'unexpected trainable'),
                                         ('mode', 'eval mode'), ('identity', 'identity/SHA256')])
def test_training_rejects_invalid_loaded_source(orchestration, mutation, match):
    config, source, info, _, train, _, _ = orchestration
    if mutation == 'trainable':
        next(source.parameters()).requires_grad_(True)
    elif mutation == 'mode':
        source.train()
    else:
        info['experiment'] = 'e1b'
    with pytest.raises(ValueError, match=match):
        script.run(config, device='cpu')
    train.assert_not_called()
    assert not Path(config['output_dir']).exists()


def test_checkpoint_provenance_and_resume_identity(checkpoint, tmp_path, monkeypatch):
    path, _ = checkpoint
    source, info = models.load_source(path, source_experiment='e1c')
    model = models.FrozenLatentProbe(source, 'disease')
    optimizer = torch.optim.AdamW(model.probe.parameters())
    config = script.read_config('configs/e1c_probe_disease.yaml')
    target = tmp_path / 'probe.pt'
    script.save_probe_checkpoint(target, model, optimizer, 2, 0.3, 2, config, info)
    state = torch.load(target, weights_only=True)
    assert state['source_checkpoint']['experiment'] == 'e1c'
    assert state['source_checkpoint']['sha256'] == models.E1C_SOURCE_SHA256
    assert state['source_checkpoint']['model'] == models.E1C_SOURCE_MODEL['name']
    assert state['latent_source'] == 'disease'
    monkeypatch.setattr(script, 'file_sha256', lambda path: models.E1C_SOURCE_SHA256)
    assert script.load_probe_checkpoint(target, model, optimizer, config, info)['epoch'] == 2
    with pytest.raises(ValueError, match='identity mismatch'):
        script.load_probe_checkpoint(target, model, optimizer, config, {**info, 'experiment': 'e1b'})


@pytest.mark.parametrize('override,match', [({'epochs': 1}, '30 epochs'),
                                         ({'source_sha256': '0' * 64}, 'SHA256')])
def test_training_protocol_overrides_rejected(orchestration, override, match):
    config, _, _, loader, train, _, _ = orchestration
    with pytest.raises(ValueError, match=match):
        script.run(config, device='cpu', **override)
    loader.assert_not_called()
    train.assert_not_called()


def test_interruption_and_resume_provenance(orchestration, monkeypatch):
    config, _, _, _, train, _, save = orchestration
    # Exercise orchestration only: no optimizer step or real validation is run.
    save.side_effect = lambda path, *_: Path(path).write_bytes(b'synthetic probe checkpoint')
    train.side_effect = [0.2, KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        script.run(config, device='cpu')
    output = Path(config['output_dir'])
    assert not (output / 'summary.json').exists()
    train.side_effect = None
    train.return_value = 0.2
    monkeypatch.setattr(script, 'load_probe_checkpoint', MagicMock(return_value={
        'epoch': 1, 'best_metric': 0.4, 'best_epoch': 1}))
    resume = output / 'latest.pt'
    script.run(config, resume=resume, device='cpu')
    events = [json.loads(line) for line in (output / 'run_events.jsonl').read_text().splitlines()]
    assert [event['event'] for event in events] == ['started', 'started', 'completed']
    assert events[1]['start_epoch'] == 2 and events[1]['requested_epochs'] == 30
    assert events[1]['resume_checkpoint'] == str(resume)
    assert events[1]['resume_checkpoint_sha256'] == hashlib.sha256(resume.read_bytes()).hexdigest()
