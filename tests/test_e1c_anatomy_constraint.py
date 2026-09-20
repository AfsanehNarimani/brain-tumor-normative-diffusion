"""E1-C architecture, masked objective, diagnostic, and protocol contracts."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from losses.segmentation_loss import SegmentationLoss
from models.anatomy_constrained_joint_reconstruction_unet3d import (
    AnatomyConstrainedJointReconstructionUNet3D,
)
from scripts.train_baseline import read_config
from scripts import train_e1c as train_script
from scripts.train_e1c import select_checkpoint
from training.e1c_anatomy import (
    compute_anatomy_constrained_loss, diagnose_loss_magnitudes,
    masked_anatomy_l1, train_e1c_epoch,
)


def test_forward_latent_routing_shapes_and_raw_outputs():
    model = AnatomyConstrainedJointReconstructionUNet3D(base_channels=2)
    joint_inputs, anatomy_inputs = [], []
    model.reconstruction_decoder.register_forward_pre_hook(
        lambda _module, args: joint_inputs.append(args))
    model.anatomy_decoder.register_forward_pre_hook(
        lambda _module, args: anatomy_inputs.append(args))
    image = torch.randn(1, 4, 8, 9, 10)
    output = model(image)
    assert output.segmentation_logits.shape == image.shape
    assert output.reconstruction.shape == output.anatomy_reconstruction.shape == image.shape
    assert output.z_a.shape == output.z_d.shape == (1, 16, 1, 1, 1)
    assert torch.equal(joint_inputs[0][0], torch.cat((output.z_a, output.z_d), dim=1))
    assert torch.equal(anatomy_inputs[0][0], output.z_a)
    assert len(anatomy_inputs[0]) == 2  # latent and output size; no encoder skips
    with torch.no_grad():
        model.anatomy_decoder.head.weight.zero_()
        model.anatomy_decoder.head.bias.fill_(-2)
    assert torch.all(model(image).anatomy_reconstruction == -2)


def test_segmentation_only_bypasses_both_reconstruction_decoders():
    model = AnatomyConstrainedJointReconstructionUNet3D(base_channels=2)
    fail = lambda *_: (_ for _ in ()).throw(AssertionError('reconstruction decoder called'))
    model.reconstruction_decoder.register_forward_pre_hook(fail)
    model.anatomy_decoder.register_forward_pre_hook(fail)
    image = torch.randn(1, 4, 8, 8, 8)
    expected = model.forward_with_latents(image).logits
    torch.testing.assert_close(model.segmentation_only(image), expected)


def test_masked_anatomy_l1_excludes_labels_one_through_four_and_broadcasts():
    image = torch.zeros(1, 4, 1, 1, 5)
    prediction = torch.ones_like(image)
    raw = torch.tensor([[[[0, 1, 2, 3, 4]]]])
    # One valid voxel across four modalities: numerator 4, denominator 1*4.
    assert masked_anatomy_l1(prediction, image, raw).item() == pytest.approx(1.0)
    lesion_changed = prediction.clone()
    lesion_changed[..., 1:] = 1000
    assert torch.equal(masked_anatomy_l1(lesion_changed, image, raw),
                       masked_anatomy_l1(prediction, image, raw))
    valid_changed = prediction.clone()
    valid_changed[..., 0] = 2
    assert masked_anatomy_l1(valid_changed, image, raw).item() == pytest.approx(2.0)


def test_zero_valid_voxels_are_finite_differentiable_zero():
    prediction = torch.randn(1, 4, 2, 2, 2, requires_grad=True)
    loss = masked_anatomy_l1(prediction, torch.zeros_like(prediction),
                             torch.ones(1, 2, 2, 2, dtype=torch.long))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert prediction.grad is not None and torch.all(prediction.grad == 0)


def output_for(image, anatomy=None):
    return SimpleNamespace(segmentation_logits=torch.zeros_like(image),
                           reconstruction=torch.zeros_like(image),
                           anatomy_reconstruction=(torch.zeros_like(image)
                                                   if anatomy is None else anatomy))


def test_joint_formula_lambda_and_shape_validation():
    image = torch.ones(1, 4, 2, 2, 2)
    target = torch.zeros_like(image)
    raw = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    criterion = lambda logits, _target: logits.sum() * 0 + 2
    losses = compute_anatomy_constrained_loss(
        output_for(image), image, target, raw, criterion, 1.0, 0.5)
    assert losses.reconstruction_loss.item() == losses.anatomy_loss.item() == 1
    assert losses.total_loss.item() == pytest.approx(3.5)
    for rec, anat in ((float('nan'), 1), (1, -1), (1, None), (1, 'invalid')):
        with pytest.raises(ValueError, match='lambda'):
            compute_anatomy_constrained_loss(
                output_for(image), image, target, raw, criterion, rec, anat)
    with pytest.raises(ValueError, match='raw_mask'):
        masked_anatomy_l1(torch.zeros_like(image), image, raw[:, :-1])


class TinyE1C(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, image):
        value = image * self.weight
        return SimpleNamespace(segmentation_logits=value, reconstruction=value,
                               anatomy_reconstruction=value)

    def segmentation_only(self, image):
        return image * self.weight


def test_diagnostic_no_update_and_training_reports_all_losses():
    model = TinyE1C()
    batch = {'image': torch.ones(1, 4, 2, 2, 2),
             'target': torch.zeros(1, 4, 2, 2, 2),
             'raw_mask': torch.tensor([[[[0, 1], [2, 3]], [[4, 0], [0, 0]]]])}
    before = model.weight.detach().clone()
    rows = diagnose_loss_magnitudes(model, [batch], SegmentationLoss(), 1.0, 0.5, 1)
    assert torch.equal(model.weight, before)
    assert set(('segmentation_loss', 'reconstruction_loss', 'anatomy_loss',
                'weighted_reconstruction_loss', 'weighted_anatomy_loss', 'total_loss',
                'abnormal_voxel_fraction', 'non_abnormal_voxel_fraction')) <= set(rows[0])
    assert rows[0]['abnormal_voxel_fraction'] == pytest.approx(0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    report = train_e1c_epoch(model, [batch], optimizer, SegmentationLoss(), 1.0, 0.5)
    assert not torch.equal(model.weight, before)
    assert report['weighted_reconstruction_loss'] == pytest.approx(report['reconstruction_loss'])
    assert report['weighted_anatomy_loss'] == pytest.approx(0.5 * report['anatomy_loss'])


def test_diagnostic_artifact_preserves_rows_provenance_and_refuses_overwrite(
        tmp_path, monkeypatch):
    model = TinyE1C()
    before = model.weight.detach().clone()
    rows = [{'batch': 1, 'segmentation_loss': 2.0, 'reconstruction_loss': 1.0,
             'anatomy_loss': 0.75, 'lambda_rec': 1.0, 'lambda_anat': 0.5,
             'weighted_reconstruction_loss': 1.0, 'weighted_anatomy_loss': 0.375,
             'total_loss': 3.375, 'abnormal_voxel_fraction': 0.25,
             'non_abnormal_voxel_fraction': 0.75}]
    monkeypatch.setattr(train_script, 'environment_metadata', lambda _count: {
        'timestamp': '2026-09-20T00:00:00+00:00'})
    sha = 'a' * 40
    monkeypatch.setattr(train_script, '_diagnostic_git_provenance', lambda: {
        'git_commit': sha, 'git_commit_sha': sha, 'git_dirty': True,
        'git_status': [' M configs/data.yaml'],
        'git_status_short': [' M configs/data.yaml'], 'git_provenance_errors': None})
    train = {'seed': 42, 'device': 'cpu', 'data_protocol': {'normalize': True},
             'reconstruction': {'lambda_rec': 1.0},
             'anatomy_reconstruction': {'lambda_anat': 0.5}}
    model_config = {'name': 'anatomy_constrained_joint_reconstruction_unet3d'}
    data = {'root': '/machine/data', 'normalize': True}
    artifact = train_script.write_diagnostic_artifact(
        tmp_path / 'diagnostic', rows, model, model_config, train, data, 1, 1.0, 0.5)
    payload = json.loads(artifact.read_text())
    assert payload['batches'] == rows
    assert payload['lambda_rec'] == 1.0 and payload['lambda_anat'] == 0.5
    assert payload['diagnostic_batch_count'] == 1
    assert payload['model_mode'] == 'train' and payload['optimizer_steps_performed'] is False
    assert payload['git_commit'] == payload['git_commit_sha'] == sha
    assert payload['git_dirty'] is True
    assert payload['git_status'] == [' M configs/data.yaml']
    assert payload['git_status_short'] == [' M configs/data.yaml']
    assert torch.equal(model.weight, before)
    with pytest.raises(FileExistsError, match='already exists'):
        train_script.write_diagnostic_artifact(
            tmp_path / 'diagnostic', rows, model, model_config, train, data, 1, 1.0, 0.5)


@pytest.mark.parametrize(('status', 'dirty'), [('', False), (' M configs/data.yaml', True)])
def test_diagnostic_git_commit_is_independent_of_worktree_status(monkeypatch, status, dirty):
    sha = 'd66fce181d46ff920a5b912a51c9c329475c83fa'
    monkeypatch.setattr(train_script, '_run_git',
                        lambda *args: sha if args[0] == 'rev-parse' else status)
    provenance = train_script._diagnostic_git_provenance()
    assert provenance['git_commit'] == provenance['git_commit_sha'] == sha
    assert provenance['git_dirty'] is dirty
    assert provenance['git_status'] == ([status] if status else [])
    assert provenance['git_status_short'] == ([status] if status else [])
    assert provenance['git_provenance_errors'] is None


def test_diagnostic_git_failure_is_explicit_and_does_not_imply_clean(monkeypatch):
    def git(*args):
        if args[0] == 'rev-parse':
            raise OSError('git unavailable for HEAD')
        return ' M configs/data.yaml'
    monkeypatch.setattr(train_script, '_run_git', git)
    provenance = train_script._diagnostic_git_provenance()
    assert provenance['git_commit'] is provenance['git_commit_sha'] is None
    assert provenance['git_dirty'] is True
    assert provenance['git_status'] == [' M configs/data.yaml']
    assert provenance['git_status_short'] == [' M configs/data.yaml']
    assert 'git unavailable for HEAD' in provenance['git_provenance_errors']['commit']


def test_e1c_config_is_minimum_difference_from_e1b():
    e1b, e1c = read_config('configs/train_e1b.yaml'), read_config('configs/train_e1c.yaml')
    for key in ('epochs', 'batch_size', 'learning_rate', 'weight_decay', 'seed',
                'num_workers', 'device', 'validation_frequency', 'threshold',
                'segmentation_loss', 'data_protocol'):
        assert e1c[key] == e1b[key]
    assert e1c['reconstruction'] == e1b['reconstruction']
    assert e1c['anatomy_reconstruction']['lambda_anat'] == 0.5
    assert read_config('configs/model_e1c.yaml')['name'] == \
        'anatomy_constrained_joint_reconstruction_unet3d'


def test_checkpoint_selection_remains_segmentation_based():
    from evaluation.metrics import EVALUATION_REGIONS
    metrics = {f'{region}_positive_cases': 1 for region in EVALUATION_REGIONS}
    metrics.update({f'{region}_case_dice': 0.6 for region in EVALUATION_REGIONS})
    assert select_checkpoint(metrics, 0.5, 3) == (pytest.approx(0.6), 3, True)
