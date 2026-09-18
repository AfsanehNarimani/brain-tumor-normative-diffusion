"""E1-B architecture, exact joint loss, diagnostics, and selection contracts."""
import csv
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from evaluation.metrics import EVALUATION_REGIONS
from losses.segmentation_loss import SegmentationLoss
from models.joint_reconstruction_unet3d import JointReconstructionUNet3D
from scripts.train_baseline import read_config
from scripts import train_e1b
from scripts.train_e1b import SegmentationView, select_checkpoint
from training.e1b_joint import compute_joint_loss, diagnose_loss_magnitudes, train_e1b_epoch


def test_joint_forward_reuses_e1a_skips_and_only_latents_reconstruct():
    model = JointReconstructionUNet3D(base_channels=2)
    image = torch.randn(1, 4, 9, 10, 11)
    encoder_features, segmentation_inputs, reconstruction_inputs = [], [], []
    model.encoder.register_forward_hook(lambda _m, _a, result: encoder_features.append(result))
    model.decoder.register_forward_pre_hook(lambda _m, args: segmentation_inputs.append(args[0]))
    model.reconstruction_decoder.register_forward_pre_hook(lambda _m, args: reconstruction_inputs.append(args))
    output = model(image)
    assert output.segmentation_logits.shape == output.reconstruction.shape == image.shape
    assert output.z_a.shape == output.z_d.shape == (1, 16, 1, 1, 1)
    assert len(encoder_features) == len(segmentation_inputs) == len(reconstruction_inputs) == 1
    assert all(left is right for left, right in zip(encoder_features[0][:3], segmentation_inputs[0][:3]))
    latents, output_size = reconstruction_inputs[0]
    assert output_size == image.shape[2:]
    assert latents.shape[1] == 32
    assert torch.equal(latents, torch.cat((output.z_a, output.z_d), dim=1))
    # ReconstructionDecoder3D receives a tensor and a size, never encoder skips.
    assert len(reconstruction_inputs[0]) == 2


def test_standard_96_cube_geometry_uses_meta_tensors():
    with torch.device('meta'):
        model = JointReconstructionUNet3D()
        output = model(torch.empty(1, 4, 96, 96, 96))
    assert output.z_a.shape == output.z_d.shape == (1, 256, 12, 12, 12)
    assert output.reconstruction.shape == output.segmentation_logits.shape == (1, 4, 96, 96, 96)
    assert [(block[0].in_channels, block[0].out_channels)
            for block in model.reconstruction_decoder.blocks] == [(512, 128), (128, 64), (64, 32)]


def test_raw_segmentation_logits_and_signed_reconstruction():
    model = JointReconstructionUNet3D(base_channels=2)
    with torch.no_grad():
        model.decoder.head.weight.zero_()
        model.decoder.head.bias.fill_(2)
        model.reconstruction_decoder.head.weight.zero_()
        model.reconstruction_decoder.head.bias.fill_(-2)
    image = torch.zeros(1, 4, 8, 8, 8)
    output = model(image)
    assert torch.all(output.segmentation_logits == 2)
    assert torch.all(output.reconstruction == -2)
    assert torch.equal(SegmentationView(model)(image), output.segmentation_logits)


def test_reconstruction_loss_reaches_both_projections_without_skips():
    model = JointReconstructionUNet3D(base_channels=2)
    image = torch.randn(1, 4, 8, 8, 8)
    target = torch.zeros_like(image)
    losses = compute_joint_loss(model(image), image, target, SegmentationLoss(), 0.5)
    losses.total_loss.backward()
    for projector in (model.projection.anatomy_projector, model.projection.disease_projector):
        assert projector.weight.grad is not None
        assert projector.weight.grad.abs().sum() > 0


def test_l1_total_loss_and_zero_lambda_use_same_input_tensor():
    image = torch.tensor([[[[[0., -2.]]]]])
    reconstruction = torch.tensor([[[[[2., 0.]]]]])
    output = SimpleNamespace(segmentation_logits=torch.tensor(3.), reconstruction=reconstruction)
    criterion = lambda logits, target: logits + target
    target = torch.tensor(1.)
    losses = compute_joint_loss(output, image, target, criterion, 0.25)
    assert losses.segmentation_loss.item() == 4
    assert losses.reconstruction_loss.item() == 2
    assert losses.total_loss.item() == 4.5
    zero = compute_joint_loss(output, image, target, criterion, 0)
    assert torch.equal(zero.total_loss, zero.segmentation_loss)
    with pytest.raises(ValueError, match='same shape'):
        compute_joint_loss(SimpleNamespace(segmentation_logits=output.segmentation_logits,
                                           reconstruction=torch.zeros(1)), image, target, criterion, 1)


class TinyJoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, image):
        return SimpleNamespace(segmentation_logits=image * self.weight,
                               reconstruction=image * self.weight)

    def segmentation_only(self, image):
        return image * self.weight


def test_epoch_reports_components_and_diagnostic_does_not_update():
    model = TinyJoint()
    batch = {'image': torch.ones(1, 4, 8, 8, 8), 'target': torch.zeros(1, 4, 8, 8, 8)}
    loader = [batch]
    criterion = SegmentationLoss()
    before = model.weight.detach().clone()
    rows = diagnose_loss_magnitudes(model, loader, criterion, 0.5, batches=1)
    assert torch.equal(model.weight, before)
    assert rows[0]['reconstruction_loss'] == pytest.approx(0.75)
    assert rows[0]['weighted_reconstruction_loss'] == pytest.approx(0.375)
    assert rows[0]['total_loss'] == pytest.approx(rows[0]['segmentation_loss'] + 0.375)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    report = train_e1b_epoch(model, loader, optimizer, criterion, 0.5)
    assert set(report) == {'total_loss', 'segmentation_loss', 'reconstruction_loss', 'lambda_rec'}
    assert report['lambda_rec'] == 0.5
    assert report['total_loss'] == pytest.approx(report['segmentation_loss'] + 0.5 * report['reconstruction_loss'])
    assert not torch.equal(model.weight, before)


def test_checkpoint_selection_uses_presence_aware_segmentation_metric():
    metrics = {f'{region}_positive_cases': 1 for region in EVALUATION_REGIONS}
    metrics.update({f'{region}_case_dice': 0.4 for region in EVALUATION_REGIONS})
    selected, epoch, improved = select_checkpoint(metrics, 0.3, 2)
    assert selected == pytest.approx(0.4) and epoch == 2 and improved
    assert select_checkpoint(metrics, 0.5, 2) == (0.5, None, False)
    metrics.update({f'{region}_positive_cases': 0 for region in EVALUATION_REGIONS})
    assert select_checkpoint(metrics, None, 2) == (None, None, False)


def test_e1b_config_keeps_e1a_training_conventions_and_unselected_lambda():
    e1a = read_config('configs/train_e1a.yaml')
    e1b = read_config('configs/train_e1b.yaml')
    for key in ('epochs', 'batch_size', 'learning_rate', 'weight_decay', 'seed',
                'num_workers', 'device', 'validation_frequency', 'threshold'):
        assert e1b[key] == e1a[key]
    assert e1b['reconstruction'] == {'enabled': True, 'loss_type': 'L1', 'lambda_rec': None}
    assert e1b['segmentation_loss'] == 'Dice+BCE'
    data = read_config('configs/data.yaml')
    assert all(data[key] == value for key, value in e1b['data_protocol'].items())
    assert read_config('configs/model_e1b.yaml')['latent_channels'] == 256


def test_diagnostic_entry_point_uses_training_image_and_writes_no_artifacts(tmp_path, monkeypatch):
    class TinyDataset:
        def __init__(self, case_id):
            self.records = [{'subject_id': case_id}]

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {'image': torch.ones(4, 8, 8, 8), 'target': torch.zeros(4, 8, 8, 8)}

    monkeypatch.setattr(train_e1b, 'read_case_ids', lambda path: list(range(1080 if path == 'train' else 270)))
    monkeypatch.setattr(train_e1b, 'make_dataset', lambda _data, split: TinyDataset(split))
    monkeypatch.setattr(train_e1b, 'JointReconstructionUNet3D', lambda **_kwargs: TinyJoint())
    output = tmp_path / 'diagnostic'
    train = dict(reconstruction={'enabled': True, 'loss_type': 'L1', 'lambda_rec': None},
                 segmentation_loss='Dice+BCE', epochs=1, batch_size=1,
                 validation_frequency=1, num_workers=0, output_dir=str(output),
                 seed=42, device='cpu')
    rows = train_e1b.run_training(train, {'name': 'joint_reconstruction_unet3d'},
                                  {'train_cases': 'train', 'val_cases': 'val'},
                                  lambda_rec=0.5, diagnostic_batches=1)
    assert len(rows) == 1 and rows[0]['reconstruction_loss'] == pytest.approx(0.75)
    assert rows[0]['total_loss'] == pytest.approx(
        rows[0]['segmentation_loss'] + rows[0]['weighted_reconstruction_loss'])
    assert not output.exists()


def test_synthetic_epoch_writes_separate_losses_and_segmentation_selected_checkpoint(tmp_path, monkeypatch):
    class TinyDataset:
        def __init__(self, case_id):
            self.records = [{'subject_id': case_id}]

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {'image': torch.ones(4, 8, 8, 8), 'target': torch.ones(4, 8, 8, 8)}

    monkeypatch.setattr(train_e1b, 'read_case_ids', lambda path: list(range(1080 if path == 'train' else 270)))
    monkeypatch.setattr(train_e1b, 'make_dataset', lambda _data, split: TinyDataset(split))
    monkeypatch.setattr(train_e1b, 'JointReconstructionUNet3D', lambda **_kwargs: TinyJoint())
    output = tmp_path / 'e1b'
    train = dict(reconstruction={'enabled': True, 'loss_type': 'L1', 'lambda_rec': 0.5},
                 segmentation_loss='Dice+BCE', epochs=1, batch_size=1,
                 validation_frequency=1, num_workers=0, output_dir=str(output),
                 seed=42, device='cpu', learning_rate=0.001, weight_decay=0.00001,
                 threshold=0.5)
    assert train_e1b.run_training(train, {'name': 'joint_reconstruction_unet3d'},
                                  {'train_cases': 'train', 'val_cases': 'val', 'patch_size': [8, 8, 8]}) == output
    with (output / 'history.csv').open(newline='') as stream:
        row = list(csv.DictReader(stream))[0]
    assert float(row['total_loss']) == pytest.approx(
        float(row['segmentation_loss']) + 0.5 * float(row['reconstruction_loss']))
    assert float(row['lambda_rec']) == 0.5
    state = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    assert state['best_epoch'] == 1 and state['best_metric'] is not None
    assert state['config']['selection_metric'] == train_e1b.SELECTION_METRIC
    resumed = {**train, 'epochs': 2}
    train_e1b.run_training(resumed, {'name': 'joint_reconstruction_unet3d'},
                            {'train_cases': 'train', 'val_cases': 'val', 'patch_size': [8, 8, 8]},
                            resume=output / 'latest.pt')
    with (output / 'history.csv').open(newline='') as stream:
        assert [row['epoch'] for row in csv.DictReader(stream)] == ['1', '2']
