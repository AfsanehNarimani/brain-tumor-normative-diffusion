"""Fast sanity-run checks without BraTS files or a GPU."""
import csv
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts import tiny_overfit_e1b as sanity


class TinyJoint(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, image):
        return SimpleNamespace(segmentation_logits=image * self.weight,
                               reconstruction=image * self.weight)


def fixed_sample():
    return dict(case_id='fixed-case', crop_start=[1, 2, 3],
                image=torch.ones(1, 4, 2, 2, 2), target=torch.zeros(1, 4, 2, 2, 2))


def test_run_updates_parameters_reuses_batch_and_reports_losses(tmp_path, monkeypatch):
    models = []
    def make_model(**kwargs):
        model = TinyJoint(**kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(sanity, 'JointReconstructionUNet3D', make_model)
    monkeypatch.setattr(sanity, 'cache_patch', lambda *_args: fixed_sample())
    original_train = sanity.train_e1b_epoch
    batches, lambdas = [], []
    def spy(model, loader, optimizer, criterion, lambda_rec, device):
        batches.append(loader[0])
        lambdas.append(lambda_rec)
        return original_train(model, loader, optimizer, criterion, lambda_rec, device)
    monkeypatch.setattr(sanity, 'train_e1b_epoch', spy)
    output = tmp_path / 'e1b_tiny_overfit'
    summary = sanity.run('unused', steps=3, lambda_rec=2.0, output_dir=output)
    assert len(batches) == 3 and all(batch is batches[0] for batch in batches)
    assert lambdas == [2.0] * 3
    assert models[0].weight.item() != 0.25
    assert summary['parameter_changed'] and summary['total_loss_decreased']
    assert summary['case_id'] == 'fixed-case' and summary['number_of_steps'] == 3
    assert summary['lambda_rec'] == 2.0
    assert all(key in summary for key in (
        'initial_total_loss', 'final_total_loss', 'initial_segmentation_loss',
        'final_segmentation_loss', 'initial_reconstruction_loss',
        'final_reconstruction_loss', 'seed', 'all_losses_finite', 'all_gradients_finite'))
    with (output / 'history.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['step']) for row in rows] == [0, 1, 2, 3]
    assert json.loads((output / 'summary.json').read_text())['case_id'] == 'fixed-case'


def test_nonfinite_loss_fails_clearly():
    model = TinyJoint()
    batch = fixed_sample()
    batch['image'].fill_(float('nan'))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(RuntimeError, match='Non-finite E1-B sanity loss'):
        sanity.optimize_fixed_sample(model, batch, optimizer, nn.MSELoss(), 1.0, 1, 'cpu')


def test_output_is_separate_and_existing_results_are_protected(tmp_path, monkeypatch):
    for path in ('outputs/e1b_joint_reconstruction', 'outputs/frozen/anything'):
        with pytest.raises(ValueError, match='separate'):
            sanity.run('unused', output_dir=path)
    output = tmp_path / 'e1b_tiny_overfit'
    output.mkdir()
    (output / 'summary.json').write_text('{}')
    with pytest.raises(FileExistsError, match='overwrite'):
        sanity.run('unused', output_dir=output)
