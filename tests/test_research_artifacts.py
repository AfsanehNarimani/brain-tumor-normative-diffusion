"""Research metadata and presence-aware prediction/report artifacts."""
import csv
import json
import nibabel as nib
import numpy as np
import pytest
import torch
import yaml
from training.artifacts import snapshot, environment_metadata, write_json
from evaluation.artifacts import evaluate_artifacts


def test_snapshot_and_environment(tmp_path, monkeypatch):
    config = dict(data={'patch_size': [96]*3}, model={'out_channels': 4}, train={'epochs': 1, 'seed': 42})
    snapshot(tmp_path, config, 123)
    assert yaml.safe_load((tmp_path / 'config.yaml').read_text()) == config
    resumed = {**config, 'train': {**config['train'], 'epochs': 2}}
    snapshot(tmp_path, resumed, 123, 1)
    assert (tmp_path / 'config.before_epoch_2.yaml').exists()
    with pytest.raises(ValueError, match='Incompatible'):
        snapshot(tmp_path, {**resumed, 'model': {'out_channels': 3}}, 123, 2)
    monkeypatch.setattr('training.artifacts.subprocess.check_output', lambda *a, **k: (_ for _ in ()).throw(OSError()))
    metadata = environment_metadata(123)
    assert metadata['git_commit'] is None and metadata['git_dirty'] is None
    assert metadata['trainable_parameter_count'] == 123
    write_json(tmp_path / 'summary.json', {'best_validation_metric': None, 'epoch': 2})
    assert json.loads((tmp_path / 'summary.json').read_text())['best_validation_metric'] is None
    with pytest.raises(ValueError):
        write_json(tmp_path / 'invalid.json', {'loss': float('nan')})


@pytest.mark.parametrize('save', [False, True])
def test_reports_and_optional_predictions(tmp_path, save):
    path = tmp_path / 'seg.nii.gz'
    affine = np.diag([2., 3., 4., 1.])
    nib.save(nib.Nifti1Image(np.zeros((8,8,8), dtype=np.uint8), affine), path)
    class Dataset:
        records = [{'subject_id': 'case-a', 'paths': {'mask': path}},
                   {'subject_id': 'case-b', 'paths': {'mask': path}}]
        def __len__(self): return 2
        def __getitem__(self, i):
            target = torch.zeros(4,8,8,8)
            if i == 0: target[1] = 1
            return {'case_id': self.records[i]['subject_id'], 'image': target * 40 - 20, 'target': target}
    output = tmp_path / 'evaluation'
    metrics = evaluate_artifacts(torch.nn.Identity(), Dataset(), output, tile_size=(8,8,8),
                                 prediction_cases=['case-a'] if save else [])
    rows = list(csv.DictReader((output / 'per_case_metrics.csv').open()))
    assert [r['case_id'] for r in rows] == ['case-a', 'case-b']
    assert rows[1]['SNFH_dice'] == '' and rows[1]['SNFH_target_present'] == '0'
    assert metrics['regions']['NETC']['dataset_dice'] is None
    assert metrics['regions']['SNFH']['case_dice'] == 1
    assert metrics['regions']['SNFH']['positive_case_count'] == 1
    assert metrics['mean_validation_metric'] == 1
    assert (output / 'predictions').exists() is save
    if save:
        files = list((output / 'predictions').glob('*.nii.gz'))
        assert len(files) == 4
        volume = nib.load(output / 'predictions/case-a-SNFH.nii.gz')
        np.testing.assert_array_equal(volume.affine, affine)
        assert volume.shape == (8,8,8)
        assert np.asarray(volume.dataobj).min() == 1
