"""Full debug path on synthetic NIfTI data and deterministic reconstruction."""
import csv
import numpy as np
import nibabel as nib
import pytest
import torch
from scripts.train_baseline import read_case_ids, run_training, build_model
from evaluation.tiled import tiled_logits
from training.checkpointing import load_checkpoint


def test_split_ids(tmp_path):
    path = tmp_path / 'ids.txt'
    path.write_text('a\nb\n')
    assert read_case_ids(path) == ['a', 'b']
    path.write_text('a\na\n')
    with pytest.raises(ValueError, match='unique'):
        read_case_ids(path)


def test_tiling_covers_odd_shape():
    image = torch.randn(1, 4, 13, 17, 19)
    torch.testing.assert_close(tiled_logits(torch.nn.Identity(), image, tile_size=(8, 8, 8)), image)


@pytest.mark.parametrize("debug", [True, False])
def test_debug_full_path(tmp_path, debug):
    root = tmp_path / 'data'
    root.mkdir()
    for case in ('a', 'b'):
        folder = root / case
        folder.mkdir()
        for suffix in ('t1n', 't1c', 't2w', 't2f', 'seg'):
            array = np.ones((8, 8, 8), dtype=np.float32)
            nib.save(nib.Nifti1Image(array, np.eye(4)), folder / f'{case}-{suffix}.nii.gz')
    train, val = tmp_path / 'train.txt', tmp_path / 'val.txt'
    train.write_text('a\n')
    val.write_text('b\n')
    data = dict(root=str(root), train_cases=str(train), val_cases=str(val), patch_size=[8]*3)
    run = dict(seed=42, epochs=2, batch_size=1, num_workers=0, device='cpu',
               learning_rate=0.001, weight_decay=1e-5, validation_frequency=1,
               output_dir=str(tmp_path / 'output'))
    run_training(data, dict(base_channels=2), run, debug=debug)
    rows = list(csv.DictReader((tmp_path / 'output/history.csv').open()))
    assert len(rows) == (1 if debug else 2)
    for row in rows:
        assert float(row["epoch_duration_seconds"]) >= float(row["training_duration_seconds"]) + float(row["validation_duration_seconds"])
    assert all(np.isfinite(float(value)) for value in rows[0].values() if value != "")
    assert rows[0]["ET_case_dice"] == ""
    assert rows[0]["ET_positive_cases"] == "0"
    model = build_model(dict(base_channels=2))
    optimizer = torch.optim.AdamW(model.parameters())
    state = load_checkpoint(tmp_path / 'output/latest.pt', model, optimizer)
    assert state['epoch'] == (1 if debug else 2) and state['seed'] == 42
    assert 0 <= state['best_metric'] <= 1
    assert state['config']['debug'] is debug
    assert (tmp_path / 'output/best.pt').exists()
    if not debug:
        # The optimizer moments and step count must continue from epoch two.
        old_steps = [float(v['step']) for v in optimizer.state.values()]
        assert old_steps and all(step == 2 for step in old_steps)
        original_history = (tmp_path / 'output/history.csv').read_text()
        latest = tmp_path / 'output/latest.pt'
        state['best_metric'] = 1.0
        state['best_epoch'] = 1
        torch.save(state, latest)
        best_bytes = (tmp_path / 'output/best.pt').read_bytes()
        resumed_run = {**run, 'epochs': 3, 'resume': str(latest)}
        run_training(data, dict(base_channels=2), resumed_run)
        assert (tmp_path / 'output/history.csv').read_text().startswith(original_history)
        resumed = load_checkpoint(latest, model, optimizer)
        assert resumed['epoch'] == 3
        assert resumed['best_metric'] == 1.0 and resumed['best_epoch'] == 1
        assert all(float(v['step']) == 3 for v in optimizer.state.values())
        assert (tmp_path / 'output/best.pt').read_bytes() == best_bytes
        rows = list(csv.DictReader((tmp_path / 'output/history.csv').open()))
        assert [row['epoch'] for row in rows] == ['1', '2', '3']
        history_bytes = (tmp_path / 'output/history.csv').read_bytes()
        run_training(data, dict(base_channels=2), resumed_run)
        assert (tmp_path / 'output/history.csv').read_bytes() == history_bytes
        with pytest.raises(ValueError, match='architecture'):
            run_training(data, dict(base_channels=4), {**resumed_run, 'epochs': 4})
        with pytest.raises(ValueError, match='patch_size'):
            run_training({**data, 'patch_size': [16]*3}, dict(base_channels=2), {**resumed_run, 'epochs': 4})



def test_case_limits():
    from types import SimpleNamespace
    from scripts.train_baseline import limit_cases
    dataset = SimpleNamespace(records=["z", "b", "a"])
    limit_cases(dataset, 2)
    assert dataset.records == ["z", "b"]
    limit_cases(dataset, None)
    assert dataset.records == ["z", "b"]
    with pytest.raises(ValueError):
        limit_cases(dataset, 0)


def test_cli_overrides(monkeypatch):
    from scripts import train_baseline as script
    import sys
    captured = {}
    monkeypatch.setattr(sys, 'argv', ['train_baseline', '--epochs', '2', '--max-train-cases', '20', '--max-val-cases', '5'])
    monkeypatch.setattr(script, 'run_training', lambda data, model, run, debug: captured.update(run=run, debug=debug))
    script.main()
    assert captured['run']['epochs'] == 2
    assert captured['run']['max_train_cases'] == 20
    assert captured['run']['max_val_cases'] == 5
    assert not captured['debug']
