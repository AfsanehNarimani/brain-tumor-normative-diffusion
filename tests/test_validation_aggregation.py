"""Empty-target cases cannot inflate presence-aware model selection."""
import pytest
import torch
from evaluation.metrics import ValidationDice, best_validation_metric, region_dice


def tensors(prediction, target):
    p = torch.zeros(1, 4, 1, 1, len(prediction))
    t = torch.zeros_like(p)
    p[0, 0, 0, 0] = torch.tensor(prediction)
    t[0, 0, 0, 0] = torch.tensor(target)
    return p * 40 - 20, t


@pytest.mark.parametrize('p,t,case,pooled,count', [
    ([0], [0], None, None, 0), ([1], [0], None, None, 0),
    ([0], [1], 0., 0., 1), ([1], [1], 1., 1., 1),
])
def test_empty_and_nonempty(p, t, case, pooled, count):
    accumulator = ValidationDice()
    logits, target = tensors(p, t)
    accumulator.update(logits, target)
    result = accumulator.compute()
    assert result['NETC_case_dice'] == case
    assert result['NETC_dataset_dice'] == pooled
    assert result['NETC_positive_cases'] == count
    assert best_validation_metric(result) == case
    if not any(p) and not any(t):
        assert region_dice(logits, target).eq(1).all()


def test_mixed_pooled_counts_and_presence_mean():
    accumulator = ValidationDice()
    for p, t in [([0, 0], [0, 0]), ([1, 1], [0, 0]),
                 ([0, 0], [1, 0]), ([1, 1], [1, 1])]:
        accumulator.update(*tensors(p, t))
    result = accumulator.compute()
    # Two positive cases: Dice 0 and 1. Empty-case false positives affect pooling.
    assert result['NETC_positive_cases'] == 2
    assert result['NETC_case_dice'] == 0.5
    assert result['NETC_dataset_dice'] == pytest.approx(4 / 7)
    assert result['ET_case_dice'] is None
    assert best_validation_metric(result) == 0.5  # NETC, WT, TC eligible.


def test_selection_is_equal_region_mean():
    accumulator = ValidationDice()
    logits, target = tensors([1, 1], [1, 1])
    target[:, 3] = 1  # RC missed, separate from WT/TC.
    accumulator.update(logits, target)
    assert best_validation_metric(accumulator.compute()) == 0.75


def test_unavailable_regions_consistent_across_epochs_and_resume(tmp_path):
    from evaluation.metrics import EVALUATION_REGIONS
    from training.checkpointing import save_checkpoint, load_checkpoint

    model = torch.nn.Conv3d(4, 4, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    target = torch.zeros(1, 4, 1, 1, 2)

    def evaluate_epoch(current_model):
        accumulator = ValidationDice()
        accumulator.update(current_model(target), target)
        metrics = accumulator.compute()
        for region in EVALUATION_REGIONS:
            assert metrics[f'{region}_positive_cases'] == 0
            assert metrics[f'{region}_case_dice'] is None
            assert metrics[f'{region}_dataset_dice'] is None
        assert best_validation_metric(metrics) is None
        return metrics

    # All regions have false positives in epoch one, empty predictions in two.
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(20)
    first = evaluate_epoch(model)
    path = tmp_path / 'latest.pt'
    save_checkpoint(path, model, optimizer, 1, {}, best_metric=None)
    restored = torch.nn.Conv3d(4, 4, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    state = load_checkpoint(path, restored, restored_optimizer)
    assert state['epoch'] == 1
    assert evaluate_epoch(restored) == first
    with torch.no_grad():
        restored.bias.fill_(-20)
    assert evaluate_epoch(restored) == first
