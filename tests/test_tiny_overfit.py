"""Small synthetic checks for fixed caching and tiny-overfit optimization."""
import pytest
import torch
from scripts import overfit_tiny_batch as tiny
from losses.segmentation_loss import SegmentationLoss


def test_cache_skips_empty_and_loads_once(monkeypatch) -> None:
    calls = []
    class Dataset:
        def __init__(self, **kwargs):
            assert kwargs['training']
            assert kwargs['foreground_sampling_probability'] == 1
            assert kwargs['generator'].initial_seed() == 42
        def __len__(self):
            return 3
        def __getitem__(self, index):
            calls.append(index)
            target = torch.zeros(4, 2, 2, 2)
            if index:
                target[-1] = 1  # RC-only is valid, with NETC/ET absent.
            return dict(image=target.clone(), target=target, case_id=str(index), crop_start=(0, 0, 0))
    monkeypatch.setattr(tiny, 'BraTSDataset', Dataset)
    patches = tiny.cache_patches('unused', 2, 42)
    assert calls == [0, 1, 2]
    assert len(patches) == 2
    model = torch.nn.Conv3d(4, 4, 1)
    criterion = SegmentationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    original = [p['image'].clone() for p in patches]
    initial, _ = tiny.measure(model, patches, criterion, 'cpu')
    for _ in range(10):
        tiny.optimization_step(model, patches, criterion, optimizer, 'cpu')
    final, scores = tiny.measure(model, patches, criterion, 'cpu')
    assert final < initial
    assert torch.isfinite(scores).all()
    assert calls == [0, 1, 2]
    for patch, image in zip(patches, original):
        torch.testing.assert_close(patch['image'], image)


@pytest.mark.parametrize('final,success', [(0.4, True), (0.5, False), (1.1, False)])
def test_success_criterion(final, success):
    reduction, result = tiny.loss_reduction(1., final)
    assert reduction == pytest.approx(100 * (1 - final))
    assert result is success
