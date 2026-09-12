"""Alignment, zero preservation and reproducibility of training augmentation."""
import torch
from datasets.augmentations import TrainingAugmentation


def inputs():
    target = torch.zeros(4, 3, 4, 5)
    target[:, 1:, 1:3, 1:4] = 1
    return target * 2, target


def test_disabled():
    image, target = inputs()
    result = TrainingAugmentation({'enabled': False})(image, target)
    assert result[0] is image and result[1] is target


def test_flip_alignment():
    image, target = inputs()
    config = {'enabled': True, 'flip': {'probability': 1},
              **{name: {'enabled': False} for name in ('intensity_scale', 'intensity_shift', 'gaussian_noise')}}
    raw = target[0].long()
    result, labels, mask = TrainingAugmentation(config)(image, target, raw)
    torch.testing.assert_close(result, image.flip((1, 2, 3)))
    torch.testing.assert_close(result / 2, labels)
    torch.testing.assert_close(mask, raw.flip((0, 1, 2)))


def test_seed_and_properties():
    image, target = inputs()
    before = image.clone()
    config = {'enabled': True, 'flip': {'enabled': False}}
    first = TrainingAugmentation(config, torch.Generator().manual_seed(42))(image, target)
    second = TrainingAugmentation(config, torch.Generator().manual_seed(42))(image, target)
    for a, b in zip(first, second):
        assert a.shape == image.shape and a.dtype == torch.float32
        torch.testing.assert_close(a, b)
    assert ((first[1] == 0) | (first[1] == 1)).all()
    assert first[0][image == 0].eq(0).all()
    assert not torch.equal(first[0], image)
    torch.testing.assert_close(first[1], target)
    torch.testing.assert_close(image, before)


def test_intensity_order():
    image, target = inputs()
    config = {'enabled': True, 'flip': {'enabled': False},
              'intensity_scale': {'min': 2., 'max': 2.},
              'intensity_shift': {'min': 1., 'max': 1.},
              'gaussian_noise': {'enabled': False}}
    result, labels = TrainingAugmentation(config)(image, target)
    assert result[image != 0].eq(5).all()
    assert result[image == 0].eq(0).all()
    torch.testing.assert_close(labels, target)
