"""Check conservative augmentation on one real foreground-centered patch."""
import argparse
import torch
from datasets.brats_dataset import BraTSDataset
from datasets.augmentations import TrainingAugmentation
from scripts.train_baseline import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = read_config('configs/data.yaml')
    sample = BraTSDataset(root=args.data_root or config['root'], training=True,
                          foreground_sampling_probability=1.,
                          generator=torch.Generator().manual_seed(args.seed))[0]
    image, target = sample['image'], sample['target']
    # Propagate voxel indices through the same flips to verify background exactly.
    indices = torch.arange(image[0].numel()).reshape(image.shape[1:])
    augment = TrainingAugmentation(config['augmentation'], torch.Generator().manual_seed(args.seed))
    after, labels, flipped_indices = augment(image, target, indices)
    aligned_before = image.flatten(1)[:, flipped_indices.flatten()].reshape_as(image)
    assert after[aligned_before == 0].eq(0).all()
    assert ((labels == 0) | (labels == 1)).all()
    assert torch.equal(labels, target.flatten(1)[:, flipped_indices.flatten()].reshape_as(target))
    print(f"Case: {sample['case_id']}; image/target: {tuple(after.shape)}/{tuple(labels.shape)}")
    for name, tensor in [('before', image), ('after', after)]:
        print(f'{name}: dtype={tensor.dtype} min={tensor.min().item():.6f} max={tensor.max().item():.6f} mean={tensor.mean().item():.6f} std={tensor.std(unbiased=False).item():.6f}')
    print('PASS: targets binary and spatially aligned; original zero background preserved.')


if __name__ == '__main__':
    main()
