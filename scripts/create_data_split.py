"""Persist a seeded case-level split stratified by tissue-presence pattern."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import nibabel as nib
import numpy as np
from datasets.brats_dataset import discover_cases
from datasets.label_mapping import KNOWN_LABELS, TISSUE_LABELS, TISSUE_NAMES


def inspect_presence(data_root: str) -> dict[str, tuple[int, ...]]:
    """Read segmentation arrays only; validate labels before recording presence."""
    metadata = {}
    records = discover_cases(data_root)
    for index, record in enumerate(records, 1):
        path = record['paths']['mask']
        volume = nib.load(path)
        if len(volume.shape) != 3:
            raise ValueError(f"{path}: segmentation must be three-dimensional.")
        labels = np.unique(np.asanyarray(volume.dataobj))
        if not np.isin(labels, KNOWN_LABELS).all():
            raise ValueError(f"{path}: invalid segmentation labels {labels.tolist()}")
        metadata[record['subject_id']] = tuple(int(label in labels) for label in TISSUE_LABELS)
        if index % 100 == 0 or index == len(records):
            print(f"Inspected {index}/{len(records)} segmentations", flush=True)
    return metadata


def split_cases(metadata: dict[str, tuple[int, ...]], train_ratio: float = 0.8,
                seed: int = 42) -> tuple[list[str], list[str]]:
    """Allocate by presence stratum, then shuffle within strata with a seed.

    Target size is round-half-up(N * ratio), bounded to nonempty partitions.
    Give each stratum of size >=2 at least one case per partition when jointly
    feasible. Singletons cannot occur in both. Starting from floor allocations,
    adjust the largest quota deficit/excess until the exact target is reached.
    Infeasible rare-stratum bounds are relaxed; ties use sorted pattern order.
    """
    if not 0 < train_ratio < 1 or len(metadata) < 2:
        raise ValueError('Require 0 < train_ratio < 1 and at least two cases.')
    groups = defaultdict(list)
    for case_id, pattern in sorted(metadata.items()):
        if len(pattern) != len(TISSUE_NAMES) or any(v not in (0, 1) for v in pattern):
            raise ValueError(f'{case_id}: presence must be a four-element binary vector.')
        groups[tuple(pattern)].append(case_id)
    keys = sorted(groups)
    target = min(len(metadata) - 1, max(1, int(len(metadata) * train_ratio + 0.5)))
    sizes = {key: len(groups[key]) for key in keys}
    lower = {key: int(sizes[key] >= 2) for key in keys}
    upper = {key: sizes[key] - int(sizes[key] >= 2) for key in keys}
    if sum(lower.values()) > target or sum(upper.values()) < target:
        lower = dict.fromkeys(keys, 0)
        upper = sizes.copy()
    quotas = {key: sizes[key] * train_ratio for key in keys}
    allocation = {key: max(lower[key], min(upper[key], int(quotas[key]))) for key in keys}
    while sum(allocation.values()) < target:
        key = max((k for k in keys if allocation[k] < upper[k]), key=lambda k: quotas[k] - allocation[k])
        allocation[key] += 1
    while sum(allocation.values()) > target:
        key = max((k for k in keys if allocation[k] > lower[k]), key=lambda k: allocation[k] - quotas[k])
        allocation[key] -= 1
    rng = random.Random(seed)
    train, val = [], []
    for key in keys:
        cases = groups[key].copy()
        rng.shuffle(cases)
        train.extend(cases[:allocation[key]])
        val.extend(cases[allocation[key]:])
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(metadata)
    assert len(train) + len(val) == len(metadata)
    return sorted(train), sorted(val)


def summarize(metadata, train, val, train_ratio, seed) -> dict:
    patterns = sorted({''.join(map(str, pattern)) for pattern in metadata.values()})
    def distribution(ids):
        counts = Counter(''.join(map(str, metadata[case])) for case in ids)
        tissues = {}
        for index, name in enumerate(TISSUE_NAMES):
            count = sum(metadata[case][index] for case in ids)
            tissues[name] = {'count': count, 'percentage': 100 * count / len(ids) if ids else 0.0}
        return {'count': len(ids), 'combinations': {key: counts[key] for key in patterns}, 'tissues': tissues}
    return {
        'seed': seed, 'train_ratio': train_ratio, 'actual_train_ratio': len(train) / len(metadata),
        'total_case_count': len(metadata), 'train_count': len(train), 'validation_count': len(val),
        'tissue_order': list(TISSUE_NAMES),
        'method': 'case-level presence-stratified quota allocation; rare strata shared when feasible',
        'full': distribution(sorted(metadata)), 'train': distribution(train), 'validation': distribution(val),
        'case_presence': {case: list(metadata[case]) for case in sorted(metadata)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1:
        parser.error('train-ratio must be between zero and one')
    metadata = inspect_presence(args.data_root)
    train, val = split_cases(metadata, args.train_ratio, args.seed)
    summary = summarize(metadata, train, val, args.train_ratio, args.seed)
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'train_cases.txt').write_text('\n'.join(train) + '\n')
    (output / 'val_cases.txt').write_text('\n'.join(val) + '\n')
    (output / 'split_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(f'Train: {len(train)}; validation: {len(val)}; total: {len(metadata)}')
    print(json.dumps({key: summary[key]['tissues'] for key in ('full', 'train', 'validation')}, indent=2))


if __name__ == '__main__':
    main()
