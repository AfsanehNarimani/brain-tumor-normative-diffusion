"""Deterministic covering tiles with averaged logits and CPU reconstruction."""
from itertools import product
import torch
from evaluation.metrics import ValidationDice


@torch.no_grad()
def tiled_logits(model, image, device='cpu', tile_size=(96, 96, 96)):
    """Cover a single full volume; overlap at boundaries is averaged uniformly."""
    if image.ndim != 5 or image.shape[0] != 1:
        raise ValueError('Tiled inference requires [1,C,D,H,W].')
    sizes = tuple(min(n, t) for n, t in zip(image.shape[2:], tile_size))
    if len(sizes) != 3 or min(sizes) < 8:
        raise ValueError('Tile and volume dimensions must be at least 8.')
    starts = []
    for n, t in zip(image.shape[2:], sizes):
        starts.append(sorted(set([*range(0, n - t + 1, t), n - t])))
    output = None
    counts = torch.zeros((1, 1, *image.shape[2:]))
    for origin in product(*starts):
        crop = (slice(None), slice(None), *(slice(s, s + t) for s, t in zip(origin, sizes)))
        logits = model(image[crop].to(device)).float().cpu()
        if output is None:
            output = torch.zeros((1, logits.shape[1], *image.shape[2:]))
        output[crop] += logits
        counts[(slice(None), slice(None), *crop[2:])] += 1
    return output / counts


@torch.no_grad()
def validate(model, loader, criterion, device='cpu', tile_size=(96, 96, 96), threshold=0.5):
    """Compute full-volume loss and subject-averaged Dice after reconstruction."""
    previous = model.training
    model.eval()
    losses, scores = [], ValidationDice()
    try:
        for batch in loader:
            logits = tiled_logits(model, batch['image'], device, tile_size)
            target = batch['target'].cpu()
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite validation loss.')
            losses.append(loss.item())
            scores.update(logits, target, threshold)
    finally:
        model.train(previous)
    if not losses:
        raise ValueError('Validation loader is empty.')
    return sum(losses) / len(losses), scores.compute()
