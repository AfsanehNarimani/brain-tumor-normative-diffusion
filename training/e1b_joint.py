"""E1-B joint objective and subject-weighted epoch reporting."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class JointLoss:
    total_loss: torch.Tensor
    segmentation_loss: torch.Tensor
    reconstruction_loss: torch.Tensor


def compute_joint_loss(output, image, target, segmentation_criterion, lambda_rec: float) -> JointLoss:
    if not math.isfinite(lambda_rec) or lambda_rec < 0:
        raise ValueError('lambda_rec must be finite and nonnegative')
    if output.reconstruction.shape != image.shape:
        raise ValueError('Reconstruction must have the same shape as the normalized input')
    segmentation_loss = segmentation_criterion(output.segmentation_logits, target)
    reconstruction_loss = F.l1_loss(output.reconstruction, image)
    total_loss = segmentation_loss + lambda_rec * reconstruction_loss
    return JointLoss(total_loss, segmentation_loss, reconstruction_loss)


def train_e1b_epoch(model: nn.Module, loader, optimizer, segmentation_criterion,
                    lambda_rec: float, device='cpu') -> dict:
    model.train()
    sums = dict(total_loss=0.0, segmentation_loss=0.0, reconstruction_loss=0.0)
    count = 0
    for batch in loader:
        image, target = batch['image'].to(device), batch['target'].to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = compute_joint_loss(model(image), image, target, segmentation_criterion, lambda_rec)
        if not all(torch.isfinite(value) for value in
                   (losses.total_loss, losses.segmentation_loss, losses.reconstruction_loss)):
            raise RuntimeError('Non-finite E1-B training loss')
        losses.total_loss.backward()
        optimizer.step()
        for name in sums:
            sums[name] += getattr(losses, name).item() * image.shape[0]
        count += image.shape[0]
    if not count:
        raise ValueError('Training loader is empty')
    return {**{name: value / count for name, value in sums.items()}, 'lambda_rec': lambda_rec}


@torch.no_grad()
def diagnose_loss_magnitudes(model, loader, segmentation_criterion, lambda_rec: float,
                             batches: int, device='cpu') -> list[dict]:
    """Forward-only diagnostic; no optimizer steps or research artifacts."""
    if batches < 1:
        raise ValueError('Diagnostic batch count must be positive')
    model.train()
    rows = []
    for number, batch in enumerate(loader, start=1):
        image, target = batch['image'].to(device), batch['target'].to(device)
        losses = compute_joint_loss(model(image), image, target, segmentation_criterion, lambda_rec)
        if not all(torch.isfinite(value) for value in
                   (losses.total_loss, losses.segmentation_loss, losses.reconstruction_loss)):
            raise RuntimeError('Non-finite E1-B diagnostic loss')
        rows.append(dict(batch=number, segmentation_loss=losses.segmentation_loss.item(),
                         reconstruction_loss=losses.reconstruction_loss.item(),
                         lambda_rec=lambda_rec,
                         weighted_reconstruction_loss=(lambda_rec * losses.reconstruction_loss).item(),
                         total_loss=losses.total_loss.item()))
        if number == batches:
            break
    if len(rows) != batches:
        raise ValueError(f'Only {len(rows)} diagnostic batches available; requested {batches}')
    return rows
