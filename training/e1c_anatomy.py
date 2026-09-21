"""E1-C masked anatomy/background reconstruction objective and reporting."""
from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AnatomyConstrainedLoss:
    total_loss: torch.Tensor
    segmentation_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    anatomy_loss: torch.Tensor


def masked_anatomy_l1(anatomy_reconstruction, image, raw_mask):
    if image.ndim != 5 or anatomy_reconstruction.shape != image.shape:
        raise ValueError('Anatomy reconstruction and image must have matching [B,C,D,H,W] shapes')
    if raw_mask.shape != (image.shape[0], *image.shape[2:]):
        raise ValueError('raw_mask must have shape [B,D,H,W] aligned with image')
    if not torch.isfinite(anatomy_reconstruction).all() or not torch.isfinite(image).all():
        raise ValueError('Anatomy reconstruction and image must be finite')
    if not torch.all((raw_mask >= 0) & (raw_mask <= 4)):
        raise ValueError('raw_mask labels must be in {0,1,2,3,4}')
    valid = (raw_mask == 0).unsqueeze(1)
    absolute_error = (anatomy_reconstruction - image).abs()
    valid_voxels = valid.sum()
    denominator = valid_voxels.clamp_min(1) * image.shape[1]
    loss = (absolute_error * valid).sum() / denominator
    return loss * (valid_voxels > 0).to(loss.dtype)


def compute_anatomy_constrained_loss(output, image, target, raw_mask,
                                     segmentation_criterion, lambda_rec, lambda_anat):
    for name, value in (('lambda_rec', lambda_rec), ('lambda_anat', lambda_anat)):
        if isinstance(value, bool) or not isinstance(value, Real) \
                or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if image.ndim != 5 or target.shape != image.shape:
        raise ValueError('image and segmentation target must have matching [B,C,D,H,W] shapes')
    if output.segmentation_logits.shape != target.shape:
        raise ValueError('Segmentation logits must match target shape')
    if output.reconstruction.shape != image.shape:
        raise ValueError('Joint reconstruction must match image shape')
    segmentation = segmentation_criterion(output.segmentation_logits, target)
    reconstruction = F.l1_loss(output.reconstruction, image)
    anatomy = masked_anatomy_l1(output.anatomy_reconstruction, image, raw_mask)
    total = segmentation + lambda_rec * reconstruction + lambda_anat * anatomy
    return AnatomyConstrainedLoss(total, segmentation, reconstruction, anatomy)


def train_e1c_epoch(model: nn.Module, loader, optimizer, segmentation_criterion,
                    lambda_rec, lambda_anat, device='cpu'):
    model.train()
    names = ('total_loss', 'segmentation_loss', 'reconstruction_loss', 'anatomy_loss')
    sums = {name: 0.0 for name in names}
    count = 0
    for batch in loader:
        image, target = batch['image'].to(device), batch['target'].to(device)
        raw_mask = batch['raw_mask'].to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = compute_anatomy_constrained_loss(
            model(image), image, target, raw_mask, segmentation_criterion,
            lambda_rec, lambda_anat)
        if not all(torch.isfinite(getattr(losses, name)) for name in names):
            raise RuntimeError('Non-finite E1-C training loss')
        losses.total_loss.backward()
        optimizer.step()
        for name in names:
            sums[name] += getattr(losses, name).item() * image.shape[0]
        count += image.shape[0]
    if not count:
        raise ValueError('Training loader is empty')
    result = {name: value / count for name, value in sums.items()}
    result.update(lambda_rec=lambda_rec, lambda_anat=lambda_anat,
                  weighted_reconstruction_loss=lambda_rec * result['reconstruction_loss'],
                  weighted_anatomy_loss=lambda_anat * result['anatomy_loss'])
    return result


@torch.no_grad()
def diagnose_loss_magnitudes(model, loader, segmentation_criterion, lambda_rec,
                             lambda_anat, batches, device='cpu'):
    if batches < 1:
        raise ValueError('Diagnostic batch count must be positive')
    model.train()
    rows = []
    for number, batch in enumerate(loader, start=1):
        image, target = batch['image'].to(device), batch['target'].to(device)
        raw_mask = batch['raw_mask'].to(device)
        losses = compute_anatomy_constrained_loss(
            model(image), image, target, raw_mask, segmentation_criterion,
            lambda_rec, lambda_anat)
        if not all(torch.isfinite(value) for value in (
                losses.total_loss, losses.segmentation_loss,
                losses.reconstruction_loss, losses.anatomy_loss)):
            raise RuntimeError('Non-finite E1-C diagnostic loss')
        abnormal = (raw_mask != 0).float().mean().item()
        rows.append(dict(batch=number,
                         segmentation_loss=losses.segmentation_loss.item(),
                         reconstruction_loss=losses.reconstruction_loss.item(),
                         anatomy_loss=losses.anatomy_loss.item(),
                         lambda_rec=lambda_rec, lambda_anat=lambda_anat,
                         weighted_reconstruction_loss=(lambda_rec * losses.reconstruction_loss).item(),
                         weighted_anatomy_loss=(lambda_anat * losses.anatomy_loss).item(),
                         total_loss=losses.total_loss.item(),
                         abnormal_voxel_fraction=abnormal,
                         non_abnormal_voxel_fraction=1.0 - abnormal))
        if number == batches:
            break
    if len(rows) != batches:
        raise ValueError(f'Only {len(rows)} diagnostic batches available; requested {batches}')
    return rows
