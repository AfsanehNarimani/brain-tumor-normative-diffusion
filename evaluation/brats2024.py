"""Frozen-threshold conversion and overlap audit for BraTS-compatible evaluation."""
from itertools import combinations
from pathlib import Path
import re
import nibabel as nib
import numpy as np
import torch
from datasets.label_mapping import TISSUE_NAMES, TISSUE_LABELS, KNOWN_LABELS

THRESHOLD = 0.5
PAIRS = tuple(combinations(range(4), 2))
PAIR_NAMES = tuple(f'{TISSUE_NAMES[a]}_{TISSUE_NAMES[b]}' for a, b in PAIRS)


def validate_probabilities(probabilities):
    if probabilities.ndim != 4 or probabilities.shape[0] != 4 or any(n == 0 for n in probabilities.shape):
        raise ValueError('Probabilities must have shape [4,D,H,W] with nonempty dimensions.')
    if not torch.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError('Probabilities must be finite values in [0,1].')


def probabilities_to_labels(probabilities):
    """Threshold at 0.5, then highest probability among positive channels.

    Exact ties select the first channel in the fixed output order
    NETC,SNFH,ET,RC (PyTorch argmax first-index rule), not a tissue priority.
    """
    validate_probabilities(probabilities)
    positive = probabilities >= THRESHOLD
    winners = probabilities.masked_fill(~positive, -float('inf')).argmax(0)
    mapping = torch.tensor(TISSUE_LABELS, device=probabilities.device, dtype=torch.uint8)
    return torch.where(positive.any(0), mapping[winners], 0)


def overlap_audit(probabilities, case_id):
    validate_probabilities(probabilities)
    positive = probabilities >= THRESHOLD
    count = positive.sum(0)
    foreground = int((count >= 1).sum())
    overlap = int((count >= 2).sum())
    return dict(case_id=case_id, foreground_voxels=foreground, overlapping_voxels=overlap,
                overlap_fraction=overlap / foreground if foreground else 0.0,
                max_overlap_channels=int(count.max()),
                **{name: int((positive[a] & positive[b]).sum()) for name, (a,b) in zip(PAIR_NAMES, PAIRS)})


def summarize_overlap(rows):
    foreground = sum(r['foreground_voxels'] for r in rows)
    overlap = sum(r['overlapping_voxels'] for r in rows)
    return dict(case_count=len(rows), cases_with_overlap=sum(r['overlapping_voxels'] > 0 for r in rows),
                total_foreground_voxels=foreground, total_overlapping_voxels=overlap,
                overall_overlap_fraction=overlap / foreground if foreground else 0.0,
                maximum_case_overlap_fraction=max((r['overlap_fraction'] for r in rows), default=0.0),
                maximum_overlap_channels_seen=max((r['max_overlap_channels'] for r in rows), default=0),
                pairwise_overlap_counts={name: sum(r[name] for r in rows) for name in PAIR_NAMES},
                zero_foreground_convention='overlap_fraction=0.0')


def validate_labelmap(array, reference):
    if array.ndim != 3 or tuple(array.shape) != reference.shape:
        raise ValueError('Prediction shape must equal the 3D reference segmentation shape.')
    if not np.issubdtype(array.dtype, np.integer) or not np.isin(array, KNOWN_LABELS).all():
        raise ValueError('Prediction must have integer dtype and labels from {0,1,2,3,4}.')
    if not np.isfinite(reference.affine).all() or abs(np.linalg.det(reference.affine[:3,:3])) < 1e-12:
        raise ValueError('Reference affine must be finite and invertible.')


def export_labelmap(labels, reference_path, output, case_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', case_id):
        raise ValueError('Invalid case ID for NIfTI filename.')
    output = Path(output)
    if output.name != f'{case_id}.nii.gz':
        raise ValueError('Prediction filename must be <case_id>.nii.gz.')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite prediction: {output}')
    reference = nib.load(reference_path)
    array = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    validate_labelmap(array, reference)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(array.astype(np.uint8), reference.affine, header)
    for kind in ('qform', 'sform'):
        matrix, code = getattr(reference, f'get_{kind}')(coded=True)
        getattr(image, f'set_{kind}')(matrix, int(code))
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, output)
    loaded = nib.load(output)
    validate_labelmap(np.asanyarray(loaded.dataobj), reference)
    if not np.allclose(loaded.affine, reference.affine, atol=1e-5, rtol=0):
        raise ValueError('Exported prediction affine differs from reference.')
    return output


def validate_exported_prediction(path, reference_path, case_id):
    """Revalidate on-disk content immediately before evaluation."""
    path = Path(path)
    if path.name != f'{case_id}.nii.gz':
        raise ValueError('Prediction filename/case ID mismatch.')
    prediction, reference = nib.load(path), nib.load(reference_path)
    validate_labelmap(np.asanyarray(prediction.dataobj), reference)
    if not np.allclose(prediction.affine, reference.affine, atol=1e-5, rtol=0):
        raise ValueError('Prediction/reference affine mismatch.')
    if not np.allclose(prediction.header.get_zooms(), reference.header.get_zooms(), atol=1e-5, rtol=0):
        raise ValueError('Prediction/reference spacing mismatch.')
    return True
