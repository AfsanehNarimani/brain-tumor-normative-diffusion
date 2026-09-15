"""Full-validation research reports and optional aligned binary NIfTI exports."""
import csv
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from evaluation.tiled import tiled_logits
from evaluation.metrics import ValidationDice, EVALUATION_REGIONS, best_validation_metric
from datasets.label_mapping import TISSUE_NAMES
from training.artifacts import write_json


@torch.no_grad()
def evaluate_artifacts(model, dataset, output, device='cpu', threshold=0.5,
                       tile_size=(96, 96, 96), prediction_cases=()):
    output = Path(output)
    selected = set(prediction_cases)
    records = {r['subject_id']: r for r in dataset.records}
    if selected - records.keys():
        raise ValueError(f'Prediction IDs outside validation split: {sorted(selected - records.keys())}')
    output.mkdir(parents=True, exist_ok=True)
    fields = ['case_id', *(f'{r}_{suffix}' for r in EVALUATION_REGIONS for suffix in ('target_present', 'dice'))]
    total = ValidationDice()
    previous = model.training
    model.eval()
    try:
        with (output / 'per_case_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index in range(len(dataset)):
                sample = dataset[index]
                case = sample['case_id']
                logits = tiled_logits(model, sample['image'].unsqueeze(0), device, tile_size)
                target = sample['target'].unsqueeze(0)
                per_case = ValidationDice()
                per_case.update(logits, target, threshold)
                total.update(logits, target, threshold)
                metrics = per_case.compute()
                row = {'case_id': case}
                for region in EVALUATION_REGIONS:
                    row[f'{region}_target_present'] = metrics[f'{region}_positive_cases']
                    row[f'{region}_dice'] = metrics[f'{region}_case_dice']
                writer.writerow(row)
                if case in selected:
                    folder = output / 'predictions'
                    folder.mkdir(exist_ok=True)
                    reference = nib.load(records[case]['paths']['mask'])
                    prediction = (logits[0].sigmoid() >= threshold).numpy()
                    # Separate binary masks preserve overlaps; no argmax reinterpretation.
                    for name, mask in zip(TISSUE_NAMES, prediction):
                        header = reference.header.copy()
                        header.set_data_dtype(np.uint8)
                        nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header),
                                 folder / f'{case}-{name}.nii.gz')
    finally:
        model.train(previous)
    metrics = total.compute()
    result = {'case_count': len(dataset), 'threshold': threshold,
              'mean_validation_metric': best_validation_metric(metrics),
              'regions': {r: {'positive_case_count': metrics[f'{r}_positive_cases'],
                              'case_dice': metrics[f'{r}_case_dice'],
                              'dataset_dice': metrics[f'{r}_dataset_dice']} for r in EVALUATION_REGIONS}}
    write_json(output / 'aggregate_metrics.json', result)
    return result
