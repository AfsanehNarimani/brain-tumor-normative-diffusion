"""Experiment 0.1: BraTS-compatible local evaluation, not hidden challenge scoring."""
import argparse
import csv
from pathlib import Path
import time
import torch
from datasets.label_mapping import TISSUE_NAMES, TISSUE_LABELS
from evaluation.brats2024 import probabilities_to_labels, overlap_audit, summarize_overlap, export_labelmap, validate_exported_prediction, THRESHOLD
from evaluation.brats_official import OfficialEvaluator, TERMINOLOGY
from evaluation.tiled import tiled_logits
from evaluation.provenance import audit_dataset, sha256_file, DATASET_NAME
from scripts.train_baseline import make_dataset, build_model
from training.artifacts import environment_metadata, write_json


def select_cases(records, max_cases):
    if max_cases is not None and max_cases < 1:
        raise ValueError('--max-cases must be positive.')
    return records[:] if max_cases is None else records[:max_cases]


def new_output_directory(experiment, max_cases, audit_only=False):
    base = Path(experiment) / 'evaluation' / 'brats2024'
    # Audit and smoke artifacts never occupy the full-run filenames.
    output = base / 'audit' if audit_only else (base if max_cases is None else base / f'smoke_{max_cases}')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not (not audit_only and max_cases is None and all(p.is_dir() and (p.name == 'audit' or p.name.startswith('smoke_')) for p in output.iterdir())):
        raise FileExistsError(f'Refusing to overwrite Experiment 0.1 outputs: {output}')
    return output.resolve()


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run(output_dir, data_root, device='cpu', max_cases=None, hash_dataset=False, audit_only=False):
    started = time.perf_counter()
    if max_cases is not None and max_cases < 1:
        raise ValueError('--max-cases must be positive.')
    experiment = Path(output_dir).expanduser().resolve()
    checkpoint = experiment / 'best.pt'
    output = new_output_directory(experiment, max_cases, audit_only)
    summary = dict(audit_only=audit_only, requested_case_count=min(max_cases, 270) if max_cases else 270,
                   successfully_exported_cases=0, failed_cases=[],
                   prediction_directory=str(output / 'predictions'), overlap_audit_completed=False,
                   official_evaluator_completed=False, official_output_paths=[], inference_duration_seconds=0.,
                   evaluation_duration_seconds=0., total_duration_seconds=0., complete_validation_run=False,
                   dataset_audit_passed=False, evaluator_audit_passed=False, prediction_integrity_passed=False)
    current_case = None
    phase = 'checkpoint'
    phase_started = time.perf_counter()
    try:
        checkpoint_hash = sha256_file(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if sha256_file(checkpoint) != checkpoint_hash:
            raise ValueError('Checkpoint changed while loading; cannot establish prediction provenance.')
        config = state['config']
        if config['train'].get('threshold', THRESHOLD) != THRESHOLD:
            raise ValueError('Frozen checkpoint threshold differs from required 0.5.')
        data = {**config['data'], 'root': data_root}
        phase = 'dataset_audit'
        audit = audit_dataset(data_root, data.get('train_cases', 'splits/train_cases.txt'),
                              data.get('val_cases', 'splits/val_cases.txt'), output, hash_dataset)
        summary['dataset_audit_passed'] = True
        phase = 'evaluator_audit'
        try:
            evaluator = OfficialEvaluator()
        except Exception as error:
            write_json(output / 'evaluator_audit.json', dict(evaluator_repository='https://github.com/BraTS/BraTS_evaluation',
                       evaluator_configuration='gli', audit_passed=False, error=str(error)))
            raise
        write_json(output / 'evaluator_audit.json', evaluator.audit)
        summary['evaluator_audit_passed'] = True
        data['train_cases'] = audit['train_split_path']
        data['val_cases'] = audit['validation_split_path']
        dataset = make_dataset(data, 'val')
        if sha256_file(data['train_cases']) != audit['train_split_sha256'] or sha256_file(data['val_cases']) != audit['validation_split_sha256']:
            raise ValueError('Split files changed after audit.')
        total_cases = len(dataset)
        if total_cases != 270:
            raise ValueError('Validation split changed after dataset audit.')
        dataset.records = select_cases(dataset.records, max_cases)
        summary['requested_case_count'] = len(dataset)
        metadata = environment_metadata(0)
        write_json(output / 'evaluation_config.json', dict(
            audit_only=audit_only, experiment_name='Experiment 0.1: BraTS-compatible local evaluation',
            dataset_name=DATASET_NAME, declared_dataset_release='v2',
            checkpoint_path=str(checkpoint), checkpoint_epoch=state.get('epoch'), checkpoint_sha256=checkpoint_hash,
            train_split_path=audit['train_split_path'], validation_split_path=audit['validation_split_path'],
            train_split_sha256=audit['train_split_sha256'], validation_split_sha256=audit['validation_split_sha256'],
            dataset_manifest_sha256=audit['manifest_sha256'], dataset_fingerprint_scope=audit['fingerprint_scope'],
            requested_case_count=len(dataset), full_validation_case_count=total_cases, smoke_run=max_cases is not None and not audit_only,
            threshold=THRESHOLD, model_output_type='raw logits followed by independent sigmoid',
            raw_channel_order=list(TISSUE_NAMES), BraTS_label_mapping=dict(zip(TISSUE_NAMES,TISSUE_LABELS)),
            overlap_resolution_policy='argmax restricted to probability >= 0.5; no positives -> background',
            tie_resolution_policy='first channel index in NETC,SNFH,ET,RC for exact equal maximum probabilities',
            official_evaluator_package=evaluator.metadata['evaluator_package'],
            official_evaluator_version=evaluator.metadata['evaluator_version'], official_evaluator_configuration='gli',
            **evaluator.metadata, git_commit=metadata['git_commit'], git_dirty_status=metadata['git_dirty'],
            timestamp=metadata['timestamp']))
        if audit_only:
            # The finally block writes the successful audit summary before returning.
            return summary
        torch.manual_seed(config['train'].get('seed',42))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = build_model(config['model']).to(device)
        model.load_state_dict(state['model'])
        model.eval()
        overlap_rows, exported, integrity = [], [], []
        phase = 'inference'
        phase_started = time.perf_counter()
        for index, record in enumerate(dataset.records):
            current_case = record['subject_id']
            sample = dataset[index]
            if sample['case_id'] != current_case:
                raise ValueError('Loaded case ID differs from split record.')
            probabilities = tiled_logits(model, sample['image'].unsqueeze(0), device,
                                         tuple(data.get('patch_size',(96,96,96))))[0].sigmoid()
            overlap_rows.append(overlap_audit(probabilities, current_case))
            prediction = export_labelmap(probabilities_to_labels(probabilities), record['paths']['mask'],
                                         output / 'predictions' / f'{current_case}.nii.gz', current_case)
            validate_exported_prediction(prediction, record['paths']['mask'], current_case)
            exported.append((current_case, prediction, record['paths']['mask']))
            integrity.append(dict(case_id=current_case, prediction_sha256=sha256_file(prediction), integrity_passed=True))
            summary['successfully_exported_cases'] += 1
            save_csv(output / 'overlap_per_case.csv', overlap_rows)
            write_json(output / 'overlap_summary.json', summarize_overlap(overlap_rows))
            write_json(output / 'prediction_integrity.json', integrity)
        summary['inference_duration_seconds'] = time.perf_counter() - phase_started
        summary['overlap_audit_completed'] = True
        phase = 'evaluation'
        phase_started = time.perf_counter()
        # Validate all exports before evaluating any; reject intervening changes.
        for (current_case, prediction, reference), identity in zip(exported, integrity):
            validate_exported_prediction(prediction, reference, current_case)
            if sha256_file(prediction) != identity['prediction_sha256']:
                raise ValueError(f'Prediction changed after export: {current_case}')
        summary['prediction_integrity_passed'] = True
        metrics = []
        for current_case, prediction, reference in exported:
            metrics.extend(dict(case_id=current_case, **row) for row in evaluator.evaluate(prediction, reference, output))
        save_csv(output / 'official_metrics.csv', metrics)
        write_json(output / 'official_evaluation_summary.json', dict(
            terminology=TERMINOLOGY,
            case_count=len(exported), complete_validation_run=max_cases is None,
            **evaluator.metadata, metrics=evaluator.raw_results,
            results='Official gli per-case region metrics; nonfinite values explicitly serialized without substitution'))
        summary['evaluation_duration_seconds'] = time.perf_counter() - phase_started
        summary['official_evaluator_completed'] = True
        summary['complete_validation_run'] = max_cases is None and len(exported) == 270
        summary['official_output_paths'] = [str(output / name) for name in ('official_metrics.csv','official_evaluation_summary.json')]
    except Exception as error:
        summary['failed_cases'].append(dict(case_id=current_case, stage=phase, error=str(error)))
        if phase in ('inference','evaluation'):
            summary[f'{phase}_duration_seconds'] = time.perf_counter() - phase_started
        raise
    finally:
        summary['total_duration_seconds'] = time.perf_counter() - started
        write_json(output / 'run_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/baseline_unet3d_full')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-cases', type=int)
    parser.add_argument('--hash-dataset', action='store_true', help='SHA256 every dataset file; full audit still runs without this flag')
    parser.add_argument('--audit-only', action='store_true', help='Audit full dataset, split, checkpoint and evaluator; stop before model construction or inference')
    args = parser.parse_args()
    try:
        print(run(args.output_dir, args.data_root, args.device, args.max_cases, args.hash_dataset, audit_only=args.audit_only))
    except Exception as error:
        parser.exit(1, f'BraTS-compatible local evaluation failed: {error}\n')


if __name__ == '__main__':
    main()
