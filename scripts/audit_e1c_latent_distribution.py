"""D0-A train-only E1-C Z_a audit; ten deterministic training-distribution patches per case."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np
import torch

from analysis.latent_distribution_audit import (audit_latents, PERCENTILE_METHOD,
    PATCHES_PER_CASE, FOREGROUND_PATCHES, patch_type)
from datasets.brats_dataset import BraTSDataset, MODALITIES
from datasets.label_mapping import TISSUE_LABELS
from datasets.patch_sampling import PatchSampler
from models.latent_probe import load_source, file_sha256, E1C_SOURCE_SHA256
from scripts.train_baseline import read_config, read_case_ids
from training.artifacts import write_json

TRAIN_SHA256 = '581dc6326a22cb0b9582fa249b8e13fd948597a60c5f454cb5ade0493fa7c18e'
EXPECTED_CASES = 1080
LATENT_SHAPE = (256, 12, 12, 12)
AUDIT_SEED = 20260917
SEED_DERIVATION = ('SHA256 of UTF-8 compact JSON array [global_audit_seed,case_id,patch_index,patch_type], '
                   'ensure_ascii=False; first 8 digest bytes interpreted big-endian, masked to 63 bits; '
                   'fresh CPU torch.Generator.manual_seed for each patch. No Python hash().')
MANIFEST_FIELDS = ('case_id', 'patch_index_within_case', 'patch_type', 'seed',
                   'start_d', 'start_h', 'start_w', 'stop_d', 'stop_h', 'stop_w',
                   'center_d', 'center_h', 'center_w', 'foreground_fallback')
DEFAULT_CHECKPOINT = 'outputs/frozen/e1c_anatomy_constraint/training/best.pt'
ARTIFACTS = ('latent_global_statistics.json', 'latent_channel_statistics.csv',
             'latent_case_statistics.csv', 'latent_patch_statistics.csv', 'patch_manifest.csv',
             'latent_distribution_summary.json', 'analysis_provenance.json')


def validate_train_split(path):
    path = Path(path).expanduser().resolve()
    if file_sha256(path) != TRAIN_SHA256:
        raise ValueError('Training split SHA256 mismatch')
    ids = read_case_ids(path)
    if len(ids) != EXPECTED_CASES:
        raise ValueError(f'Expected {EXPECTED_CASES} unique training cases; got {len(ids)}')
    return ids


def make_train_dataset(data, case_ids):
    """Select train IDs before file inspection; reuse the existing full-volume loader.

    A temporary manifest avoids discover_cases inspecting non-training directories.
    Paths follow the same layout as the established make_dataset/discover_cases path.
    No validation split is read, and no non-training volume is opened.
    """
    if not data.get('normalize', True) or tuple(data.get('patch_size', ())) != (96, 96, 96):
        raise ValueError('D0-A requires existing full-volume normalization and 96^3 patch geometry')
    if data.get('foreground_sampling_probability', 0.7) != 0.7:
        raise ValueError('D0-A requires the established 0.7 foreground sampling policy')
    with tempfile.TemporaryDirectory(prefix='d0-train-manifest-') as directory:
        manifest = Path(directory) / 'train.csv'
        with manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['subject_id', *MODALITIES, 'mask'])
            writer.writeheader()
            for case in case_ids:
                if Path(case).name != case or case in ('.', '..'):
                    raise ValueError('Invalid training case ID')
                paths = {key: f'{case}/{case}-{key.lower()}.nii.gz' for key in MODALITIES}
                writer.writerow(dict(subject_id=case, **paths, mask=f'{case}/{case}-seg.nii.gz'))
        dataset = BraTSDataset(manifest=manifest, root=data['root'], normalize=True, training=False)
    if [r['subject_id'] for r in dataset.records] != case_ids:
        raise ValueError('Training dataset identity mismatch')
    return dataset


def require_frozen(source):
    if any(p.requires_grad for p in source.parameters()):
        raise ValueError('Source must remain frozen')
    if any(module.training for module in source.modules()):
        raise ValueError('Source must remain in eval mode')


@torch.no_grad()
def extract_anatomy(source, image):
    require_frozen(source)
    if image.shape != (1, 4, 96, 96, 96) or not torch.isfinite(image).all():
        raise ValueError('Expected finite [1,4,96,96,96] input')
    # Same encoder/projector-only path as FrozenLatentProbe.extract_latent.
    bottleneck = source.encoder(image)[-1]
    latent = source.projection.anatomy_projector(bottleneck)
    require_frozen(source)
    if tuple(latent.shape) != (1, *LATENT_SHAPE):
        raise ValueError(f'Unexpected latent shape: {tuple(latent.shape)}')
    if not torch.isfinite(latent).all():
        raise ValueError('Non-finite latent')
    if latent.dtype != torch.float32:
        raise ValueError('Expected float32 source latent')
    return latent[0].cpu().numpy()


def patch_seed(global_seed, case_id, index, kind):
    payload = json.dumps([global_seed, case_id, index, kind], ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest()[:8], 'big') & ((1 << 63) - 1)


def patch_plan(case_ids, global_seed):
    for case in case_ids:
        for index in range(PATCHES_PER_CASE):
            kind = patch_type(index)
            yield dict(case_id=case, patch_index_within_case=index, patch_type=kind,
                       seed=patch_seed(global_seed, case, index, kind))


def training_foreground(raw_mask):
    # Exact BraTSDataset.__getitem__ union of raw tissue labels, including RC.
    foreground = torch.zeros_like(raw_mask, dtype=torch.bool)
    for label in TISSUE_LABELS:
        foreground |= raw_mask == label
    return foreground


class RecordedPatchSampler(PatchSampler):
    """Observe randint results only; inherit sample_slices unchanged from training."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draws = []

    def _randint(self, low, high):
        value = super()._randint(low, high)
        self.draws.append(value)
        return value


def select_patch(foreground, metadata, patch_size=(96, 96, 96)):
    is_foreground = metadata['patch_type'] == 'foreground'
    sampler = RecordedPatchSampler(patch_size, 1.0 if is_foreground else 0.0,
                                   torch.Generator().manual_seed(metadata['seed']))
    slices = sampler.sample_slices(foreground)
    fallback = is_foreground and not bool(foreground.any())
    center = ('', '', '')
    if is_foreground and not fallback:
        # Recover the recorded selected voxel without another random draw or a
        # second full-volume nonzero coordinate array. Original nonzero order is
        # lexicographic; plane cumulative counts locate that same selected index.
        selected = sampler.draws[0]
        cumulative = foreground.sum(dim=(1, 2)).cumsum(0)
        d = int(torch.searchsorted(cumulative, selected + 1))
        previous = int(cumulative[d - 1]) if d else 0
        h, w = foreground[d].nonzero()[selected - previous].tolist()
        center = (d, h, w)
    row = dict(metadata, foreground_fallback=fallback)
    for axis, section, value in zip(('d', 'h', 'w'), slices, center):
        row[f'start_{axis}'], row[f'stop_{axis}'] = section.start, section.stop
        row[f'center_{axis}'] = value
    return slices, row


def samples(source, dataset, case_ids, device, audit_seed, manifest):
    with Path(manifest).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for index, case in enumerate(case_ids):
            sample = dataset[index]  # load -> normalize full volume -> target construction
            if sample['case_id'] != case:
                raise ValueError('Training sample identity mismatch')
            foreground = training_foreground(sample['raw_mask'])
            if not bool(foreground.any()):
                raise ValueError(f'{case}: empty foreground cannot supply exactly seven foreground-centered patches; training sampler would fall back to random')
            for metadata in patch_plan([case], audit_seed):
                slices, row = select_patch(foreground, metadata)
                patch = sample['image'][(slice(None), *slices)].contiguous()
                latent = extract_anatomy(source, patch.unsqueeze(0).to(device))
                writer.writerow(row)
                yield metadata, latent


def check_output(output, overwrite, checkpoint, split):
    requested = Path(output).expanduser()
    if requested.is_symlink():
        raise ValueError('Output must not be a symlink')
    output = requested.resolve()
    frozen = (Path(__file__).resolve().parents[1] / 'outputs' / 'frozen').resolve()
    if output == frozen or frozen in output.parents:
        raise ValueError('Output must not be under outputs/frozen')
    for path in (checkpoint, split):
        if output == path.parent or output in path.parents or path.parent in output.parents:
            raise ValueError('Output must be separate from source inputs')
    if output.exists():
        if not output.is_dir():
            raise ValueError('Output must be a directory')
        entries = list(output.iterdir())
        if entries and not overwrite:
            raise FileExistsError('Non-empty audit directory; use --overwrite explicitly')
        if any(p.name not in ARTIFACTS or not p.is_file() or p.is_symlink() for p in entries):
            raise ValueError('Refusing to overwrite unrelated files or symlinks')
    return output


def run(data_config='configs/data.yaml', checkpoint=DEFAULT_CHECKPOINT,
        output_dir='outputs/d0_latent_distribution_audit', device='cpu', overwrite=False, audit_seed=AUDIT_SEED):
    data_path, checkpoint = Path(data_config).expanduser().resolve(), Path(checkpoint).expanduser().resolve()
    data = read_config(data_path)
    split = Path(data['train_cases']).expanduser().resolve()
    output = check_output(output_dir, overwrite, checkpoint, split)
    ids = validate_train_split(split)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, info = load_source(checkpoint, source_experiment='e1c',
                               expected_sha256=E1C_SOURCE_SHA256, expected_epoch=11)
    source.to(device)
    require_frozen(source)
    dataset = make_train_dataset(data, ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch_bytes = len(ids) * PATCHES_PER_CASE * int(np.prod(LATENT_SHAPE)) * 4
    if shutil.disk_usage(output.parent).free < scratch_bytes + 64 * 1024**2:
        raise OSError(f'Insufficient scratch disk; need at least {scratch_bytes} bytes plus output headroom')
    with tempfile.TemporaryDirectory(prefix='.d0-audit-', dir=output.parent) as staging:
        stage = Path(staging)
        global_stats, summary = audit_latents(
            samples(source, dataset, ids, device, audit_seed, stage / 'patch_manifest.csv'),
            ids, stage, LATENT_SHAPE)
        require_frozen(source)
        with (stage / 'patch_manifest.csv').open(newline='') as stream:
            fallback_count = sum(row['foreground_fallback'] == 'True' for row in csv.DictReader(stream))
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        provenance = dict(
            analysis_name='D0-A E1-C Z_a latent distribution audit',
            purpose='Pre-diffusion train-split-only audit of E1-C anatomy/background latent scale and distribution.',
            source_model_identity=info['model'], source_checkpoint_path=info['path'],
            source_checkpoint_sha256=info['sha256'], source_epoch=info['epoch'],
            latent_source='anatomy / Z_a',
            latent_extraction='torch.no_grad(): source.projection.anatomy_projector(source.encoder(image)[-1]); no decoders',
            expected_latent_shape=list(LATENT_SHAPE), train_split_path=str(split), train_split_sha256=TRAIN_SHA256,
            case_count=len(ids), validation_data_used=False, validation_statement='validation data were not used',
            statistics_convention='population standard deviation, ddof=0', accumulation_dtype='float64',
            latent_dtype='float32', percentile_method=PERCENTILE_METHOD, scratch_bytes=scratch_bytes,
            scratch_gib=scratch_bytes / 1024**3, mapped_channel_bytes=scratch_bytes // LATENT_SHAPE[0],
            full_scratch_file_mapped=False, output_size_estimate='approximately 8-15 MiB of text; data-dependent; 64 MiB reserved',
            memory_design='Statistical heap bounded by ten float32 patches plus float64 temporaries (conservative <256 MiB); one channel mapped at a time. Excludes existing full-volume dataset tensors, sampler foreground coordinates, source model, and inference allocations; OS file cache is reclaimable.',
            data_config_path=str(data_path), data_config_sha256=file_sha256(data_path),
            data_root=str(Path(data['root']).expanduser().resolve()),
            preprocessing='Existing BraTSDataset full-volume normalize_nonzero; training=False; no augmentation',
            sampling_purpose='deterministic audit realization of the established training patch distribution',
            patches_per_case=PATCHES_PER_CASE, foreground_patches_per_case=FOREGROUND_PATCHES,
            random_patches_per_case=PATCHES_PER_CASE - FOREGROUND_PATCHES,
            total_patch_count=len(ids) * PATCHES_PER_CASE, audit_sampling_seed=audit_seed,
            deterministic_seed_derivation=SEED_DERIVATION,
            patch_manifest_artifact='patch_manifest.csv', patch_manifest_sha256=file_sha256(stage / 'patch_manifest.csv'),
            foreground_definition='BraTSDataset raw_mask union of TISSUE_LABELS (1,2,3,4), including RC',
            crop_semantics='Inherited PatchSampler.sample_slices; forced branch probability 1 for indices 0-6, 0 for 7-9; original branch RNG draw retained; uniform foreground voxel, original jitter and boundary clipping, no padding/resizing',
            empty_foreground_policy='Adapter preserves existing random fallback; audit rejects empty foreground to enforce seven genuinely foreground-centered patches',
            foreground_fallback_count=fallback_count,
            scope='1080 biological cases; ten repeated patch samples per case are not independent subjects. Cropping distribution before training augmentation; no disentanglement or pathology localization test',
            device=str(device), python_version=platform.python_version(), pytorch_version=str(torch.__version__),
            numpy_version=np.__version__, nibabel_version=nib.__version__, cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == 'cuda' else None,
            deterministic_algorithms=True, cudnn_benchmark=False, tf32=False,
            cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
            git_commit_sha=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')),
            git_status_short=git('status', '--short').splitlines(), timestamp_utc=datetime.now(timezone.utc).isoformat(),
            model_optimization_occurred=False, diffusion_model_trained=False, normalization_decision_made=False)
        write_json(stage / 'latent_global_statistics.json', global_stats)
        write_json(stage / 'latent_distribution_summary.json', summary)
        write_json(stage / 'analysis_provenance.json', provenance)
        # Recheck protection just before publishing successfully computed artifacts.
        check_output(output, overwrite, checkpoint, split)
        output.mkdir(exist_ok=True)
        for name in ARTIFACTS:
            (stage / name).replace(output / name)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', default='configs/data.yaml')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output-dir', default='outputs/d0_latent_distribution_audit')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--audit-seed', type=int, default=AUDIT_SEED)
    print(run(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
