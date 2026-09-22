"""D0-B0: train-only raw-MRI support audit using 20 unfiltered random patches per case."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile

import nibabel as nib
import numpy as np
import torch
import yaml

from datasets.brats_dataset import MODALITIES, load_subject
from datasets.patch_sampling import PatchSampler

EXPECTED_CASES = 1080
TRAIN_SHA256 = '581dc6326a22cb0b9582fa249b8e13fd948597a60c5f454cb5ade0493fa7c18e'
EXPERIMENT_BASE_COMMIT = 'cd2fdb16a159af6626c6cabb2c10739b1a505afd'
PATCHES_PER_CASE = 20
PATCH_SHAPE = (96, 96, 96)
LATENT_SHAPE = (12, 12, 12)
BLOCK_SHAPE = (8, 8, 8)
AUDIT_SEED = 20260917
THRESHOLDS = (0.10, 0.25, 0.50, 0.75, 0.90)
FRACTION_FIELDS = tuple(f'latent_fraction_ge_{value:.2f}'.replace('.', '_') for value in THRESHOLDS)
METRICS = ('input_support_fraction', 'latent_occupancy_mean', *FRACTION_FIELDS)
IDENTITY_FIELDS = ('case_id', 'patch_index', 'seed', 'crop_start_d', 'crop_start_h', 'crop_start_w',
                   'crop_stop_d', 'crop_stop_h', 'crop_stop_w')
ARTIFACTS = ('patch_support_statistics.csv', 'case_support_statistics.csv', 'patch_manifest.csv',
             'support_mask_summary.json', 'analysis_provenance.json')
SUPPORT_DEFINITION = 'Any raw T1n/T1c/T2w/T2f modality nonzero at the voxel, before intensity normalization; no segmentation labels.'
SAMPLING_DEFINITION = '20 random 96^3 patches per training case; existing PatchSampler.sample_slices with foreground_sampling_probability=0; uniform valid starts, original RNG draw order, repeated coordinates allowed, no rejection or augmentation.'
SEED_DEFINITION = 'SHA256 of UTF-8 compact JSON [global_audit_seed,case_id,patch_index], ensure_ascii=False; first 8 digest bytes as unsigned big-endian integer masked to 63 bits; fresh CPU torch.Generator per patch. No Python hash().'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_ids(path):
    path = Path(path).expanduser().resolve()
    if sha256(path) != TRAIN_SHA256:
        raise ValueError('Training split SHA256 mismatch')
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate training case IDs')
    if len(ids) != EXPECTED_CASES:
        raise ValueError(f'Expected exactly {EXPECTED_CASES} training cases')
    if any(Path(case).name != case or case in ('.', '..') for case in ids):
        raise ValueError('Invalid training case ID')
    return ids


def load_raw_image(root, case):
    """Use project shape/spacing/affine/finite checks, visiting this train case only.

    The shared loader also reads/validates the mask file; it is discarded here.
    No label conversion, normalization, foreground construction or model is used.
    """
    folder = Path(root).expanduser() / case
    paths = {key: folder / f'{case}-{key.lower()}.nii.gz' for key in MODALITIES}
    paths['mask'] = folder / f'{case}-seg.nii.gz'
    image, _ = load_subject(dict(subject_id=case, paths=paths))
    return image


def image_support(image):
    if image.ndim != 4 or image.shape[0] != 4 or min(image.shape[1:]) < 96:
        raise ValueError('Expected raw [4,D,H,W] image with spatial dimensions >=96')
    if not torch.isfinite(image).all():
        raise ValueError('Non-finite raw MRI')
    return image.ne(0).any(dim=0)


def latent_occupancy(support):
    if tuple(support.shape) != PATCH_SHAPE or support.dtype != torch.bool:
        raise ValueError('Expected binary boolean support of shape [96,96,96]')
    # Integer sums followed by division by a power of two are exact in float64.
    counts = support.reshape(12, 8, 12, 8, 12, 8).sum(dim=(1, 3, 5))
    return counts.to(torch.float64) / 512


def stable_seed(global_seed, case_id, patch_index):
    payload = json.dumps([global_seed, case_id, patch_index], ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest()[:8], 'big') & ((1 << 63) - 1)


def random_patches(support, case_id, audit_seed):
    if support.ndim != 3 or support.dtype != torch.bool or min(support.shape) < 96:
        raise ValueError('Expected 3D boolean volume support with dimensions >=96')
    # Only geometry reaches the sampler. MRI support and segmentation cannot bias it.
    shape_only = torch.zeros((), dtype=torch.bool).expand(support.shape)
    for index in range(PATCHES_PER_CASE):
        seed = stable_seed(audit_seed, case_id, index)
        sampler = PatchSampler(PATCH_SHAPE, 0.0, torch.Generator().manual_seed(seed))
        slices = sampler.sample_slices(shape_only)
        identity = dict(case_id=case_id, patch_index=index, seed=seed)
        for axis, section in zip(('d', 'h', 'w'), slices):
            identity[f'crop_start_{axis}'], identity[f'crop_stop_{axis}'] = section.start, section.stop
        yield identity, support[slices]


def patch_statistics(support):
    occupancy = latent_occupancy(support).numpy()
    return dict(input_support_fraction=float(support.sum().item() / support.numel()),
                latent_occupancy_mean=float(occupancy.mean()),
                latent_occupancy_std=float(occupancy.std(ddof=0)),
                latent_occupancy_min=float(occupancy.min()), latent_occupancy_max=float(occupancy.max()),
                **{field: float((occupancy >= threshold).mean())
                   for field, threshold in zip(FRACTION_FIELDS, THRESHOLDS)})


def case_statistics(case_id, patches):
    if len(patches) != PATCHES_PER_CASE:
        raise ValueError('Expected exactly 20 patches per case')
    row = dict(case_id=case_id, patch_count=len(patches))
    for field in METRICS[:2]:
        values = np.array([p[field] for p in patches], dtype=np.float64)
        for name, function in (('mean', np.mean), ('median', np.median), ('min', np.min), ('max', np.max)):
            row[f'{field}_{name}'] = float(function(values))
    row.update({f'{field}_mean': float(np.mean([p[field] for p in patches])) for field in FRACTION_FIELDS})
    return row


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(count=len(values), mean=float(values.mean()), std=float(values.std(ddof=0)),
                min=float(values.min()), max=float(values.max()),
                **{f'p{p:02d}': float(np.percentile(values, p, method='linear'))
                   for p in (1, 5, 25, 50, 75, 95, 99)})


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_output(path, overwrite, inputs):
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError('Output must not be a symlink')
    output = path.resolve()
    frozen = (Path(__file__).resolve().parents[1] / 'outputs/frozen').resolve()
    if output == frozen or frozen in output.parents:
        raise ValueError('Output must not be under outputs/frozen')
    if any(output == p or output in p.parents for p in inputs):
        raise ValueError('Output overlaps source inputs')
    if output.exists():
        if not output.is_dir():
            raise ValueError('Output must be a directory')
        files = list(output.iterdir())
        if files and not overwrite:
            raise FileExistsError('Output already exists; use --overwrite explicitly')
        if any(p.name not in ARTIFACTS or p.is_symlink() or not p.is_file() for p in files):
            raise ValueError('Refusing to overwrite unrelated files')
    return output


def run(data_config='configs/data.yaml', output_dir='outputs/d0b_support_mask_audit',
        audit_seed=AUDIT_SEED, overwrite=False):
    config_path = Path(data_config).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError('Expected data YAML mapping')
    split_path = Path(data['train_cases']).expanduser().resolve()
    output = check_output(output_dir, overwrite, (config_path, split_path))
    ids = train_ids(split_path)
    root = Path(data['root']).expanduser().resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('Output must be separate from dataset root')
    input_hashes = dict(data_config_sha256=sha256(config_path), train_split_sha256=sha256(split_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    values = {field: [] for field in METRICS}
    with tempfile.TemporaryDirectory(prefix='.d0b-support-', dir=output.parent) as temporary:
        stage = Path(temporary)
        with (stage / 'patch_support_statistics.csv').open('w', newline='') as patch_stream, \
                (stage / 'case_support_statistics.csv').open('w', newline='') as case_stream, \
                (stage / 'patch_manifest.csv').open('w', newline='') as manifest_stream:
            manifest = csv.DictWriter(manifest_stream, fieldnames=IDENTITY_FIELDS)
            manifest.writeheader()
            patch_writer = case_writer = None
            for case in ids:
                support = image_support(load_raw_image(root, case))
                patches = []
                for identity, patch in random_patches(support, case, audit_seed):
                    metrics = patch_statistics(patch)
                    row = {**identity, **metrics}
                    if patch_writer is None:
                        patch_writer = csv.DictWriter(patch_stream, fieldnames=list(row))
                        patch_writer.writeheader()
                    patch_writer.writerow(row)
                    manifest.writerow(identity)
                    patches.append(metrics)
                    for field in METRICS:
                        values[field].append(metrics[field])
                row = case_statistics(case, patches)
                if case_writer is None:
                    case_writer = csv.DictWriter(case_stream, fieldnames=list(row))
                    case_writer.writeheader()
                case_writer.writerow(row)
        total = len(values['input_support_fraction'])
        if total != len(ids) * PATCHES_PER_CASE:
            raise ValueError('Unexpected total patch count')
        definitions = dict(case_count=len(ids), patches_per_case=PATCHES_PER_CASE, total_patch_count=total,
            patch_shape=list(PATCH_SHAPE), latent_shape=list(LATENT_SHAPE), latent_block_shape=list(BLOCK_SHAPE),
            support_definition=SUPPORT_DEFINITION, sampling_definition=SAMPLING_DEFINITION,
            validation_data_used=False, augmentation_used=False, rejection_sampling_used=False,
            segmentation_used_for_support=False, segmentation_used_for_sampling=False,
            support_threshold_selected=False, candidate_thresholds=list(THRESHOLDS),
            statistical_unit='Biological subjects are training cases; patches are repeated samples within cases.')
        summary = dict(**definitions, empirical_patch_distributions={key: distribution(v) for key, v in values.items()})
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        provenance = dict(**definitions, analysis_identity='D0-B0 latent support-mask audit',
            git_commit_sha=git('rev-parse', 'HEAD'), experiment_base_commit=EXPERIMENT_BASE_COMMIT,
            git_dirty=bool(git('status', '--porcelain')), git_status_short=git('status', '--short').splitlines(),
            timestamp_utc=datetime.now(timezone.utc).isoformat(), train_split_path=str(split_path),
            data_config_path=str(config_path), **input_hashes, dataset_root=str(root), dataset_release=data.get('release'),
            global_audit_seed=audit_seed, seed_derivation=SEED_DEFINITION, expected_total_patches=len(ids) * PATCHES_PER_CASE,
            loader='datasets.brats_dataset.load_subject: raw float32 MRI; shared shape/spacing/affine/finite checks; mask loaded/validated but discarded',
            preprocessing='Raw intensities only; normalization and augmentation not applied',
            occupancy_method='Exact integer reduction of non-overlapping 8x8x8 support blocks divided by 512 in float64',
            std_convention='population, ddof=0', percentile_method='NumPy linear, exact over all patch summary values',
            validation_statement='Validation data were not used.',
            segmentation_statement='Segmentation labels were not used to define support or select patches.',
            rejection_statement='No patch rejection occurred; zero-support patches and repeated coordinates retained.',
            diffusion_statement='This audit does not train or evaluate a diffusion model.',
            model_training=False, optimizer_used=False, backward_pass=False, e1c_forward_used=False,
            device='cpu', python_version=platform.python_version(), numpy_version=np.__version__,
            pytorch_version=str(torch.__version__), nibabel_version=nib.__version__,
            patch_manifest_sha256=sha256(stage / 'patch_manifest.csv'))
        if input_hashes != dict(data_config_sha256=sha256(config_path), train_split_sha256=sha256(split_path)):
            raise ValueError('Input config or split changed during audit')
        write_json(stage / 'support_mask_summary.json', summary)
        write_json(stage / 'analysis_provenance.json', provenance)
        check_output(output, overwrite, (config_path, split_path))
        output.mkdir(exist_ok=True)
        for name in ARTIFACTS:
            (stage / name).replace(output / name)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', default='configs/data.yaml')
    parser.add_argument('--output-dir', default='outputs/d0b_support_mask_audit')
    parser.add_argument('--audit-seed', type=int, default=AUDIT_SEED)
    parser.add_argument('--overwrite', action='store_true')
    print(run(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
