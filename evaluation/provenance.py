"""Read-only local dataset/split provenance audit for Experiment 0.1."""
import csv
import hashlib
import io
from pathlib import Path
import nibabel as nib
import numpy as np
from datasets.label_mapping import TISSUE_NAMES, TISSUE_LABELS, KNOWN_LABELS
from training.artifacts import write_json

MODALITIES = ('t1n', 't1c', 't2w', 't2f', 'seg')
DATASET_NAME = 'BraTS 2024 Adult Glioma Post-Treatment'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def split_identity(path):
    path = Path(path).expanduser().resolve()
    content = path.read_bytes()
    ids = [line.strip() for line in content.decode('utf-8').splitlines() if line.strip()]
    normalized = ('\n'.join(sorted(ids)) + '\n').encode()
    return ids, {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(),
                 'normalized_ids_sha256': hashlib.sha256(normalized).hexdigest()}


def audit_dataset(data_root, train_split, validation_split, output, hash_dataset=False,
                  expected_total=1350, expected_train=1080, expected_validation=270):
    """Inspect every header and segmentation; optional streaming full-file hashes.

    Counts are configurable for synthetic tests, not weakened by smoke limits.
    On failure, persist the audit and manifest before raising. Declared identity
    is not independently authenticated by directory structure or these hashes.
    """
    root, output = Path(data_root).expanduser().resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    audit = dict(dataset_name=DATASET_NAME, declared_dataset_release='v2', data_root=str(root),
                 release_authentication='Declared identity only; no publisher-signed manifest was supplied.',
                 expected_total_case_count=expected_total, expected_train_case_count=expected_train,
                 expected_validation_case_count=expected_validation, expected_modalities=list(MODALITIES),
                 expected_label_values=list(KNOWN_LABELS), missing_files=[], unexpected_files=[],
                 geometry_mismatches=[], segmentation_errors=[], file_errors=[], duplicate_physical_cases=[], audit_passed=False,
                 full_content_hashing=hash_dataset,
                 leakage_audit_scope='Case-ID overlap and physical-directory alias checks only; patient-level independence is not verified.',
                 fingerprint_scope='file-content identity (SHA256 per file)' if hash_dataset else
                                   'structure/metadata identity (relative paths and sizes); NOT byte-level verification')
    identities, split_ids = {}, {}
    for name, path in (('train', train_split), ('validation', validation_split)):
        try:
            ids, identity = split_identity(path)
        except (OSError, ValueError) as error:
            ids, identity = [], {'path': str(path), 'sha256': None, 'normalized_ids_sha256': None}
            errors.append(f'{name} split could not be read: {error}')
        split_ids[name] = ids
        identities[name] = identity
        audit[f'{name}_split_path'] = identity['path']
        audit[f'{name}_split_sha256'] = identity['sha256']
        audit[f'{name}_normalized_ids_sha256'] = identity['normalized_ids_sha256']
    train, val = split_ids['train'], split_ids['validation']
    audit['duplicate_case_ids'] = {name: sorted({case for case in ids if ids.count(case) > 1})
                                   for name, ids in split_ids.items()}
    if any(audit['duplicate_case_ids'].values()):
        errors.append('Duplicate case IDs in split files.')
    if not root.is_dir():
        directories = []
        errors.append(f'Dataset root is not a directory: {root}')
    else:
        directories = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
        audit['unexpected_files'].extend(str(p.relative_to(root)) for p in sorted(root.iterdir()) if p.is_file())
    discovered = {p.name for p in directories}
    intersection = set(train) & set(val)
    union = set(train) | set(val)
    audit.update(discovered_total_case_count=len(discovered), train_case_count=len(train), validation_case_count=len(val),
                 train_validation_intersection_count=len(intersection), train_validation_union_count=len(union),
                 overlapping_case_ids=sorted(intersection), missing_expected_cases=sorted(union - discovered),
                 unexpected_cases=sorted(discovered - union))
    if len(discovered) != expected_total: errors.append('Discovered case count differs from expected total.')
    if len(train) != expected_train or len(val) != expected_validation: errors.append('Frozen split counts differ from expected counts.')
    if intersection: errors.append('Train/validation split overlap exists.')
    if union - discovered: errors.append('Split references unknown cases.')
    if discovered - union: errors.append('Split union does not cover discovered cases.')
    manifest, patterns, shapes, spacings, global_labels = [], set(), set(), set(), set()
    presence = dict.fromkeys(TISSUE_NAMES, 0)
    real_paths = {}
    for case_index, directory in enumerate(directories, 1):
        case = directory.name
        if directory.resolve() in real_paths:
            audit['duplicate_physical_cases'].append([case, real_paths[directory.resolve()]])
            errors.append(f'Duplicate physical case directory: {case} and {real_paths[directory.resolve()]}')
        real_paths[directory.resolve()] = case
        membership = 'both' if case in intersection else ('train' if case in train else ('validation' if case in val else 'unassigned'))
        row = {'case_id': case, 'split': membership}
        expected_names = {f'{case}-{modality}.nii.gz' for modality in MODALITIES}
        audit['unexpected_files'].extend(str(p.relative_to(root)) for p in sorted(directory.iterdir()) if p.name not in expected_names)
        reference = None
        for modality in MODALITIES:
            path = directory / f'{case}-{modality}.nii.gz'
            row[f'{modality}_filename'] = path.name
            row[f'{modality}_relative_path'] = str(path.relative_to(root))
            row[f'{modality}_size_bytes'] = None
            if hash_dataset: row[f'{modality}_sha256'] = None
            if not path.is_file():
                audit['missing_files'].append(str(path.relative_to(root)))
                continue
            patterns.add(f'<case>-{modality}.nii.gz')
            try:
                before = path.stat()
                row[f'{modality}_size_bytes'] = before.st_size
                if hash_dataset: row[f'{modality}_sha256'] = sha256_file(path)
            except OSError as error:
                audit['file_errors'].append({'file': str(path.relative_to(root)), 'error': str(error)})
                continue
            try:
                volume = nib.load(path)
                shape = tuple(int(v) for v in volume.shape)
                spacing = tuple(float(v) for v in volume.header.get_zooms()[:3])
                shapes.add(shape)
                if np.isfinite(spacing).all(): spacings.add(spacing)
                invalid = len(shape) != 3 or any(n <= 0 for n in shape) or not np.isfinite(spacing).all() or any(v <= 0 for v in spacing) or not np.isfinite(volume.affine).all() or abs(np.linalg.det(volume.affine[:3,:3])) < 1e-12
                if reference is not None:
                    invalid = invalid or shape != reference.shape or not np.allclose(volume.affine, reference.affine, atol=1e-5, rtol=0) or not np.allclose(spacing, reference.header.get_zooms()[:3], atol=1e-5, rtol=0)
                if invalid:
                    audit['geometry_mismatches'].append(str(path.relative_to(root)))
                if reference is None: reference = volume
                if modality == 'seg':
                    labels = np.unique(np.asanyarray(volume.dataobj))
                    global_labels.update(float(v) for v in labels if np.isfinite(v))
                    if not np.isin(labels, KNOWN_LABELS).all():
                        audit['segmentation_errors'].append({'case_id': case, 'error': 'unexpected labels',
                                                              'labels': [float(v) if np.isfinite(v) else str(v) for v in labels]})
                    for name, value in zip(TISSUE_NAMES, TISSUE_LABELS):
                        presence[name] += int(value in labels)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    audit['file_errors'].append({'file': str(path.relative_to(root)), 'error': 'File changed during audit'})
            except Exception as error:
                audit['geometry_mismatches'].append(str(path.relative_to(root)))
                audit['segmentation_errors'].append({'case_id': case, 'file': path.name, 'error': str(error)})
        manifest.append(row)
        if case_index % 100 == 0 or case_index == len(directories):
            print(f'Dataset audit: {case_index}/{len(directories)} cases inspected', flush=True)
    if audit['file_errors']: errors.append('File identity reads failed or files changed during audit.')
    if audit['missing_files']: errors.append('Required modality or segmentation files are missing.')
    if audit['geometry_mismatches']: errors.append('Incompatible or unreadable case geometry.')
    if audit['segmentation_errors']: errors.append('Segmentation labels or file reads failed validation.')
    audit.update(observed_filename_patterns=sorted(patterns), observed_global_label_values=[int(v) if v.is_integer() else v for v in sorted(global_labels)],
                 per_label_case_presence=presence, unique_shapes=[list(v) for v in sorted(shapes)],
                 unique_voxel_spacings=[list(v) for v in sorted(spacings)],
                 geometry_mismatch_count=len(set(audit['geometry_mismatches'])))
    fields = ['case_id','split']
    for modality in MODALITIES:
        fields.extend(f'{modality}_{key}' for key in ('filename','relative_path','size_bytes'))
        if hash_dataset: fields.append(f'{modality}_sha256')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(manifest)
    content = buffer.getvalue().encode('utf-8')
    (output / 'dataset_manifest.csv').write_bytes(content)
    audit.update(manifest_sha256=hashlib.sha256(content).hexdigest(), errors=errors, audit_passed=not errors)
    write_json(output / 'dataset_audit.json', audit)
    if errors:
        raise ValueError('Dataset audit failed before inference: ' + ' '.join(errors))
    return audit
