"""D0-B3 train-only pathology-aware input corruption operator audit."""

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
from torch.nn import functional as F

from analysis.rf_geometry import derive_encoder_geometry
from datasets.transforms import normalize_nonzero
from models.latent_probe import E1C_SOURCE_SHA256, file_sha256, load_source
from scripts.audit_d0b1_abnormal_occupancy import (
    BLOCK_SHAPE, DEFAULT_MANIFEST, EXPECTED_CASES, MANIFEST_FIELDS,
    MANIFEST_SHA256, PATCH_SHAPE, PATCHES_PER_CASE, RAW_ABNORMAL_LABELS,
    SUPPORT_THRESHOLD, TRAIN_SHA256, abnormal_mask, block_occupancy, load_case,
    read_manifest, support_mask,
)
from scripts.audit_d0b2_rf_aware_exclusion import rf_contaminated
from scripts.train_baseline import read_case_ids, read_config
from models.encoder import Encoder

EXPERIMENT_BASE_COMMIT = '3434345919ec47d18336cf1040b3dee59deacaea'
DEFAULT_OUTPUT = 'outputs/d0b3_corruption_operator_audit'
DEFAULT_CHECKPOINT = 'outputs/frozen/e1c_anatomy_constraint/training/best.pt'
DEFAULT_CONTRACT = 'outputs/frozen/d0_latent_distribution_audit/d0_za_normalization_contract.json'
DEFAULT_RF_GEOMETRY = 'outputs/frozen/d0b2_rf_aware_exclusion_audit/rf_geometry.json'
DEFAULT_RF_SUMMARY = 'outputs/frozen/d0b2_rf_aware_exclusion_audit/rf_exclusion_summary.json'
EXPECTED_CONTRACT_SHA256 = 'bca5c0f8f1617d73fc4d0850fe5b6f45e4b2bcbacbb4018f1d5d825de49ad792'
EXPECTED_RF_GEOMETRY_SHA256 = '23bb5ff94c83b634e46ae4fe96449a8d5de2ea078764234d767c7964e5c83922'
EXPECTED_RF_SUMMARY_SHA256 = 'b1b0039802c5099e5406323c6c1048db5f15d57c14db6ff022350e6aa6cf79c8'
SUBSET_SEED = 20260917
SUBSET_CASE_COUNT = 200
TOTAL_PATCHES = SUBSET_CASE_COUNT * PATCHES_PER_CASE
LATENT_SHAPE = (12, 12, 12)
LATENT_CHANNELS = 256
DONOR_RADIUS = 8
OPERATORS = ('zero_fill', 'local_mean_fill', 'local_empirical_resample')
MODALITY_NAMES = ('T1n', 'T1c', 'T2w', 'T2f')
EXPECTED_BRANCH = 'experiment/d0b3-corruption-operator-audit'
OUTPUTS = ('selected_patch_manifest.csv', 'patch_corruption_statistics.csv',
           'case_corruption_statistics.csv', 'corruption_operator_summary.json',
           'analysis_provenance.json')


def sha256(path):
    return file_sha256(path)


def git_value(*args):
    return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()


def runtime_git_provenance():
    return dict(runtime_git_branch=git_value('rev-parse', '--abbrev-ref', 'HEAD'),
                runtime_git_commit=git_value('rev-parse', 'HEAD'),
                git_status_short=git_value('status', '--short').splitlines())


def select_cases(case_ids, seed=SUBSET_SEED, count=SUBSET_CASE_COUNT):
    if len(case_ids) != len(set(case_ids)):
        raise ValueError('Case IDs must be unique')
    ranked = sorted(((hashlib.sha256(f'{seed}:{case}'.encode()).hexdigest(), case)
                     for case in case_ids), key=lambda item: (item[0], item[1]))
    if count > len(ranked):
        raise ValueError('Requested subset exceeds training cases')
    return [case for _, case in ranked[:count]]


def select_manifest(rows, selected_cases):
    selected = set(selected_cases)
    selected_rows = [row for row in rows if row['case_id'] in selected]
    if len(selected_rows) != len(selected_cases) * PATCHES_PER_CASE:
        raise ValueError('Selected cases do not have exactly 20 frozen patches each')
    if any(sum(row['case_id'] == case for row in selected_rows) != PATCHES_PER_CASE
           for case in selected_cases):
        raise ValueError('Selected manifest patch counts are invalid')
    if len({(row['case_id'], row['patch_index']) for row in selected_rows}) != len(selected_rows):
        raise ValueError('Selected manifest contains duplicate patch identities')
    return selected_rows


def selection_rows(case_ids, seed=SUBSET_SEED):
    ranked = sorted(((hashlib.sha256(f'{seed}:{case}'.encode()).hexdigest(), case)
                     for case in case_ids), key=lambda item: (item[0], item[1]))
    return {case: (rank, digest) for rank, (digest, case) in enumerate(ranked)}


def donor_shell(abnormal, radius=DONOR_RADIUS):
    abnormal = torch.as_tensor(abnormal, dtype=torch.bool)
    if tuple(abnormal.shape) != PATCH_SHAPE:
        raise ValueError(f'Expected abnormal patch {PATCH_SHAPE}')
    if radius < 0:
        raise ValueError('Donor radius must be nonnegative')
    kernel = 2 * radius + 1
    volume = abnormal[None, None].float()
    for kernel_size, padding in (((kernel, 1, 1), (radius, 0, 0)),
                                 ((1, kernel, 1), (0, radius, 0)),
                                 ((1, 1, kernel), (0, 0, radius))):
        volume = F.max_pool3d(volume, kernel_size=kernel_size, stride=1, padding=padding)
    dilated = volume[0, 0].bool()
    return dilated & ~abnormal


def donor_pools(raw_image, abnormal, support, radius=DONOR_RADIUS):
    raw_image = torch.as_tensor(raw_image)
    support = torch.as_tensor(support, dtype=torch.bool)
    abnormal = torch.as_tensor(abnormal, dtype=torch.bool)
    if raw_image.shape != (4, *PATCH_SHAPE) or support.shape != PATCH_SHAPE:
        raise ValueError('Unexpected raw image/support shape')
    shell = donor_shell(abnormal, radius)
    if not bool(abnormal.any()):
        empty = torch.zeros_like(abnormal)
        return [empty.clone() for _ in range(4)], [False] * 4
    pools = []
    fallbacks = []
    for modality in range(4):
        local = shell & support & ~abnormal & raw_image[modality].ne(0)
        fallback = not bool(local.any())
        if fallback:
            local = support & ~abnormal & raw_image[modality].ne(0)
        if not bool(local.any()):
            raise ValueError(f'No valid donor values for modality {modality}')
        pools.append(local)
        fallbacks.append(fallback)
    return pools, fallbacks


def empirical_seed(global_seed, case_id, patch_index, operator, modality_index):
    payload = f'{global_seed}:{case_id}:{patch_index}:{operator}:{modality_index}'
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') & ((1 << 63) - 1)


def corrupt_input(normalized_image, abnormal, pools, operator, *, global_seed=SUBSET_SEED,
                  case_id='', patch_index=0):
    if operator not in OPERATORS:
        raise ValueError(f'Unsupported corruption operator: {operator}')
    image = torch.as_tensor(normalized_image).clone().float()
    abnormal = torch.as_tensor(abnormal, dtype=torch.bool)
    if image.shape != (4, *PATCH_SHAPE) or abnormal.shape != PATCH_SHAPE:
        raise ValueError('Unexpected normalized image/abnormal shape')
    if not bool(abnormal.any()):
        return image
    if len(pools) != 4:
        raise ValueError('Expected one donor pool per modality')
    for modality, pool in enumerate(pools):
        if operator == 'zero_fill':
            image[modality][abnormal] = 0
        else:
            values = image[modality][pool]
            if values.numel() == 0:
                raise ValueError(f'Empty donor values for modality {modality}')
            if operator == 'local_mean_fill':
                image[modality][abnormal] = values.mean()
            else:
                generator = torch.Generator(device='cpu').manual_seed(
                    empirical_seed(global_seed, case_id, patch_index, operator, modality))
                indices = torch.randint(values.numel(), (int(abnormal.sum()),), generator=generator)
                image[modality][abnormal] = values[indices]
    return image


def region_masks(raw_image, raw_mask, geometry):
    support_occupancy = block_occupancy(support_mask(raw_image))
    abnormal = abnormal_mask(raw_mask)
    direct_abnormal = block_occupancy(abnormal) > 0
    rf = rf_contaminated(abnormal, geometry)
    valid = support_occupancy >= SUPPORT_THRESHOLD
    region_a = valid & direct_abnormal
    region_b = valid & ~direct_abnormal & rf
    region_c = valid & ~rf
    if not torch.equal(region_a | region_b | region_c, valid):
        raise AssertionError('D0-B3 region union does not equal valid support')
    if bool((region_a & region_b).any() or (region_a & region_c).any() or (region_b & region_c).any()):
        raise AssertionError('D0-B3 regions overlap')
    return dict(abnormal=abnormal, valid_support=valid, direct_abnormal=direct_abnormal,
                rf_contaminated=rf, region_a=region_a, region_b=region_b, region_c=region_c)


def load_contract(path):
    path = Path(path).expanduser().resolve()
    if sha256(path) != EXPECTED_CONTRACT_SHA256:
        raise ValueError('D0-A normalization contract SHA256 mismatch')
    contract = json.loads(path.read_text())
    channels = contract.get('channels')
    if contract.get('channel_count') != LATENT_CHANNELS or not isinstance(channels, list):
        raise ValueError('Normalization contract channel schema mismatch')
    if [row.get('channel_index') for row in channels] != list(range(LATENT_CHANNELS)):
        raise ValueError('Normalization contract channels must be 0..255')
    means = torch.tensor([row['mean'] for row in channels], dtype=torch.float64)
    stds = torch.tensor([row['population_std'] for row in channels], dtype=torch.float64)
    if not torch.isfinite(means).all() or not torch.isfinite(stds).all() or bool((stds <= 0).any()):
        raise ValueError('Normalization contract contains invalid values')
    return contract, means, stds


def standardize_za(latent, means, stds):
    latent = torch.as_tensor(latent)
    if latent.ndim != 5 or latent.shape[1:] != (LATENT_CHANNELS, *LATENT_SHAPE):
        raise ValueError(f'Expected latent [B,{LATENT_CHANNELS},12,12,12]')
    return (latent - means[None, :, None, None, None]) / stds[None, :, None, None, None]


@torch.no_grad()
def extract_za(source, batch):
    batch = torch.as_tensor(batch)
    if batch.ndim != 5 or batch.shape[1:] != (4, *PATCH_SHAPE):
        raise ValueError('Expected [B,4,96,96,96] input')
    if any(parameter.requires_grad for parameter in source.parameters()) or source.training:
        raise ValueError('Frozen E1-C source must be eval with requires_grad=False')
    latent = source.projection.anatomy_projector(source.encoder(batch)[-1])
    if tuple(latent.shape[1:]) != (LATENT_CHANNELS, *LATENT_SHAPE) or not torch.isfinite(latent).all():
        raise ValueError('Unexpected or non-finite Za latent')
    return latent


def validate_source_info(source_info):
    if not isinstance(source_info, dict):
        raise ValueError('E1-C source_info must be a dict')
    if source_info.get('epoch') is None or not source_info.get('model'):
        raise ValueError('E1-C source_info is missing epoch/model identity')
    return source_info['epoch'], source_info['model']


def region_values(delta, masks):
    values = {}
    for name, mask in (('a', masks['region_a']), ('b', masks['region_b']),
                       ('c', masks['region_c']), ('all_support', masks['valid_support'])):
        count = int(mask.sum())
        summed = float(delta[mask].sum()) if count else 0.0
        values[f'region_{name}_count'] = count
        values[f'region_{name}_sum'] = summed
        values[f'region_{name}_mean'] = (summed / count) if count else None
    return values


def input_change_metrics(corrupted, original, abnormal):
    corrupted = torch.as_tensor(corrupted)
    original = torch.as_tensor(original)
    abnormal = torch.as_tensor(abnormal, dtype=torch.bool)
    if corrupted.shape != (4, *PATCH_SHAPE) or original.shape != corrupted.shape or abnormal.shape != PATCH_SHAPE:
        raise ValueError('Unexpected input-change diagnostic shape')
    target = int(abnormal.sum())
    result = {'targeted_abnormal_voxel_count': target,
              'targeted_abnormal_fraction_of_patch': target / float(np.prod(PATCH_SHAPE))}
    union = torch.zeros_like(abnormal)
    for modality in range(4):
        difference = (corrupted[modality] - original[modality]).abs()
        changed = difference.ne(0) & abnormal
        count = int(changed.sum())
        result[f'changed_abnormal_voxel_count_{modality}'] = count
        result[f'changed_abnormal_voxel_fraction_{modality}'] = count / target if target else None
        result[f'input_change_mean_abs_{modality}'] = float(difference[abnormal].mean()) if target else None
        result[f'input_change_sum_{modality}'] = float(difference[abnormal].sum())
        union |= changed
    count = int(union.sum())
    result['changed_any_modality_abnormal_voxel_count'] = count
    result['changed_any_modality_abnormal_voxel_fraction'] = count / target if target else None
    return result


def distribution(values):
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(values):
        return {'count': 0}
    return dict(count=int(values.size), mean=float(values.mean()), std=float(values.std(ddof=0)),
                min=float(values.min()), max=float(values.max()),
                **{f'p{p:02d}': float(np.percentile(values, p, method='linear'))
                   for p in (1, 5, 25, 50, 75, 95, 99)})


def safe_output(path, overwrite, inputs):
    output = Path(path).expanduser().resolve()
    frozen = (Path(__file__).resolve().parents[1] / 'outputs/frozen').resolve()
    if output == frozen or frozen in output.parents:
        raise ValueError('D0-B3 output must not be under outputs/frozen')
    if any(output == source or output in source.parents for source in inputs):
        raise ValueError('D0-B3 output overlaps an input')
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError('D0-B3 output is non-empty; use --overwrite')
    if output.exists() and any(item.name not in OUTPUTS for item in output.iterdir()):
        raise ValueError('D0-B3 output contains unrelated files')
    return output


def _case_operator_row(case, operator, metrics):
    row = {'case_id': case, 'operator': operator, 'patch_count': metrics['patch_count'],
           'abnormal_positive_patch_count': metrics['abnormal_positive_patch_count'],
           'abnormal_voxel_count': metrics['abnormal_voxel_count'],
           'donor_pool_fallback_count': metrics['donor_pool_fallback_count']}
    for name in ('a', 'b', 'c', 'all_support'):
        for scope in ('positive', 'all'):
            count = metrics.get(f'{scope}_region_{name}_count', 0)
            row[f'{scope}_region_{name}_count'] = count
            row[f'{scope}_region_{name}_delta_std_mean'] = metrics.get(f'{scope}_region_{name}_delta_std_sum', 0.0) / count if count else None
            row[f'{scope}_region_{name}_delta_raw_mean'] = metrics.get(f'{scope}_region_{name}_delta_raw_sum', 0.0) / count if count else None
    row['primary_delta_std_region_c_mean'] = row['positive_region_c_delta_std_mean']
    row['all_patches_delta_std_region_c_mean'] = row['all_region_c_delta_std_mean']
    row['primary_delta_raw_region_c_mean'] = row['positive_region_c_delta_raw_mean']
    row['all_patches_delta_raw_region_c_mean'] = row['all_region_c_delta_raw_mean']
    for modality in range(4):
        target = metrics.get(f'targeted_abnormal_voxel_count_{modality}', 0)
        changed = metrics.get(f'changed_abnormal_voxel_count_{modality}', 0)
        row[f'input_change_mean_abs_{modality}'] = metrics.get(f'input_change_sum_{modality}', 0.0) / target if target else None
        row[f'targeted_abnormal_voxel_count_{modality}'] = target
        row[f'changed_abnormal_voxel_count_{modality}'] = changed
        row[f'changed_abnormal_voxel_fraction_{modality}'] = changed / target if target else None
    target = metrics.get('targeted_abnormal_voxel_count', 0)
    changed = metrics.get('changed_any_modality_abnormal_voxel_count', 0)
    row['targeted_abnormal_voxel_count'] = target
    row['targeted_abnormal_fraction_of_patch'] = target / float(np.prod(PATCH_SHAPE) * metrics['patch_count']) if target else 0.0
    row['changed_any_modality_abnormal_voxel_count'] = changed
    row['changed_any_modality_abnormal_voxel_fraction'] = changed / target if target else None
    return row


def run(data_config='configs/data.yaml', checkpoint=DEFAULT_CHECKPOINT,
        normalization_contract=DEFAULT_CONTRACT, patch_manifest=DEFAULT_MANIFEST,
        rf_geometry=DEFAULT_RF_GEOMETRY, rf_summary=DEFAULT_RF_SUMMARY,
        output_dir=DEFAULT_OUTPUT, device='cpu', overwrite=False, subset_seed=SUBSET_SEED):
    if subset_seed != SUBSET_SEED:
        raise ValueError(f'D0-B3 scientific subset seed is locked to {SUBSET_SEED}')
    git_info = runtime_git_provenance()
    runtime_branch = git_info['runtime_git_branch']
    if runtime_branch != EXPECTED_BRANCH:
        raise RuntimeError(f'D0-B3 must run on {EXPECTED_BRANCH}; current branch is {runtime_branch}')
    runtime_status = git_info['git_status_short']
    config_path = Path(data_config).expanduser().resolve()
    data = read_config(config_path)
    split = Path(data['train_cases']).expanduser().resolve()
    if sha256(split) != TRAIN_SHA256:
        raise ValueError('Training split SHA256 mismatch')
    case_ids = read_case_ids(split)
    if len(case_ids) != EXPECTED_CASES:
        raise ValueError('Expected exactly 1080 training cases')
    manifest_path = Path(patch_manifest).expanduser().resolve()
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError('D0-B0 patch manifest SHA256 mismatch')
    rows = read_manifest(manifest_path, set(case_ids))
    selected = select_cases(case_ids, subset_seed)
    selected_rows = select_manifest(rows, selected)
    if len(selected_rows) != TOTAL_PATCHES:
        raise ValueError('Selected patch manifest must contain exactly 4000 rows')
    contract_path = Path(normalization_contract).expanduser().resolve()
    contract, means, stds = load_contract(contract_path)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if sha256(checkpoint_path) != E1C_SOURCE_SHA256:
        raise ValueError('E1-C checkpoint SHA256 mismatch')
    rf_geometry_path = Path(rf_geometry).expanduser().resolve()
    if sha256(rf_geometry_path) != EXPECTED_RF_GEOMETRY_SHA256:
        raise ValueError('D0-B2 RF geometry SHA256 mismatch')
    rf_summary_path = Path(rf_summary).expanduser().resolve()
    if sha256(rf_summary_path) != EXPECTED_RF_SUMMARY_SHA256:
        raise ValueError('D0-B2 RF summary SHA256 mismatch')
    geometry_json = json.loads(rf_geometry_path.read_text())
    if (geometry_json.get('input_size'), geometry_json.get('output_size'), geometry_json.get('effective_stride'), geometry_json.get('theoretical_rf_size')) != (96, 12, 8, 68):
        raise ValueError('D0-B2 RF geometry contract mismatch')
    output = safe_output(output_dir, overwrite, (config_path, split, manifest_path, contract_path, checkpoint_path, rf_geometry_path, rf_summary_path))
    source, source_info = load_source(checkpoint_path, source_experiment='e1c', expected_sha256=E1C_SOURCE_SHA256, expected_epoch=11)
    source_epoch, source_model_identity = validate_source_info(source_info)
    source.to(device).eval()
    source.requires_grad_(False)
    means, stds = means.to(device), stds.to(device)
    geometry = derive_encoder_geometry(Encoder())
    if (geometry.input_size, geometry.output_size, geometry.effective_stride, geometry.theoretical_rf_size) != (96, 12, 8, 68):
        raise ValueError('Runtime encoder RF geometry mismatch')
    root = Path(data['root']).expanduser().resolve()
    selected_info = selection_rows(case_ids, subset_seed)
    by_case = {case: [row for row in selected_rows if row['case_id'] == case] for case in selected}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.d0b3-', dir=output.parent) as temporary:
        stage = Path(temporary)
        manifest_fields = list(MANIFEST_FIELDS) + ['selection_digest', 'selection_rank']
        with (stage / 'selected_patch_manifest.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=manifest_fields); writer.writeheader()
            for row in selected_rows:
                digest, rank = selected_info[row['case_id']][1], selected_info[row['case_id']][0]
                writer.writerow({**row, 'selection_digest': digest, 'selection_rank': rank})
        patch_rows = []; case_acc = {}
        for case in selected:
            raw_image, raw_mask = load_case(root, case)
            normalized = normalize_nonzero(raw_image)
            for identity in by_case[case]:
                slices = tuple(slice(identity[f'crop_start_{axis}'], identity[f'crop_stop_{axis}']) for axis in 'dhw')
                raw_patch = raw_image[(slice(None), *slices)]
                normalized_patch = normalized[(slice(None), *slices)].contiguous()
                raw_mask_patch = raw_mask[slices]
                masks = region_masks(raw_patch, raw_mask_patch, geometry)
                support = support_mask(raw_patch)
                pools, fallbacks = donor_pools(raw_patch, masks['abnormal'], support)
                versions = [normalized_patch] + [corrupt_input(normalized_patch, masks['abnormal'], pools, op, global_seed=subset_seed, case_id=case, patch_index=identity['patch_index']) for op in OPERATORS]
                raw_latent = extract_za(source, torch.stack(versions).to(device))
                latent = standardize_za(raw_latent, means, stds).cpu()
                raw_latent = raw_latent.cpu()
                abnormal_count = int(masks['abnormal'].sum())
                positive = bool(abnormal_count)
                base = dict(case_id=case, patch_index=identity['patch_index'], abnormal_voxel_count=abnormal_count,
                            abnormal_positive_patch=positive, valid_support_count=int(masks['valid_support'].sum()),
                            region_a_count=int(masks['region_a'].sum()), region_b_count=int(masks['region_b'].sum()), region_c_count=int(masks['region_c'].sum()),
                            region_all_support_count=int(masks['valid_support'].sum()), donor_pool_fallback_count=int(sum(fallbacks)))
                for op_index, operator in enumerate(OPERATORS, start=1):
                    delta_std = (latent[0] - latent[op_index]).abs().mean(dim=0)
                    delta_raw = (raw_latent[0] - raw_latent[op_index]).abs().mean(dim=0)
                    stats = region_values(delta_std, masks)
                    raw_stats = region_values(delta_raw, masks)
                    row = {**base, 'operator': operator,
                           'rf_contaminated_count': int(masks['rf_contaminated'].sum()),
                           'rf_contaminated_fraction_all_latent': float(masks['rf_contaminated'].double().mean()),
                           'valid_support_fraction': float(masks['valid_support'].double().mean()),
                           'abnormal_input_fraction': abnormal_count / float(np.prod(PATCH_SHAPE)),
                           'targeted_abnormal_voxel_count': abnormal_count,
                           'targeted_abnormal_fraction_of_patch': abnormal_count / float(np.prod(PATCH_SHAPE))}
                    for name in ('a', 'b', 'c', 'all_support'):
                        row[f'region_{name}_count'] = stats[f'region_{name}_count']
                        row[f'delta_std_region_{name}_mean'] = stats[f'region_{name}_mean']
                        row[f'delta_raw_region_{name}_mean'] = raw_stats[f'region_{name}_mean']
                    change = input_change_metrics(versions[op_index], normalized_patch, masks['abnormal'])
                    row.update(change)
                    for modality in range(4):
                        row[f'donor_pool_count_{modality}'] = int(pools[modality].sum())
                        row[f'donor_pool_fallback_{modality}'] = fallbacks[modality]
                    patch_rows.append(row)
                    accumulator = case_acc.setdefault((case, operator), {'patch_count': 0, 'abnormal_positive_patch_count': 0, 'abnormal_voxel_count': 0, 'donor_pool_fallback_count': 0})
                    accumulator['patch_count'] += 1; accumulator['abnormal_positive_patch_count'] += positive; accumulator['abnormal_voxel_count'] += abnormal_count; accumulator['donor_pool_fallback_count'] += sum(fallbacks) if positive else 0
                    for name in ('a', 'b', 'c', 'all_support'):
                        accumulator[f'all_region_{name}_count'] = accumulator.get(f'all_region_{name}_count', 0) + stats[f'region_{name}_count']
                        accumulator[f'all_region_{name}_delta_std_sum'] = accumulator.get(f'all_region_{name}_delta_std_sum', 0.0) + stats[f'region_{name}_sum']
                        accumulator[f'all_region_{name}_delta_raw_sum'] = accumulator.get(f'all_region_{name}_delta_raw_sum', 0.0) + raw_stats[f'region_{name}_sum']
                        if positive:
                            accumulator[f'positive_region_{name}_count'] = accumulator.get(f'positive_region_{name}_count', 0) + stats[f'region_{name}_count']
                            accumulator[f'positive_region_{name}_delta_std_sum'] = accumulator.get(f'positive_region_{name}_delta_std_sum', 0.0) + stats[f'region_{name}_sum']
                            accumulator[f'positive_region_{name}_delta_raw_sum'] = accumulator.get(f'positive_region_{name}_delta_raw_sum', 0.0) + raw_stats[f'region_{name}_sum']
                    for modality in range(4):
                        accumulator[f'targeted_abnormal_voxel_count_{modality}'] = accumulator.get(f'targeted_abnormal_voxel_count_{modality}', 0) + abnormal_count
                        accumulator[f'changed_abnormal_voxel_count_{modality}'] = accumulator.get(f'changed_abnormal_voxel_count_{modality}', 0) + change[f'changed_abnormal_voxel_count_{modality}']
                        accumulator[f'input_change_sum_{modality}'] = accumulator.get(f'input_change_sum_{modality}', 0.0) + change[f'input_change_sum_{modality}']
                    accumulator['targeted_abnormal_voxel_count'] = accumulator.get('targeted_abnormal_voxel_count', 0) + abnormal_count
                    accumulator['changed_any_modality_abnormal_voxel_count'] = accumulator.get('changed_any_modality_abnormal_voxel_count', 0) + change['changed_any_modality_abnormal_voxel_count']
        if len(selected_rows) != TOTAL_PATCHES:
            raise AssertionError('Selected patch row count is not 4000')
        if len(patch_rows) != TOTAL_PATCHES * len(OPERATORS):
            raise AssertionError('Patch corruption row count is not 12000')
        if len(case_rows := [_case_operator_row(case, op, case_acc[(case, op)]) for case in selected for op in OPERATORS]) != SUBSET_CASE_COUNT * len(OPERATORS):
            raise AssertionError('Case/operator row count is not 600')
        if any(sum(row['operator'] == operator for row in patch_rows) != TOTAL_PATCHES for operator in OPERATORS):
            raise AssertionError('Each operator must have exactly 4000 patch rows')
        if any(sum(row['operator'] == operator for row in case_rows) != SUBSET_CASE_COUNT for operator in OPERATORS):
            raise AssertionError('Each operator must have exactly 200 case rows')
        with (stage / 'patch_corruption_statistics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(patch_rows[0])); writer.writeheader(); writer.writerows(patch_rows)
        with (stage / 'case_corruption_statistics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(case_rows[0])); writer.writeheader(); writer.writerows(case_rows)
        distributions = {}
        case_distributions = {}
        for operator in OPERATORS:
            op_rows = [row for row in patch_rows if row['operator'] == operator]
            distributions[operator] = {f'delta_std_region_{name}_mean': distribution([row[f'delta_std_region_{name}_mean'] for row in op_rows]) for name in ('a', 'b', 'c', 'all_support')}
            distributions[operator].update({f'delta_raw_region_{name}_mean': distribution([row[f'delta_raw_region_{name}_mean'] for row in op_rows]) for name in ('a', 'b', 'c', 'all_support')})
            case_op_rows = [row for row in case_rows if row['operator'] == operator]
            case_distributions[operator] = {f'primary_delta_std_region_c_mean': distribution([row['primary_delta_std_region_c_mean'] for row in case_op_rows]),
                                            'all_patches_delta_std_region_c_mean': distribution([row['all_patches_delta_std_region_c_mean'] for row in case_op_rows])}
            case_distributions[operator].update({f'positive_delta_std_region_{name}_mean': distribution([row[f'positive_region_{name}_delta_std_mean'] for row in case_op_rows]) for name in ('a', 'b', 'c', 'all_support')})
            case_distributions[operator].update({f'all_delta_std_region_{name}_mean': distribution([row[f'all_region_{name}_delta_std_mean'] for row in case_op_rows]) for name in ('a', 'b', 'c', 'all_support')})
            case_distributions[operator].update({f'input_change_mean_abs_{m}': distribution([row[f'input_change_mean_abs_{m}'] for row in case_op_rows]) for m in range(4)})
            case_distributions[operator].update({f'changed_abnormal_voxel_fraction_{m}': distribution([row[f'changed_abnormal_voxel_fraction_{m}'] for row in case_op_rows]) for m in range(4)})
            case_distributions[operator]['changed_any_modality_abnormal_voxel_fraction'] = distribution([row['changed_any_modality_abnormal_voxel_fraction'] for row in case_op_rows])
        base_patch_rows = [row for row in patch_rows if row['operator'] == 'zero_fill']
        positive_patch_rows = [row for row in base_patch_rows if row['abnormal_positive_patch']]
        summary = dict(case_count=SUBSET_CASE_COUNT, patch_count=TOTAL_PATCHES, operators=list(OPERATORS),
                       abnormal_positive_patch_count=len(positive_patch_rows),
                       abnormal_negative_patch_count=TOTAL_PATCHES - len(positive_patch_rows),
                       distributions=distributions, case_distributions=case_distributions,
                       primary_metric='case-level weighted mean standardized Za absolute difference over REGION_C, restricted to abnormal-positive patches',
                       primary_metric_defined_case_counts={op: sum(row['primary_delta_std_region_c_mean'] is not None for row in case_rows if row['operator'] == op) for op in OPERATORS},
                       all_patch_secondary_metric='case-level weighted mean standardized Za absolute difference over REGION_C using all patches',
                       donor_pools_shared=True, zero_fill_uses_donor_values=False,
                       donor_pool_fallback_frequencies_abnormal_positive={str(m): sum(row[f'donor_pool_fallback_{m}'] for row in positive_patch_rows) / len(positive_patch_rows) if positive_patch_rows else None for m in range(4)},
                       no_operator_selected=True, partition_integrity=True)
        (stage / 'corruption_operator_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        provenance = dict(schema='d0b3_corruption_operator_audit', schema_version=1,
                          audit_name='D0-B3 Corruption Operator Audit', expected_branch=EXPECTED_BRANCH,
                          runtime_git_branch=runtime_branch, runtime_git_commit=git_info['runtime_git_commit'],
                          git_dirty=bool(runtime_status), git_status_short=runtime_status,
                          experiment_base_commit=EXPERIMENT_BASE_COMMIT, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                          train_split_path=str(split), train_split_sha256=sha256(split),
                          e1c_checkpoint_path=str(checkpoint_path), e1c_checkpoint_sha256=sha256(checkpoint_path),
                          d0a_normalization_contract_path=str(contract_path), d0a_normalization_contract_sha256=sha256(contract_path),
                          d0b0_manifest_path=str(manifest_path), d0b0_manifest_sha256=sha256(manifest_path),
                          d0b2_rf_geometry_path=str(rf_geometry_path), d0b2_rf_geometry_sha256=sha256(rf_geometry_path),
                          d0b2_rf_summary_path=str(rf_summary_path), d0b2_rf_summary_sha256=sha256(rf_summary_path),
                          data_config_path=str(config_path), data_config_sha256=sha256(config_path), dataset_root=str(root),
                          python_version=platform.python_version(), numpy_version=np.__version__, pytorch_version=str(torch.__version__),
                          device=str(device), cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
                          cuda_device_name=(torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == 'cuda' and torch.cuda.is_available() else None),
                          nibabel_version=nib.__version__, source_epoch=source_epoch, source_model_identity=source_model_identity,
                          subset_seed=subset_seed, selection_algorithm='SHA256(f"{seed}:{case_id}") lexicographic digest, case_id tie-breaker',
                          selected_case_count=SUBSET_CASE_COUNT, patches_per_case=PATCHES_PER_CASE, selected_patch_count=TOTAL_PATCHES,
                          corruption_operators=list(OPERATORS), corruption_margin_voxels=0,
                          corruption_definitions={
                              'zero_fill': {'replacement': 'normalized value 0', 'donor_values_used': False},
                              'local_mean_fill': {'replacement': 'arithmetic mean of normalized values from valid donor pool', 'donor_values_used': True},
                              'local_empirical_resample': {'replacement': 'sampling with replacement from actual normalized donor values', 'donor_values_used': True}},
                          donor_pools_shared=True, zero_fill_uses_donor_values=False, donor_radius_voxels=8,
                          donor_distance='Chebyshev', empirical_resample_with_replacement=True,
                          empirical_seed_derivation='SHA256(global_seed:case_id:patch_index:operator_name:modality_index); first 8 digest bytes, big-endian, masked to 63 bits',
                          corruption_stage='after full-volume normalization and frozen 96^3 crop', donor_value_space='normalized MRI values',
                          donor_eligibility_space='original raw MRI/support', raw_abnormal_labels=list(RAW_ABNORMAL_LABELS), donor_shell='Chebyshev radius 8; cubic 17x17x17 dilation AND NOT abnormal',
                          donor_fallback='same patch, non-abnormal, original multimodal support, modality raw value nonzero; shared diagnostic for donor-based operators only',
                          support_definition='ANY original raw MRI modality nonzero before normalization; exact 8x8x8 occupancy; valid threshold >=0.50',
                          support_threshold=SUPPORT_THRESHOLD, primary_statistical_unit='case', primary_metric_scope='abnormal-positive patches only',
                          geometric_rf=dict(input_size=96, output_size=12, effective_stride=8, theoretical_rf_size=68, interpretation='local Conv/Pool geometric RF only'),
                          validation_used=False, model_training=False, diffusion_loaded=False, new_patch_sampling=False, patch_rejection=False,
                          augmentation=False, final_operator_selected=False, healthy_synthesis_claim=False, normative_latent_claim=False,
                          methodology_caveat='The 68^3 value represents the local Conv/Pool geometric receptive field. GroupNorm introduces spatial statistical coupling outside this local geometric receptive field; therefore conv_rf_clean cells are not interpreted as guaranteed pathology-independent latent cells.')
        (stage / 'analysis_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        output.mkdir(exist_ok=True)
        for name in OUTPUTS: (stage / name).replace(output / name)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', default='configs/data.yaml')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--normalization-contract', default=DEFAULT_CONTRACT)
    parser.add_argument('--patch-manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--rf-geometry', default=DEFAULT_RF_GEOMETRY)
    parser.add_argument('--rf-summary', default=DEFAULT_RF_SUMMARY)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--overwrite', action='store_true')
    print(run(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
