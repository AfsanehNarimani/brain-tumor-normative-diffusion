"""Microbenchmark exact target-after-crop equivalence and lossless MRI storage.

This is diagnostic infrastructure, not a change to the scientific preprocessing
protocol or a production cache.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time

import nibabel as nib
import numpy as np
import torch
import yaml

from datasets.brats_dataset import MODALITIES, load_subject
from datasets.label_mapping import KNOWN_LABELS, TISSUE_LABELS, TISSUE_NAMES, segmentation_to_tissue_masks
from datasets.patch_sampling import DEFAULT_PATCH_SIZE, PatchSampler
from datasets.transforms import normalize_nonzero
from scripts.train_baseline import read_case_ids


TARGET_TIMINGS = (
    "full_volume_target_construction_ms",
    "full_target_crop_ms",
    "cropped_mask_target_construction_ms",
    "original_target_path_ms",
)
CACHE_TIMINGS = (
    "save_compression_ms",
    "cached_load_decompression_ms",
    "production_subject_load_plus_normalization_ms",
)


def distribution(values):
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "p95": p95,
    }


def sha256_file(path):
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


def git_provenance():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    short = subprocess.run(
        ["git", "status", "--short"], capture_output=True, text=True, check=True
    ).stdout
    return {
        "git_commit_sha": commit,
        "git_dirty": bool(porcelain),
        "git_status_short": [line for line in short.splitlines() if line],
    }


def benchmark_target_crop(full_mask, sampler, case_id, crop_number):
    # BraTSDataset.__getitem__ builds this raw-tissue foreground union before
    # calling the same configured PatchSampler. There is no project helper for
    # this short inline step; no crop or label conversion logic is replaced.
    foreground = torch.zeros_like(full_mask, dtype=torch.bool)
    for label in TISSUE_LABELS:
        foreground |= full_mask == label
    slices = sampler.sample_slices(foreground)

    start = time.perf_counter()
    original_full_target = segmentation_to_tissue_masks(full_mask)
    full_construction_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    original_target_patch = original_full_target[(slice(None), *slices)].contiguous()
    full_crop_ms = (time.perf_counter() - start) * 1000.0

    candidate_mask_patch = full_mask[slices]
    start = time.perf_counter()
    candidate_target_patch = segmentation_to_tissue_masks(candidate_mask_patch)
    cropped_construction_ms = (time.perf_counter() - start) * 1000.0

    expected_shape = (len(TISSUE_LABELS), *(part.stop - part.start for part in slices))
    if original_target_patch.shape != expected_shape or candidate_target_patch.shape != expected_shape:
        raise RuntimeError(f"{case_id} crop {crop_number}: target shape mismatch")
    if original_target_patch.dtype != torch.float32 or candidate_target_patch.dtype != torch.float32:
        raise RuntimeError(f"{case_id} crop {crop_number}: target dtype mismatch")
    if not torch.equal(torch.unique(original_target_patch), torch.unique(candidate_target_patch)):
        raise RuntimeError(f"{case_id} crop {crop_number}: target unique values differ")
    if not torch.equal(original_target_patch, candidate_target_patch):
        raise RuntimeError(f"{case_id} crop {crop_number}: target values differ")
    for channel, label in enumerate(TISSUE_LABELS):
        expected_channel = (candidate_mask_patch == label).to(torch.float32)
        if not torch.equal(candidate_target_patch[channel], expected_channel):
            raise RuntimeError(f"{case_id} crop {crop_number}: {TISSUE_NAMES[channel]} channel is out of order")

    return {
        "crop_number": crop_number,
        "crop_start": [part.start for part in slices],
        "crop_shape": list(expected_shape),
        "target_dtype": str(candidate_target_patch.dtype),
        "target_unique_values": torch.unique(candidate_target_patch).tolist(),
        "exact_equal": True,
        "full_volume_target_construction_ms": full_construction_ms,
        "full_target_crop_ms": full_crop_ms,
        "cropped_mask_target_construction_ms": cropped_construction_ms,
        "original_target_path_ms": full_construction_ms + full_crop_ms,
    }


def benchmark_cache(reference_tensor, production_subject_ms, case_id, cache_path):
    if reference_tensor.dtype != torch.float32 or not torch.isfinite(reference_tensor).all():
        raise RuntimeError(f"{case_id}: normalized reference must be finite float32")
    reference_numpy = reference_tensor.numpy()
    if reference_numpy.dtype != np.float32 or not np.isfinite(reference_numpy).all():
        raise RuntimeError(f"{case_id}: NumPy reference must be finite float32")

    try:
        start = time.perf_counter()
        np.savez_compressed(cache_path, image=reference_numpy)
        save_ms = (time.perf_counter() - start) * 1000.0
        compressed_bytes = cache_path.stat().st_size

        start = time.perf_counter()
        with np.load(cache_path, allow_pickle=False) as archive:
            loaded_numpy = archive["image"]
        loaded_tensor = torch.from_numpy(loaded_numpy)
        cached_load_ms = (time.perf_counter() - start) * 1000.0

        if loaded_numpy.dtype != np.float32 or loaded_tensor.dtype != torch.float32:
            raise RuntimeError(f"{case_id}: NPZ round-trip dtype mismatch")
        if loaded_numpy.shape != reference_numpy.shape or loaded_tensor.shape != reference_tensor.shape:
            raise RuntimeError(f"{case_id}: NPZ round-trip shape mismatch")
        if not np.isfinite(loaded_numpy).all() or not torch.isfinite(loaded_tensor).all():
            raise RuntimeError(f"{case_id}: NPZ round-trip contains non-finite values")
        numpy_equal = bool(np.array_equal(reference_numpy, loaded_numpy))
        torch_equal = bool(torch.equal(reference_tensor, loaded_tensor))
        if not numpy_equal or not torch_equal:
            raise RuntimeError(f"{case_id}: NPZ round-trip is not exactly equal")

        logical_bytes = reference_numpy.nbytes
        return {
            "case_id": case_id,
            "numpy_exact_equal": numpy_equal,
            "torch_exact_equal": torch_equal,
            "shape": list(reference_numpy.shape),
            "dtype": str(reference_numpy.dtype),
            "logical_uncompressed_bytes": logical_bytes,
            "compressed_file_bytes": compressed_bytes,
            "compression_ratio": logical_bytes / compressed_bytes,
            "save_compression_ms": save_ms,
            "cached_load_decompression_ms": cached_load_ms,
            "production_subject_load_plus_normalization_ms": production_subject_ms,
            "production_subject_path_to_cached_mri_load_ratio": production_subject_ms / cached_load_ms,
        }
    finally:
        cache_path.unlink(missing_ok=True)


def benchmark_case(record, sampler, crops_per_case, cache_path):
    case_id = record["subject_id"]
    start = time.perf_counter()
    image, full_mask = load_subject(record)
    normalized_image = normalize_nonzero(image)
    production_subject_ms = (time.perf_counter() - start) * 1000.0
    del image

    labels = set(torch.unique(full_mask).tolist())
    if not labels <= set(KNOWN_LABELS):
        raise RuntimeError(f"{case_id}: unexpected segmentation labels: {sorted(labels - set(KNOWN_LABELS))}")

    cache_result = benchmark_cache(normalized_image, production_subject_ms, case_id, cache_path)
    del normalized_image
    crops = [
        benchmark_target_crop(full_mask, sampler, case_id, number + 1)
        for number in range(crops_per_case)
    ]
    return crops, cache_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--split-file", default="splits/train_cases.txt")
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--crops-per-case", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.cases < 1 or args.crops_per_case < 1:
        parser.error("--cases and --crops-per-case must be positive")

    with Path(args.data_config).expanduser().open() as stream:
        config = yaml.safe_load(stream)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    case_ids = read_case_ids(args.split_file)
    if args.cases > len(case_ids):
        parser.error(f"--cases exceeds selected train split size ({len(case_ids)})")

    # Match discover_cases record paths for selected split IDs only; load_subject
    # performs the production file checks and MRI/segmentation loading below.
    root = Path(config["root"]).expanduser()
    selected = []
    for case_id in case_ids[:args.cases]:
        directory = root / case_id
        paths = {key: directory / f"{case_id}-{key.lower()}.nii.gz" for key in MODALITIES}
        paths["mask"] = directory / f"{case_id}-seg.nii.gz"
        selected.append({"subject_id": case_id, "paths": paths})
    crop_generator = torch.Generator().manual_seed(args.seed)
    sampler = PatchSampler(
        config.get("patch_size", DEFAULT_PATCH_SIZE),
        config.get("foreground_sampling_probability", 0.7),
        generator=crop_generator,
    )
    target_cases = []
    cache_cases = []
    with tempfile.TemporaryDirectory(prefix="brats-preprocessing-microbenchmark-") as temporary:
        cache_path = Path(temporary) / "normalized_image.npz"
        for record in selected:
            crops, cache_result = benchmark_case(
                record, sampler, args.crops_per_case, cache_path
            )
            target_cases.append({"case_id": record["subject_id"], "crops": crops})
            cache_cases.append(cache_result)

    comparisons = [crop for case in target_cases for crop in case["crops"]]
    original_mean = statistics.mean(crop["original_target_path_ms"] for crop in comparisons)
    candidate_mean = statistics.mean(crop["cropped_mask_target_construction_ms"] for crop in comparisons)
    source_mean = statistics.mean(case["production_subject_load_plus_normalization_ms"] for case in cache_cases)
    cached_mean = statistics.mean(case["cached_load_decompression_ms"] for case in cache_cases)

    summary = {
        "benchmark_purpose": "computational equivalence and performance feasibility only",
        "tested_case_ids": [record["subject_id"] for record in selected],
        "case_count": len(selected),
        "crops_per_case": args.crops_per_case,
        "seed": args.seed,
        "data_root": str(Path(config["root"]).expanduser().resolve()),
        "split_file": str(Path(args.split_file).expanduser().resolve()),
        "split_file_sha256": sha256_file(args.split_file),
        "data_config_sha256": sha256_file(args.data_config),
        **git_provenance(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pytorch_version": torch.__version__,
        "nibabel_version": nib.__version__,
        "part_a_target_after_crop": {
            "total_comparisons": len(comparisons),
            "exact_equality_pass_count": len(comparisons),
            "exact_equality_failure_count": 0,
            "channel_order": list(TISSUE_NAMES),
            "timings_ms": {
                name: distribution([crop[name] for crop in comparisons]) for name in TARGET_TIMINGS
            },
            "original_mean_target_path_ms": original_mean,
            "candidate_mean_target_path_ms": candidate_mean,
            "speedup_ratio": original_mean / candidate_mean,
            "per_case": target_cases,
        },
        "part_b_lossless_normalized_mri_npz": {
            "all_exact_equal": all(case["numpy_exact_equal"] and case["torch_exact_equal"] for case in cache_cases),
            "reference_path_note": (
                "The production reference includes four MRI NIfTI loads, one segmentation NIfTI load, "
                "production validation and tensor assembly in load_subject, and full-volume normalization. "
                "The cached path includes only normalized MRI cache loading/decompression and conversion "
                "back to a torch tensor. The ratio is a feasibility comparison against the current "
                "production subject-loading path, not an isolated MRI-I/O speedup."
            ),
            "timings_ms": {
                name: distribution([case[name] for case in cache_cases]) for name in CACHE_TIMINGS
            },
            "logical_uncompressed_bytes": distribution([case["logical_uncompressed_bytes"] for case in cache_cases]),
            "compressed_file_bytes": distribution([case["compressed_file_bytes"] for case in cache_cases]),
            "compression_ratio": distribution([case["compression_ratio"] for case in cache_cases]),
            "production_subject_path_mean_ms": source_mean,
            "cached_mri_load_mean_ms": cached_mean,
            "production_subject_path_to_cached_mri_load_ratio": source_mean / cached_mean,
            "per_case": cache_cases,
        },
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output_json:
        args.output_json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
