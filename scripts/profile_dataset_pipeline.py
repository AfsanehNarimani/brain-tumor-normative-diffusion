"""Diagnose CPU sample preparation without changing the BraTS preprocessing protocol."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
import yaml

from datasets import brats_dataset
from scripts.train_baseline import make_dataset


STAGES = (
    "nifti_open_metadata_ms",
    "image_get_fdata_ms",
    "segmentation_get_fdata_ms",
    "load_validation_and_tensor_assembly_ms",
    "normalization_ms",
    "target_construction_ms",
    "patch_sampling_ms",
    "augmentation_ms",
    "foreground_mask_patch_extraction_and_sample_assembly_ms",
    "total_sample_ms",
)
EXCLUSIVE_STAGES = STAGES[:-1]


def percentile_95(values):
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def git_sha():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_worktree_status():
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    short = subprocess.run(
        ["git", "status", "--short"], capture_output=True, text=True, check=True
    )
    return bool(porcelain.stdout), [line for line in short.stdout.splitlines() if line]


def sha256_file(path):
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


def profile(dataset, samples, warmup):
    measurements = {stage: [] for stage in STAGES}
    # Values in current are seconds. load_subject_ms and subject_nifti_open_ms
    # are inclusive accounting helpers, never independent reported stages.
    current = {}
    state = {"in_load_subject": False, "record": None, "loaded_images": {}, "get_fdata_keys": []}

    def timed(stage, operation):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return operation(*args, **kwargs)
            finally:
                current[stage] = current.get(stage, 0.0) + time.perf_counter() - start

        return wrapper

    original_load = nib.load
    original_get_fdata = nib.dataobj_images.DataobjImage.get_fdata
    original_load_subject = brats_dataset.load_subject

    def timed_nifti_load(*args, **kwargs):
        stage = "nifti_open_metadata_ms"
        start = time.perf_counter()
        try:
            image = original_load(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            current[stage] = current.get(stage, 0.0) + elapsed
            if state["in_load_subject"]:
                current["subject_nifti_open_ms"] = current.get("subject_nifti_open_ms", 0.0) + elapsed
        if state["in_load_subject"]:
            path = Path(args[0])
            keys = [key for key, expected in state["record"]["paths"].items() if path == expected]
            if len(keys) != 1:
                raise RuntimeError(f"Unexpected NIfTI path inside load_subject: {path}")
            state["loaded_images"][id(image)] = (image, keys[0])
        return image

    def timed_get_fdata(image, *args, **kwargs):
        loaded = state["loaded_images"].get(id(image))
        if not state["in_load_subject"] or loaded is None or loaded[0] is not image:
            raise RuntimeError("get_fdata called outside a tracked load_subject image")
        key = loaded[1]
        if key in state["get_fdata_keys"]:
            raise RuntimeError(f"Repeated get_fdata for {key}")
        state["get_fdata_keys"].append(key)
        stage = "segmentation_get_fdata_ms" if key == "mask" else "image_get_fdata_ms"
        return timed(stage, original_get_fdata)(image, *args, **kwargs)

    def timed_load_subject(record):
        state["in_load_subject"] = True
        state["record"] = record
        start = time.perf_counter()
        try:
            return original_load_subject(record)
        finally:
            current["load_subject_ms"] = time.perf_counter() - start
            state["in_load_subject"] = False
            state["record"] = None
            state["loaded_images"].clear()

    # These wrappers call the production functions in their original order. The
    # inline foreground union, crop, and sample-dict assembly cannot be separated
    # safely without copying scientific dataset logic, so they share a remainder.
    with ExitStack() as stack:
        stack.enter_context(patch.object(nib, "load", timed_nifti_load))
        stack.enter_context(patch.object(nib.dataobj_images.DataobjImage, "get_fdata", timed_get_fdata))
        stack.enter_context(patch.object(brats_dataset, "load_subject", timed_load_subject))
        stack.enter_context(patch.object(
            brats_dataset, "normalize_nonzero",
            timed("normalization_ms", brats_dataset.normalize_nonzero),
        ))
        stack.enter_context(patch.object(
            dataset, "label_mapping", timed("target_construction_ms", dataset.label_mapping)
        ))
        stack.enter_context(patch.object(
            dataset.patch_sampler, "sample_slices",
            timed("patch_sampling_ms", dataset.patch_sampler.sample_slices),
        ))
        stack.enter_context(patch.object(
            dataset, "augmentation", timed("augmentation_ms", dataset.augmentation)
        ))

        for number in range(warmup + samples):
            current.clear()
            state["loaded_images"].clear()
            state["get_fdata_keys"].clear()
            start = time.perf_counter()
            dataset[number % len(dataset)]
            current["total_sample_ms"] = time.perf_counter() - start
            expected_keys = {*brats_dataset.MODALITIES, "mask"}
            if len(state["get_fdata_keys"]) != 5 or set(state["get_fdata_keys"]) != expected_keys:
                raise RuntimeError("Expected one get_fdata for each of four modalities and the segmentation")

            # load_subject validates shapes, spacing, affines, and finite values
            # between loads. Its remainder also contains file checks and tensor
            # assembly, which cannot be split without replacing that function.
            current["load_validation_and_tensor_assembly_ms"] = (
                current["load_subject_ms"]
                - current.get("subject_nifti_open_ms", 0.0)
                - current.get("image_get_fdata_ms", 0.0)
                - current.get("segmentation_get_fdata_ms", 0.0)
            )
            current["foreground_mask_patch_extraction_and_sample_assembly_ms"] = (
                current["total_sample_ms"]
                - sum(current.get(stage, 0.0) for stage in EXCLUSIVE_STAGES
                      if stage != "foreground_mask_patch_extraction_and_sample_assembly_ms")
            )
            if number >= warmup:
                tolerance = max(1e-6, current["total_sample_ms"] * 1e-6)
                for stage in EXCLUSIVE_STAGES:
                    if current.get(stage, 0.0) < -tolerance:
                        raise RuntimeError(f"Negative exclusive timing for {stage}")
                    current[stage] = max(0.0, current.get(stage, 0.0))
                if abs(sum(current[stage] for stage in EXCLUSIVE_STAGES)
                       - current["total_sample_ms"]) > tolerance:
                    raise RuntimeError("Exclusive stage timings do not match total sample time")
                for stage in STAGES:
                    measurements[stage].append(current.get(stage, 0.0) * 1000.0)

    return measurements


def summarize(measurements):
    mean_total = statistics.mean(measurements["total_sample_ms"])
    stages = {}
    for name, values in measurements.items():
        mean = statistics.mean(values)
        stages[name] = {
            "mean_ms": mean,
            "median_ms": statistics.median(values),
            "std_ms": statistics.pstdev(values),
            "p95_ms": percentile_95(values),
            "pct_mean_total": 100.0 * mean / mean_total,
        }

    def fraction(names):
        return 100.0 * sum(stages[name]["mean_ms"] for name in names) / mean_total

    io = STAGES[:3]
    # The extra nib.load for sample spacing stays in NIfTI I/O. The final
    # remainder also includes minor dataset dispatch overhead, so this is not
    # labeled strictly "post-load".
    preparation = EXCLUSIVE_STAGES[4:]
    fractions = {
        "nifti_io_loading_pct": fraction(io),
        "full_volume_normalization_pct": fraction(("normalization_ms",)),
        "sample_preparation_excluding_io_and_load_validation_pct": fraction(preparation),
    }
    return stages, fractions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--split-file", default="splits/train_cases.txt")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error("--samples must be positive and --warmup must be nonnegative")

    with Path(args.data_config).expanduser().open() as stream:
        config = yaml.safe_load(stream)
    config["train_cases"] = args.split_file
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = make_dataset(config, "train")

    measurements = profile(dataset, args.samples, args.warmup)
    stages, fractions = summarize(measurements)
    git_dirty, git_status_short = git_worktree_status()
    summary = {
        "measured_samples": args.samples,
        "warmup_samples": args.warmup,
        "seed": args.seed,
        "data_root": str(Path(config["root"]).expanduser().resolve()),
        "split_file": str(Path(args.split_file).expanduser().resolve()),
        "split_file_sha256": sha256_file(args.split_file),
        "data_config_sha256": sha256_file(args.data_config),
        "split_cases": len(dataset),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "nibabel_version": nib.__version__,
        "pytorch_version": torch.__version__,
        "git_commit_sha": git_sha(),
        "git_dirty": git_dirty,
        "git_status_short": git_status_short,
        "stages": stages,
        "fractions": fractions,
    }

    print("=== CPU DATASET PIPELINE PROFILE ===")
    for key, value in summary.items():
        if key not in ("stages", "fractions"):
            print(f"{key}: {value}")
    print("\nstage                                                       mean_ms median_ms   std_ms   p95_ms pct_total")
    for name, values in stages.items():
        print(f"{name:59s} {values['mean_ms']:8.2f} {values['median_ms']:9.2f} "
              f"{values['std_ms']:8.2f} {values['p95_ms']:8.2f} {values['pct_mean_total']:9.2f}")
    for name, value in fractions.items():
        print(f"{name}: {value:.2f}")
    print("\nCombined stages include validation with tensor assembly, and foreground mask "
          "with crop and sample assembly, because those operations are inline in BraTSDataset.")

    if args.output_json:
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
