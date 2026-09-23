"""Non-scientific, one-microbatch GPU memory/shape preflight for D0-B4a."""

from __future__ import annotations

import argparse
import hashlib
import random

import numpy as np
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from datasets.transforms import normalize_nonzero
from models.latent_diffusion_unet3d import LatentDiffusionUNet3D, cosine_schedule, masked_epsilon_mse, q_sample
from models.latent_probe import E1C_SOURCE_SHA256, file_sha256, load_source
from scripts.audit_d0b1_abnormal_occupancy import block_occupancy, read_manifest, support_mask, load_case
from scripts.audit_d0b3_corruption_operator import corrupt_input, donor_pools, select_cases, select_manifest, extract_za
from scripts.train_baseline import read_case_ids, read_config

EXPECTED_BRANCH = "experiment/d0b4-diffusion-design"
TRAIN_SHA256 = "581dc6326a22cb0b9582fa249b8e13fd948597a60c5f454cb5ade0493fa7c18e"
CONTRACT_SHA256 = "bca5c0f8f1617d73fc4d0850fe5b6f45e4b2bcbacbb4018f1d5d825de49ad792"
CHECKPOINT_SHA256 = E1C_SOURCE_SHA256
PATCH_MANIFEST_SHA256 = "9c4ba67fc195537a5500229722ce55e4772ea5b7d135b351b3a2fc4b857f44b9"
PREFLIGHT_SEED = 20260923
EFFECTIVE_BATCH = 8
D0B3_DECISION = Path("outputs/frozen/d0b3_corruption_operator_audit/decision/corruption_operator_decision.json")
D0B3_DECISION_SHA256 = "095e1f9671411eca0d12a2f834833d78477a14664504bde3e6e1b46766c6cc2f"


def runtime_git_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    git = lambda *args: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    return {"runtime_git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "runtime_git_commit": git("rev-parse", "HEAD"),
            "git_status_short": git("status", "--short").splitlines()}


def validate_runtime_branch(branch: str) -> None:
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"D0-B4a preflight requires branch {EXPECTED_BRANCH}; got {branch}")


def validate_microbatch(microbatch: int) -> int:
    if microbatch not in (1, 2, 4, 8) or EFFECTIVE_BATCH % microbatch:
        raise ValueError("microbatch must be one of 8, 4, 2, 1")
    return EFFECTIVE_BATCH // microbatch


def assert_za_shape(za: torch.Tensor) -> None:
    if za.ndim != 5 or tuple(za.shape[1:]) != (256, 12, 12, 12):
        raise ValueError(f"Unexpected Za shape: {tuple(za.shape)}")


def _hash_checked(path: Path, expected: str, label: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")
    return actual


def _load_contract(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    _hash_checked(path, CONTRACT_SHA256, "D0-A normalization contract")
    payload = json.loads(path.read_text())
    channels = payload.get("channels")
    if payload.get("channel_count") != 256 or not isinstance(channels, list) or [c.get("channel_index") for c in channels] != list(range(256)):
        raise ValueError("Invalid D0-A channel contract")
    means = torch.tensor([c["mean"] for c in channels], dtype=torch.float32)
    stds = torch.tensor([c["population_std"] for c in channels], dtype=torch.float32)
    if not torch.isfinite(means).all() or not torch.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("Invalid D0-A channel statistics")
    return means, stds


PASS_FLAGS = (
    "loss_finite", "gradient_finite", "all_za_shapes_valid", "all_output_shapes_valid",
    "source_gradient_free", "optimizer_step_performed", "optimizer_state_materialized",
    "memory_headroom_pass",
)
NON_SCIENTIFIC = dict(scientific_training=False, validation_used=False,
                      model_selection=False, scientific_interpretation=False)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_seed(row, purpose):
    return int(fingerprint([PREFLIGHT_SEED, row["case_id"], int(row["patch_index"]), purpose])[:16], 16) & ((1 << 63) - 1)


def select_effective_batch(ids, records):
    selected = select_cases(ids, seed=PREFLIGHT_SEED, count=EFFECTIVE_BATCH)
    rows = select_manifest(records, selected)
    return [dict(min((r for r in rows if r["case_id"] == case),
                     key=lambda r: (sample_seed(r, "patch_selection"), int(r["patch_index"]))))
            for case in selected]


def randomness_metadata(rows):
    metadata = []
    for row in rows:
        seed = sample_seed(row, "timestep")
        timestep = int(torch.randint(1000, (1,), generator=torch.Generator().manual_seed(seed)).item())
        metadata.append(dict(case_id=row["case_id"], patch_index=int(row["patch_index"]),
                             timestep_seed=seed, noise_seed=sample_seed(row, "epsilon"), timestep=timestep))
    return metadata


def sample_randomness(rows, shape, device):
    metadata = randomness_metadata(rows)
    # CPU generators make each sample independent of CUDA kernels and batch partitioning.
    noise = torch.stack([torch.randn(shape, generator=torch.Generator().manual_seed(m["noise_seed"]))
                         for m in metadata]).to(device)
    return torch.tensor([m["timestep"] for m in metadata], device=device), noise


def validate_output_shape(prediction, z_batch):
    assert_za_shape(z_batch)
    if prediction.shape != z_batch.shape:
        raise ValueError("Output must match its Za microbatch shape")


def source_gradient_free(source):
    return not source.training and all(not p.requires_grad and p.grad is None for p in source.parameters())


def candidate_pass(result):
    return (not result["oom"] and result.get("failure_reason") is None
            and result["effective_batch_size"] == EFFECTIVE_BATCH
            and all(result[k] for k in PASS_FLAGS))


def verify_d0b3(decision_path):
    decision_hash = _hash_checked(decision_path, D0B3_DECISION_SHA256, "D0-B3 decision")
    manifest = decision_path.parent.parent / "MANIFEST.sha256"
    entries = [line.split(maxsplit=1) for line in manifest.read_text().splitlines() if line.strip()]
    matches = [digest for digest, name in entries if (manifest.parent / name).resolve() == decision_path.resolve()]
    if matches != [decision_hash]:
        raise ValueError("D0-B3 decision does not match MANIFEST entry")
    if json.loads(decision_path.read_text()).get("decision", {}).get("selected_operator") != "local_empirical_resample":
        raise ValueError("D0-B3 frozen decision does not select local_empirical_resample")
    return dict(d0b3_decision_sha256=decision_hash, d0b3_manifest_path=str(manifest),
                d0b3_manifest_decision_verified=True, d0b3_manifest_sha256=file_sha256(manifest))


def prepare_microbatch(rows, dataset_root, source, means, stds, device):
    patches, supports = [], []
    for row in rows:
        image, raw_mask = load_case(dataset_root, row["case_id"])
        normalized = normalize_nonzero(image)
        slices = tuple(slice(int(row[f"crop_start_{axis}"]), int(row[f"crop_stop_{axis}"])) for axis in "dhw")
        raw_patch = image[(slice(None), *slices)]
        if tuple(raw_patch.shape) != (4, 96, 96, 96):
            raise ValueError(f"Expected raw MRI patch [4,96,96,96]; got {tuple(raw_patch.shape)}")
        raw_support = support_mask(raw_patch)
        abnormal = torch.isin(raw_mask[slices], torch.tensor([1, 2, 3, 4]))
        pools, _ = donor_pools(raw_patch, abnormal, raw_support)
        patches.append(corrupt_input(normalized[(slice(None), *slices)], abnormal, pools,
                                     "local_empirical_resample", global_seed=PREFLIGHT_SEED,
                                     case_id=row["case_id"], patch_index=int(row["patch_index"])))
        supports.append(block_occupancy(raw_support).ge(0.50))
    with torch.no_grad():
        za = extract_za(source, torch.stack(patches).float().to(device))
        z0 = (za.float() - means[None, :, None, None, None]) / stds[None, :, None, None, None]
    return z0, torch.stack(supports)[:, None].float().to(device)


def run(microbatch: int, *, output_json: str | Path, config_path: str | Path = "configs/data.yaml", patch_manifest: str | Path = "outputs/frozen/d0b_support_mask_audit/patch_manifest.csv", checkpoint: str | Path = "outputs/frozen/e1c_anatomy_constraint/training/best.pt", contract: str | Path = "outputs/frozen/d0_latent_distribution_audit/d0_za_normalization_contract.json", device: str = "cuda") -> dict:
    destination = Path(output_json).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    accum = validate_microbatch(microbatch)
    provenance = runtime_git_provenance()
    validate_runtime_branch(provenance["runtime_git_branch"])
    manifest_path = Path(patch_manifest).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_sha256 = _hash_checked(manifest_path, PATCH_MANIFEST_SHA256, "D0-B0 patch manifest")
    checkpoint_sha256 = _hash_checked(checkpoint_path, CHECKPOINT_SHA256, "E1-C checkpoint")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D0-B4a real preflight requires CUDA; CPU is for unit tests only")
    random.seed(PREFLIGHT_SEED)
    np.random.seed(PREFLIGHT_SEED)
    torch.manual_seed(PREFLIGHT_SEED)
    root = Path(__file__).resolve().parents[1]
    config = read_config(Path(config_path))
    train_split = Path(config["train_cases"])
    if not train_split.is_absolute():
        train_split = root / train_split
    _hash_checked(train_split, TRAIN_SHA256, "training split")
    integrity = verify_d0b3(root / D0B3_DECISION)
    ids = read_case_ids(train_split)
    rows = select_effective_batch(ids, read_manifest(manifest_path, set(ids)))
    means, stds = _load_contract(Path(contract))
    metadata = randomness_metadata(rows)
    result = dict(schema="d0b4_diffusion_preflight", schema_version=2,
                  timestamp_utc=datetime.now(timezone.utc).isoformat(), **provenance, **integrity,
                  **NON_SCIENTIFIC, **{k: False for k in PASS_FLAGS},
                  expected_branch=EXPECTED_BRANCH, seed=PREFLIGHT_SEED,
                  train_split_sha256=TRAIN_SHA256, normalization_contract_sha256=CONTRACT_SHA256,
                  patch_manifest_path=str(manifest_path), patch_manifest_sha256=manifest_sha256,
                  source_checkpoint_path=str(checkpoint_path), source_checkpoint_sha256=checkpoint_sha256,
                  d0b3_selected_operator="local_empirical_resample", dataset_root=config["root"],
                  selected_cases=[r["case_id"] for r in rows], patch_coordinates=rows,
                  effective_batch_fingerprint=fingerprint(rows), sample_randomness=metadata,
                  randomness_fingerprint=fingerprint(metadata),
                  microbatch=microbatch, gradient_accumulation_steps=accum, effective_batch_size=len(rows),
                  architecture=dict(latent_channels=256, latent_shape=[12, 12, 12], base_channels=[64, 128, 256],
                                    decoder_channels=[128, 64], residual_blocks_per_level=2, attention=False, dropout=False),
                  diffusion_schedule=dict(timesteps=1000, cosine_offset=0.008, max_beta=0.999),
                  precision_contract=dict(source_extraction="FP32", standardization="FP32", diffusion="CUDA AMP FP16", master_parameters="FP32"),
                  torch_version=torch.__version__, python_version=platform.python_version(), cuda_version=torch.version.cuda,
                  step_shapes=[], losses=[], oom=False, failure_stage=None, failure_reason=None)
    stage = "cuda_initialization"
    started = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset before allocations so resident source/model/optimizer/EMA count toward peak.
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.manual_seed_all(PREFLIGHT_SEED)
        result["total_cuda_memory_bytes"] = total = torch.cuda.get_device_properties(device).total_memory
        result["gpu_name"] = torch.cuda.get_device_name(device)
        stage = "source_allocation"
        source, result["source_info"] = load_source(checkpoint_path, source_experiment="e1c", expected_sha256=CHECKPOINT_SHA256, expected_epoch=11)
        source.eval().requires_grad_(False).to(device)
        means, stds = means.to(device), stds.to(device)
        stage = "diffusion_optimizer_ema_allocation"
        model = LatentDiffusionUNet3D().to(device)
        result["model_parameter_count"] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
        # A post-step hook distinguishes a real optimizer update from a GradScaler skip.
        def mark_step(*_):
            result["optimizer_step_performed"] = True
        optimizer.register_step_post_hook(mark_step)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        ema = [p.detach().clone() for p in model.parameters()]
        schedule = {k: v.to(device) for k, v in cosine_schedule().items()}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result.update(loss_finite=True, all_za_shapes_valid=True, all_output_shapes_valid=True)
        for step in range(accum):
            stage = f"microbatch_{step}_extraction"
            batch_rows = rows[step * microbatch:(step + 1) * microbatch]
            z_batch, mask = prepare_microbatch(batch_rows, config["root"], source, means, stds, device)
            valid = z_batch.shape == (microbatch, 256, 12, 12, 12)
            result["all_za_shapes_valid"] &= valid
            if not valid:
                raise ValueError("Unexpected Za microbatch shape")
            stage = f"microbatch_{step}_diffusion"
            t, noise = sample_randomness(batch_rows, tuple(z_batch.shape[1:]), device)
            zt = q_sample(z_batch, t, mask, schedule, noise=noise)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                prediction = model(zt, t)
            result["all_output_shapes_valid"] &= prediction.shape == z_batch.shape
            validate_output_shape(prediction, z_batch)
            result["step_shapes"].append(dict(za=list(z_batch.shape), output=list(prediction.shape)))
            loss = masked_epsilon_mse(prediction.float(), noise * mask, mask)
            result["loss_finite"] &= bool(torch.isfinite(loss))
            result["losses"].append(float(loss.detach()) if bool(torch.isfinite(loss)) else None)
            if not result["loss_finite"]:
                raise ValueError("Non-finite loss")
            stage = f"microbatch_{step}_backward"
            scaler.scale(loss / accum).backward()
            del z_batch, mask, t, noise, zt, prediction, loss
        stage = "gradient_clipping"
        scaler.unscale_(optimizer)
        result["gradient_finite"] = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        result["gradient_finite"] &= bool(torch.isfinite(norm))
        if not result["gradient_finite"]:
            raise ValueError("Non-finite gradients")
        stage = "optimizer_step"
        scaler.step(optimizer)
        scaler.update()
        result["optimizer_state_materialized"] = all(
            p in optimizer.state and all(k in optimizer.state[p] for k in ("step", "exp_avg", "exp_avg_sq"))
            and optimizer.state[p]["step"].item() >= 1 for p in model.parameters())
        result["source_gradient_free"] = source_gradient_free(source)
        if not result["source_gradient_free"]:
            raise AssertionError("Frozen E1-C source acquired gradients")
        stage = "ema_update"
        with torch.no_grad():
            for e, p in zip(ema, model.parameters()):
                e.mul_(0.999).add_(p, alpha=0.001)
        stage = "memory_collection"
        torch.cuda.synchronize()
        result["elapsed_seconds"] = time.perf_counter() - started
        result.update(peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
                      current_memory_allocated_bytes=torch.cuda.memory_allocated(),
                      current_memory_reserved_bytes=torch.cuda.memory_reserved())
        result["peak_reserved_fraction"] = result["peak_memory_reserved_bytes"] / total
        result["peak_allocated_fraction"] = result["peak_memory_allocated_bytes"] / total
        result["memory_headroom_pass"] = result["peak_reserved_fraction"] <= 0.90
    except Exception as exc:
        result.update(oom=isinstance(exc, torch.cuda.OutOfMemoryError), failure_stage=stage,
                      failure_reason=f"{type(exc).__name__}: {exc}")
        try:
            result.update(peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())
            torch.cuda.empty_cache()
        except Exception:
            pass
    result["pass"] = candidate_pass(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--config", dest="config_path", default="configs/data.yaml")
    parser.add_argument("--patch-manifest", default="outputs/frozen/d0b_support_mask_audit/patch_manifest.csv")
    parser.add_argument("--checkpoint", default="outputs/frozen/e1c_anatomy_constraint/training/best.pt")
    parser.add_argument("--contract", default="outputs/frozen/d0_latent_distribution_audit/d0_za_normalization_contract.json")
    args = parser.parse_args()
    run(args.microbatch, output_json=args.output_json, config_path=args.config_path, patch_manifest=args.patch_manifest, checkpoint=args.checkpoint, contract=args.contract)


if __name__ == "__main__":
    main()
