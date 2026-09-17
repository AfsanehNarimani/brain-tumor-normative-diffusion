"""DEVELOPMENT/SANITY: fit both probes to one fixed real BraTS training patch."""
import argparse
import csv
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch

from losses.segmentation_loss import SegmentationLoss
from models.latent_probe import LatentProbeDecoder3D, PROBE_ARCHITECTURE, SOURCE_SHA256, load_source
from scripts.tiny_overfit_e1a import cache_patch, tensor_sha256
from scripts.train_latent_probe import read_config
from training.artifacts import environment_metadata, write_json

SOURCE_CHECKPOINT = Path("outputs/e1a_latent_split/best.pt")
SANITY_ROOT = Path("outputs/e1a_probe_tiny_overfit_sanity")
PATCH_SIZE = (96, 96, 96)


def extract_fixed_latents(source, image):
    """Run the encoder once and only the two bottleneck projectors."""
    source.eval()
    if any(p.requires_grad for p in source.parameters()):
        raise RuntimeError("E1-A source must be frozen before latent extraction")
    with torch.no_grad():
        bottleneck = source.encoder(image)[-1]
        anatomy = source.projection.anatomy_projector(bottleneck).detach()
        disease = source.projection.disease_projector(bottleneck).detach()
    if anatomy.shape != disease.shape or anatomy.requires_grad or disease.requires_grad:
        raise RuntimeError("Expected equal, detached frozen latents")
    return {"anatomy": anatomy, "disease": disease}


def state_snapshot(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def state_changed(before, module):
    return any(not torch.equal(value, module.state_dict()[name].detach().cpu())
               for name, value in before.items())


def fit_fixed_latent(latent, target, *, seed, steps, learning_rate, weight_decay,
                     history_path, source):
    """Optimize one fresh probe; history contains pre-update losses and final loss."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if latent.requires_grad or latent.ndim != 5 or latent.shape[1] != 256:
        raise ValueError("Expected detached [B,256,d,h,w] latent")
    if target.ndim != 5 or target.shape[1] != 4 or target.shape[0] != latent.shape[0]:
        raise ValueError("Expected aligned four-channel target")
    torch.manual_seed(seed)
    if latent.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    probe = LatentProbeDecoder3D().to(latent.device).train()
    initial_state = state_snapshot(probe)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = SegmentationLoss()
    spatial = tuple(target.shape[2:])
    losses = []
    probe_gradient_present = False
    all_losses_finite = True
    with Path(history_path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "loss_before_update", "learning_rate"])
        writer.writeheader()
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(latent, spatial), target)
            finite = bool(torch.isfinite(loss))
            all_losses_finite &= finite
            if not finite:
                raise RuntimeError(f"Non-finite probe loss at step {step}")
            loss.backward()
            gradients = [p.grad for p in probe.parameters()]
            if any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients):
                raise RuntimeError(f"Non-finite probe gradient at step {step}")
            probe_gradient_present |= any(g is not None and bool(g.abs().sum() > 0) for g in gradients)
            optimizer.step()
            value = loss.item()
            losses.append(value)
            writer.writerow(dict(step=step, loss_before_update=value, learning_rate=learning_rate))
            stream.flush()
            print(f"{Path(history_path).stem} step={step} loss={value:.6f}", flush=True)
    with torch.no_grad():
        final = criterion(probe(latent, spatial), target).item()
    all_losses_finite &= bool(np.isfinite(final))
    if not all_losses_finite:
        raise RuntimeError("Non-finite final probe loss")
    source_gradient_present = any(p.grad is not None for p in source.parameters())
    return dict(initial_state=initial_state, probe=probe,
                result=dict(initial_loss=losses[0], final_loss=final,
                            minimum_loss=min(*losses, final), steps=steps,
                            all_losses_finite=all_losses_finite,
                            source_gradient_present=source_gradient_present,
                            probe_gradient_present=probe_gradient_present,
                            probe_parameters_changed=state_changed(initial_state, probe),
                            loss_reduction=losses[0] - final,
                            loss_decreased=final < losses[0],
                            loss_trajectory=losses + [final]))


def run(*, data_root=None, train_cases=None, device="cuda", steps=50, seed=42,
        output_dir=str(SANITY_ROOT)):
    if steps < 1:
        raise ValueError("steps must be positive")
    output = Path(output_dir).expanduser().resolve()
    allowed = SANITY_ROOT.resolve()
    if output != allowed and allowed not in output.parents:
        raise ValueError(f"Sanity artifacts must stay under {allowed}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty sanity directory: {output}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    config = read_config("configs/data.yaml")
    root = data_root or config["root"]
    split = train_cases or config["train_cases"]
    if tuple(config["patch_size"]) != PATCH_SIZE:
        raise ValueError("Expected configured 96x96x96 BraTS patches")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.perf_counter()
    # No source path or identity override: the sanity CLI always verifies the exact best.pt.
    source, source_info = load_source(SOURCE_CHECKPOINT, expected_sha256=SOURCE_SHA256, expected_epoch=15)
    source = source.to(device).eval()
    source_before = state_snapshot(source)
    sample = cache_patch(root, split, seed)
    image, target = sample.pop("image").to(device), sample.pop("target").to(device)
    patch_hashes = dict(image_sha256=tensor_sha256(image), target_sha256=tensor_sha256(target))
    latents = extract_fixed_latents(source, image)
    expected_shape = (1, 256, *(size // 8 for size in PATCH_SIZE))
    if any(tuple(value.shape) != expected_shape for value in latents.values()):
        raise ValueError(f"Expected both E1-A latents to be {expected_shape}")
    latent_hashes = {key: tensor_sha256(value) for key, value in latents.items()}
    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    initial_states = {}
    for name in ("anatomy", "disease"):
        fitted = fit_fixed_latent(latents[name], target, seed=seed, steps=steps,
                                  learning_rate=0.001, weight_decay=0.00001,
                                  history_path=output / f"{name}_history.csv", source=source)
        initial_states[name] = fitted["initial_state"]
        outputs[name] = fitted["result"]
        del fitted
    identical = all(torch.equal(value, initial_states["disease"][key])
                    for key, value in initial_states["anatomy"].items())
    if not identical:
        raise RuntimeError("Anatomy and disease probes did not initialize identically")
    source_changed = state_changed(source_before, source)
    for item in outputs.values():
        item["source_parameters_changed"] = source_changed
    if source_changed or any(item["source_gradient_present"] for item in outputs.values()):
        raise RuntimeError("Frozen E1-A source changed or received gradients")
    if patch_hashes != dict(image_sha256=tensor_sha256(image), target_sha256=tensor_sha256(target)):
        raise RuntimeError("Fixed image or target changed")
    environment = environment_metadata(sum(v.numel() for v in initial_states["anatomy"].values()))
    summary = dict(artifact_type="DEVELOPMENT/SANITY; not a scientific probe result",
                   source_checkpoint=source_info, case_id=sample["case_id"],
                   crop_start=sample["crop_start"], patch_size=list(PATCH_SIZE),
                   data_root=str(Path(root).expanduser().resolve()),
                   train_cases=str(Path(split).expanduser().resolve()),
                   train_split_sha256=hashlib.sha256(Path(split).expanduser().read_bytes()).hexdigest(),
                   patch_hashes=patch_hashes, latent_hashes=latent_hashes,
                   foreground_voxel_count=int(target.bool().any(1).sum()),
                   normalize=True, foreground_sampling_probability=1.0,
                   augmentation=False, seed=seed, device=device, steps=steps,
                   optimizer="AdamW", learning_rate=0.001, weight_decay=0.00001,
                   loss="SegmentationLoss (Dice+BCE)", probe_architecture=PROBE_ARCHITECTURE,
                   probe_parameter_count=sum(v.numel() for v in initial_states["anatomy"].values()),
                   identical_initial_probe_states=identical,
                   source_parameters_changed=source_changed,
                   results=outputs, environment=environment,
                   duration_seconds=time.perf_counter() - started)
    write_json(output / "summary.json", summary)
    print(f"Anatomy loss {outputs['anatomy']['initial_loss']:.6f} -> {outputs['anatomy']['final_loss']:.6f}; "
          f"disease loss {outputs['disease']['initial_loss']:.6f} -> {outputs['disease']['final_loss']:.6f}. "
          "No scientific pass threshold.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="BraTS data root; defaults to configs/data.yaml")
    parser.add_argument("--train-cases", help="Fixed training split; defaults to configs/data.yaml")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(SANITY_ROOT))
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
