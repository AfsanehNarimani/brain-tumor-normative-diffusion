"""Memorize cached real BraTS patches as a sanity check, not a full training run."""
import argparse
import math
import torch
from datasets.brats_dataset import BraTSDataset
from datasets.label_mapping import TISSUE_NAMES
from evaluation.metrics import region_dice
from losses.segmentation_loss import SegmentationLoss
from models.baseline_unet3d import BaselineUNet3D


def cache_patches(data_root: str, case_count: int, seed: int) -> list[dict]:
    """Select first foreground-containing cases in sorted order, loading once.

    A seeded CPU generator fixes one 96³ foreground-centered patch per case.
    Missing individual tissues are allowed; entirely empty cases are skipped.
    """
    if case_count < 1:
        raise ValueError("case-count must be positive.")
    dataset = BraTSDataset(root=data_root, training=True,
                          foreground_sampling_probability=1.0,
                          generator=torch.Generator().manual_seed(seed))
    if case_count > len(dataset):
        raise ValueError(f"Requested {case_count} cases but dataset contains {len(dataset)}.")
    cached = []
    for index in range(len(dataset)):
        sample = dataset[index]
        counts = sample["target"].sum((1, 2, 3))
        if not counts.any():
            print(f"Skipping empty foreground: {sample['case_id']}", flush=True)
            continue
        # Clone only patch data, freeing any full-volume tensor storage/views.
        cached.append({"image": sample["image"].clone(), "target": sample["target"].clone()})
        summary = ", ".join(f"{name}={int(count)}" for name, count in zip(TISSUE_NAMES, counts))
        print(f"Cached {sample['case_id']} crop_start={sample['crop_start']}: {summary}", flush=True)
        if len(cached) == case_count:
            return cached
    raise ValueError(f"Found only {len(cached)} foreground-containing cases; requested {case_count}.")


@torch.no_grad()
def measure(model, patches, criterion, device) -> tuple[float, torch.Tensor]:
    """Mean per-case loss and Dice on precisely the cached optimization samples."""
    was_training = model.training
    model.eval()
    losses, scores = [], []
    try:
        for patch in patches:
            target = patch["target"].unsqueeze(0).to(device)
            logits = model(patch["image"].unsqueeze(0).to(device))
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite tiny-batch loss.")
            losses.append(loss.item())
            scores.append(region_dice(logits, target)[0, :len(TISSUE_NAMES)].cpu())
    finally:
        model.train(was_training)
    return sum(losses) / len(losses), torch.stack(scores).mean(0)


def optimization_step(model, patches, criterion, optimizer, device) -> None:
    """One Adam update using the mean gradient across all fixed patches.

    Accumulating one case at a time reduces peak memory versus a stacked batch.
    GroupNorm has no running batch statistics, so this preserves the objective.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for patch in patches:
        logits = model(patch["image"].unsqueeze(0).to(device))
        target = patch["target"].unsqueeze(0).to(device)
        loss = criterion(logits, target) / len(patches)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite optimization loss.")
        loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Non-finite optimization gradients.")
    optimizer.step()


def loss_reduction(initial: float, final: float) -> tuple[float, bool]:
    """Heuristic success: strictly more than 50% reduction, independent of Dice."""
    if not math.isfinite(initial) or not math.isfinite(final) or initial <= 0 or final < 0:
        raise ValueError("Expected finite nonnegative losses and positive initial loss.")
    reduction = 100 * (initial - final) / initial
    return reduction, reduction > 50


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--case-count", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu, cuda, or mps")
    args = parser.parse_args()
    if args.steps < 1 or args.case_count < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("steps, case-count, and learning-rate must be positive and finite")
    torch.manual_seed(args.seed)
    # Reproducible same-device execution; unsupported deterministic operations
    # raise explicitly rather than silently switching to nondeterministic ones.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    patches = cache_patches(args.data_root, args.case_count, args.seed)
    model = BaselineUNet3D().to(device)
    criterion = SegmentationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial, _ = measure(model, patches, criterion, device)
    print(f"Initial loss: {initial:.6f}", flush=True)
    print("Dice is mean per-case volumetric Dice; both-empty=1, one-empty=0. Absent tissues are allowed.", flush=True)
    for step in range(1, args.steps + 1):
        optimization_step(model, patches, criterion, optimizer, device)
        if step % 10 == 0 or step == args.steps:
            final, scores = measure(model, patches, criterion, device)
            details = " ".join(f"{name}={score:.4f}" for name, score in zip(TISSUE_NAMES, scores))
            print(f"step={step} total_loss={final:.6f} Dice: {details}", flush=True)
    reduction, success = loss_reduction(initial, final)
    print(f"Initial loss: {initial:.6f}\nFinal loss: {final:.6f}\nLoss reduction: {reduction:.2f}%")
    print("SUCCESS: tiny-batch memorization criterion met (>50% loss reduction)." if success else
          "NOT YET MEMORIZED: loss reduction did not exceed 50%; inspect the loss trajectory.", flush=True)


if __name__ == "__main__":
    main()
