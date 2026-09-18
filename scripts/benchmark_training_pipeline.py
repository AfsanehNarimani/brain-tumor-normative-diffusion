"""Benchmark the training pipeline without producing scientific results."""

import argparse
import statistics
import time

import torch
import yaml
from torch.utils.data import DataLoader

from losses.segmentation_loss import SegmentationLoss
from models.latent_probe import FrozenLatentProbe, load_source
from scripts.train_baseline import make_dataset, seed_worker


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def mean_ms(values):
    return statistics.mean(values) * 1000.0 if values else 0.0


def benchmark(
    config_path,
    batch_size,
    num_workers,
    iterations,
    warmup,
    device,
    pin_memory=False,
    persistent_workers=False,
    non_blocking=False,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(config["data_config"]) as f:
        data_config = yaml.safe_load(f)

    source, source_info = load_source(config["source_checkpoint"])

    model = FrozenLatentProbe(
        source,
        config["latent_source"],
    ).to(device)

    model.train()

    dataset = make_dataset(data_config, "train")

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(config["seed"]),
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers

    loader = DataLoader(dataset, **loader_kwargs)

    optimizer = torch.optim.AdamW(
        model.probe.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    criterion = SegmentationLoss()

    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    loader_wait_times = []
    transfer_times = []
    forward_times = []
    backward_times = []
    total_times = []

    iterator = iter(loader)

    for step in range(warmup + iterations):
        iteration_start = time.perf_counter()

        wait_start = time.perf_counter()

        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        wait_end = time.perf_counter()

        transfer_start = time.perf_counter()

        image = batch["image"].to(device, non_blocking=non_blocking)
        target = batch["target"].to(device, non_blocking=non_blocking)

        sync(device)

        transfer_end = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        forward_start = time.perf_counter()

        logits = model(image)
        loss = criterion(logits, target)

        sync(device)

        forward_end = time.perf_counter()

        backward_start = time.perf_counter()

        loss.backward()
        optimizer.step()

        sync(device)

        backward_end = time.perf_counter()

        iteration_end = time.perf_counter()

        if step >= warmup:
            loader_wait_times.append(wait_end - wait_start)
            transfer_times.append(transfer_end - transfer_start)
            forward_times.append(forward_end - forward_start)
            backward_times.append(backward_end - backward_start)
            total_times.append(iteration_end - iteration_start)

    samples = iterations * batch_size
    total_seconds = sum(total_times)

    if torch.device(device).type == "cuda":
        peak_memory_gb = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
    else:
        peak_memory_gb = 0.0

    result = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "non_blocking": non_blocking,
        "iterations": iterations,
        "warmup": warmup,
        "loader_wait_ms": mean_ms(loader_wait_times),
        "transfer_ms": mean_ms(transfer_times),
        "forward_ms": mean_ms(forward_times),
        "backward_optimizer_ms": mean_ms(backward_times),
        "iteration_ms": mean_ms(total_times),
        "samples_per_second": samples / total_seconds,
        "peak_gpu_memory_gb": peak_memory_gb,
        "source_checkpoint_sha256": source_info["sha256"],
    }

    print()
    print("=== TRAINING PIPELINE BENCHMARK ===")

    for key, value in result.items():
        print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/probe_disease.yaml",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--non-blocking", action="store_true")

    args = parser.parse_args()

    benchmark(
        args.config,
        args.batch_size,
        args.num_workers,
        args.iterations,
        args.warmup,
        args.device,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        non_blocking=args.non_blocking,
    )


if __name__ == "__main__":
    main()