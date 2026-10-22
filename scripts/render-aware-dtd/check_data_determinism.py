from __future__ import annotations

import argparse
import hashlib

import torch

from enriched_dataset import create_enriched_datasets
from train_dtd import create_loader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify deterministic real-data batches across worker counts."
    )
    parser.add_argument("--enriched-root", required=True)
    parser.add_argument("--sequence-seconds", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epoch", type=int, default=7)
    parser.add_argument("--aec-narrow-weight", type=float, default=0.45)
    parser.add_argument("--aec-fullband-weight", type=float, default=0.20)
    parser.add_argument("--wenet-weight", type=float, default=0.15)
    parser.add_argument("--aec-real-weight", type=float, default=0.20)
    parser.add_argument("--wenet-far-only-probability", type=float, default=0.40)
    parser.add_argument("--v5-reset-probability", type=float, default=0.0)
    parser.add_argument("--v5-hard-far-only-probability", type=float, default=0.0)
    parser.add_argument("--v5-maximum-gap-hops", type=int, default=8)
    return parser.parse_args(argv)


def batch_digest(
    root: str,
    *,
    sequence_seconds: float,
    batch_size: int,
    batches: int,
    num_workers: int,
    seed: int,
    epoch: int,
    dataset_weights: tuple[float, float, float, float],
    wenet_far_only_probability: float,
    v5_reset_probability: float,
    v5_hard_far_only_probability: float,
    v5_maximum_gap_hops: int,
) -> str:
    examples = max(batch_size * (batches + 1), 32)
    dataset, _, _ = create_enriched_datasets(
        root,
        sequence_seconds=sequence_seconds,
        examples_per_epoch=examples,
        validation_examples=1,
        seed=seed,
        dataset_weights=dataset_weights,
        wenet_far_only_probability=wenet_far_only_probability,
        v5_reset_probability=v5_reset_probability,
        v5_hard_far_only_probability=v5_hard_far_only_probability,
        v5_maximum_gap_hops=v5_maximum_gap_hops,
    )
    dataset.set_epoch(epoch)
    generator = torch.Generator()
    generator.manual_seed(seed + epoch * 10_000_019)
    loader = create_loader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
    )
    digest = hashlib.sha256()
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        for key in (
            "microphone",
            "render",
            "target",
            "valid_mask",
            "reset_before",
            "stream_epoch",
            "gap_hops",
        ):
            digest.update(batch[key].contiguous().numpy().tobytes())
        for key in ("source", "mode", "file_id", "cluster_id"):
            digest.update("\0".join(batch[key]).encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    common = {
        "root": args.enriched_root,
        "sequence_seconds": args.sequence_seconds,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "seed": args.seed,
        "epoch": args.epoch,
        "dataset_weights": (
            args.aec_narrow_weight,
            args.aec_fullband_weight,
            args.wenet_weight,
            args.aec_real_weight,
        ),
        "wenet_far_only_probability": args.wenet_far_only_probability,
        "v5_reset_probability": args.v5_reset_probability,
        "v5_hard_far_only_probability": args.v5_hard_far_only_probability,
        "v5_maximum_gap_hops": args.v5_maximum_gap_hops,
    }
    single_worker_digest = batch_digest(num_workers=0, **common)
    multi_worker_digest = batch_digest(num_workers=args.num_workers, **common)
    repeated_digest = batch_digest(num_workers=args.num_workers, **common)
    print(f"workers=0 sha256={single_worker_digest}")
    print(f"workers={args.num_workers} sha256={multi_worker_digest}")
    print(f"workers={args.num_workers} repeat_sha256={repeated_digest}")
    if len({single_worker_digest, multi_worker_digest, repeated_digest}) != 1:
        raise RuntimeError("real-data batches are not deterministic")
    print("determinism_check=PASS")


if __name__ == "__main__":
    main()
