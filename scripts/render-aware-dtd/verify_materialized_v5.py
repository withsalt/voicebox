from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from aec_dataset import HOP_LENGTH, SAMPLE_RATE
from enriched_dataset import labels_for, load_jsonl
from materialized_v5_dataset import MaterializedV5Dataset
from prepare_materialized_v5 import (
    V5MaterializationConfig,
    generate_example,
    load_source_pools,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify materialized DTD v5.")
    parser.add_argument(
        "--root",
        default=r"E:\Projects\render-aware-dtd\datasets\render-aware-dtd-v5",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--loader-batches", type=int, default=16)
    return parser.parse_args()


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_row(
    root: Path,
    row: dict[str, object],
) -> dict[str, object]:
    path = root / str(row["audio"])
    try:
        signals, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as error:
        return {"file_id": row["file_id"], "error": f"read: {error}"}
    expected_samples = SAMPLE_RATE * 8
    if sample_rate != SAMPLE_RATE or signals.shape != (expected_samples, 3):
        return {
            "file_id": row["file_id"],
            "error": f"format: rate={sample_rate}, shape={signals.shape}",
        }
    if not np.isfinite(signals).all():
        return {"file_id": row["file_id"], "error": "non-finite audio"}
    microphone, render, target = signals.T
    mode = str(row["mode"])
    target_peak = float(np.max(np.abs(target), initial=0.0))
    render_peak = float(np.max(np.abs(render), initial=0.0))
    microphone_peak = float(np.max(np.abs(microphone), initial=0.0))
    if microphone_peak <= 1e-5:
        return {"file_id": row["file_id"], "error": "silent microphone"}
    if mode == "far_only" and target_peak > 1e-5:
        return {"file_id": row["file_id"], "error": "far-only target is active"}
    if mode == "near_only" and render_peak > 1e-5:
        return {"file_id": row["file_id"], "error": "near-only render is active"}
    if mode in {"near_only", "mixture"} and target_peak <= 1e-5:
        return {"file_id": row["file_id"], "error": "near target is silent"}
    if mode in {"far_only", "mixture"} and render_peak <= 1e-5:
        return {"file_id": row["file_id"], "error": "far render is silent"}

    gap_hops = int(row.get("gap_hops", 0))
    if gap_hops:
        padding = gap_hops * HOP_LENGTH
        if np.any(signals[-padding:] != 0.0):
            return {"file_id": row["file_id"], "error": "reset tail is not zero"}
    cold_onset_delay_hops: int | None = None
    if str(row.get("reset_variant")) == "cold_onset":
        reset_frame = int(row["reset_frame"])
        labels = labels_for(target)
        active = np.flatnonzero(labels > 0.5)
        if not len(active):
            return {"file_id": row["file_id"], "error": "cold onset has no label"}
        first_active = int(active[0])
        if first_active < reset_frame - 1 or first_active > reset_frame + 19:
            return {
                "file_id": row["file_id"],
                "error": (
                    f"cold onset outside reset window: reset={reset_frame}, "
                    f"active={first_active}"
                ),
            }
        cold_onset_delay_hops = first_active - reset_frame
    return {
        "file_id": row["file_id"],
        "bytes": path.stat().st_size,
        "clipped_samples": int(np.count_nonzero(np.abs(signals) >= 0.9999)),
        "total_samples": int(signals.size),
        "cold_onset_delay_hops": cold_onset_delay_hops,
    }


def validate_split(
    root: Path,
    split: str,
    workers: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = load_jsonl(root / "manifests" / f"materialized_{split}.jsonl")
    file_ids = [str(row["file_id"]) for row in rows]
    audio_paths = [str(row["audio"]) for row in rows]
    if len(file_ids) != len(set(file_ids)) or len(audio_paths) != len(set(audio_paths)):
        raise ValueError(f"duplicate materialized identity in {split}")
    started = time.perf_counter()
    failures: list[dict[str, object]] = []
    bytes_total = 0
    clipped_samples = 0
    total_samples = 0
    onset_delays: list[int] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for completed, result in enumerate(
            executor.map(lambda row: validate_row(root, row), rows), start=1
        ):
            if "error" in result:
                failures.append(result)
            else:
                bytes_total += int(result["bytes"])
                clipped_samples += int(result["clipped_samples"])
                total_samples += int(result["total_samples"])
                delay = result["cold_onset_delay_hops"]
                if delay is not None:
                    onset_delays.append(int(delay))
            if completed % 5_000 == 0 or completed == len(rows):
                print(
                    json.dumps(
                        {
                            "verify_split": split,
                            "completed": completed,
                            "total": len(rows),
                            "failures": len(failures),
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    duration = time.perf_counter() - started
    statistics = {
        "examples": len(rows),
        "bytes": bytes_total,
        "seconds": duration,
        "audio_mib_per_second": bytes_total / (1024**2) / duration,
        "clipped_sample_fraction": (
            clipped_samples / total_samples if total_samples else math.nan
        ),
        "cold_onset_examples": len(onset_delays),
        "cold_onset_delay_ms": {
            "minimum": min(onset_delays) * 20 if onset_delays else None,
            "maximum": max(onset_delays) * 20 if onset_delays else None,
            "mean": float(np.mean(onset_delays)) * 20 if onset_delays else None,
        },
    }
    return statistics, failures


def verify_but_leakage(root: Path) -> dict[str, bool]:
    values = {
        field: {split: set() for split in ("train", "validation", "test")}
        for field in ("utterance_id", "speaker", "room")
    }
    for split in ("train", "validation", "test"):
        path = root / "manifests" / f"but_replayed_{split}.jsonl"
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                for field in values:
                    values[field][split].add(str(row[field]))
    checks: dict[str, bool] = {}
    for field, splits in values.items():
        checks[field] = not (
            splits["train"] & splits["validation"]
            or splits["train"] & splits["test"]
            or splits["validation"] & splits["test"]
        )
    return checks


def loader_probe(
    root: Path,
    workers: int,
    batches: int,
) -> dict[str, object]:
    dataset = MaterializedV5Dataset(
        root,
        "materialized_train",
        sequence_seconds=8,
    )
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=workers,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    digest = hashlib.sha256()
    started = time.perf_counter()
    examples = 0
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        for name in ("microphone", "render", "target", "valid_mask", "reset_before"):
            tensor = batch[name]
            digest.update(tensor.numpy().tobytes())
        examples += int(batch["microphone"].shape[0])
    seconds = time.perf_counter() - started
    return {
        "workers": workers,
        "batches": batches,
        "examples": examples,
        "seconds": seconds,
        "examples_per_second": examples / seconds,
        "sha256": digest.hexdigest(),
    }


def regeneration_probe(root: Path) -> dict[str, object]:
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    config = V5MaterializationConfig(**summary["configuration"])
    probe_root = root / "verification_regeneration"
    results: dict[str, object] = {}
    for split, indices in (("train", (0, 997, 47_999)), ("validation", (0, 1499))):
        pools = load_source_pools(root, split)
        split_results: list[dict[str, object]] = []
        for index in indices:
            row = generate_example(probe_root, pools, split, index, config)
            regenerated = probe_root / str(row["audio"])
            original = root / str(row["audio"])
            actual = sha256_file(regenerated)
            expected = sha256_file(original)
            split_results.append(
                {
                    "index": index,
                    "original_sha256": expected,
                    "regenerated_sha256": actual,
                    "match": actual == expected,
                }
            )
        results[split] = split_results
    return results


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    content: dict[str, object] = {}
    failures: list[dict[str, object]] = []
    for split in ("train", "validation"):
        statistics, split_failures = validate_split(root, split, args.workers)
        content[split] = statistics
        failures.extend(split_failures)

    base_hashes: dict[str, bool] = {}
    v4_root = Path(summary["roots"]["v4"])
    for name, expected in summary["base_v4_manifest_hashes"].items():
        base_hashes[name] = (
            sha256_file(v4_root / "manifests" / name) == expected
            and sha256_file(root / "manifests" / name) == expected
        )
    leakage = verify_but_leakage(root)
    regeneration = regeneration_probe(root)
    loader = [
        loader_probe(root, workers, args.loader_batches)
        for workers in (0, args.workers)
    ]
    loader_deterministic = loader[0]["sha256"] == loader[1]["sha256"]
    regeneration_deterministic = all(
        bool(item["match"])
        for split_results in regeneration.values()
        for item in split_results  # type: ignore[union-attr]
    )
    passed = (
        not failures
        and all(base_hashes.values())
        and all(leakage.values())
        and loader_deterministic
        and regeneration_deterministic
    )
    report = {
        "passed": passed,
        "root": str(root),
        "full_audio_validation": content,
        "failures": failures,
        "v4_manifest_hashes_match": base_hashes,
        "but_split_leakage_checks": leakage,
        "regeneration": regeneration,
        "regeneration_deterministic": regeneration_deterministic,
        "loader_probes": loader,
        "loader_deterministic": loader_deterministic,
        "torch_version": torch.__version__,
    }
    atomic_write_json(root / "verification_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
