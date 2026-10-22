from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


LOW_FPR_KEYS = ("0.100%", "0.500%", "1.000%", "2.000%", "5.000%")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize 256x3 training, resources, and retry records."
    )
    parser.add_argument(
        "--artifact-root",
        default=r"E:\Projects\render-aware-dtd\artifacts-v4",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=list(range(7, 13)),
    )
    parser.add_argument("--selection")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def finite_float(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validation_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record["validation"]
    return {
        "selection_score": finite_float(
            record["selection_score"],
            "selection score",
        ),
        "average_precision": finite_float(
            metrics["average_precision"],
            "average precision",
        ),
        "roc_auc": finite_float(metrics["roc_auc"], "roc auc"),
        "grid_best_f1": finite_float(metrics["f1"], "grid best f1"),
        "recall_at_fpr": {
            key: finite_float(
                metrics["low_fpr_operating_points"][key]["recall"],
                f"recall@{key}",
            )
            for key in LOW_FPR_KEYS
        },
    }


def subtract_snapshots(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: left[key] - right[key]
        for key in (
            "selection_score",
            "average_precision",
            "roc_auc",
            "grid_best_f1",
        )
    } | {
        "recall_at_fpr": {
            key: left["recall_at_fpr"][key] - right["recall_at_fpr"][key]
            for key in LOW_FPR_KEYS
        }
    }


def overfit_assessment(
    history: list[dict[str, Any]],
    best_index: int,
) -> dict[str, Any]:
    best = validation_snapshot(history[best_index])
    last = validation_snapshot(history[-1])
    decline = best["selection_score"] - last["selection_score"]
    final_window_start = max(0, len(history) - 20)
    if decline > 0.005 and best_index < final_window_start:
        judgment = "material_late_regression"
    elif decline <= 0.002 or best_index >= final_window_start:
        judgment = "no_material_overfit"
    else:
        judgment = "mild_late_regression"
    trailing_scores = [
        finite_float(record["selection_score"], "selection score")
        for record in history[-10:]
    ]
    previous_scores = [
        finite_float(record["selection_score"], "selection score")
        for record in history[-20:-10]
    ]
    return {
        "judgment": judgment,
        "rule": (
            "material if best precedes final 20 epochs and last score is "
            ">0.005 below best; no-material if decline <=0.002 or best is "
            "within final 20; otherwise mild"
        ),
        "best_epoch": best_index,
        "last_epoch": int(history[-1]["epoch"]),
        "last_minus_best": subtract_snapshots(last, best),
        "last_10_score_mean": sum(trailing_scores) / len(trailing_scores),
        "previous_10_score_mean": (
            sum(previous_scores) / len(previous_scores)
            if previous_scores
            else None
        ),
    }


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def resource_summary(directory: Path) -> dict[str, Any]:
    logs = sorted(directory.glob("resource-monitor*.csv"))
    peak_commit = 0
    peak_tree = 0
    peak_gpu = 0
    minimum_available: int | None = None
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    monitored_seconds = 0.0
    samples = 0
    details: list[dict[str, Any]] = []
    for path in logs:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            details.append(
                {
                    "path": str(path),
                    "samples": 0,
                    "sha256": sha256_file(path),
                }
            )
            continue
        timestamps = [parse_timestamp(row["timestamp"]) for row in rows]
        log_first = min(timestamps)
        log_last = max(timestamps)
        monitored_seconds += (log_last - log_first).total_seconds()
        first_timestamp = (
            log_first
            if first_timestamp is None
            else min(first_timestamp, log_first)
        )
        last_timestamp = (
            log_last
            if last_timestamp is None
            else max(last_timestamp, log_last)
        )
        samples += len(rows)
        peak_commit = max(
            peak_commit,
            max(int(row["percent_committed"]) for row in rows),
        )
        peak_tree = max(
            peak_tree,
            max(int(row["process_tree_working_set_bytes"]) for row in rows),
        )
        peak_gpu = max(
            peak_gpu,
            max(int(row["gpu_used_mib"]) for row in rows),
        )
        log_minimum_available = min(
            int(row["available_bytes"]) for row in rows
        )
        minimum_available = (
            log_minimum_available
            if minimum_available is None
            else min(minimum_available, log_minimum_available)
        )
        details.append(
            {
                "path": str(path),
                "samples": len(rows),
                "first_timestamp": log_first.isoformat(),
                "last_timestamp": log_last.isoformat(),
                "duration_seconds": (log_last - log_first).total_seconds(),
                "sha256": sha256_file(path),
            }
        )
    return {
        "logs": details,
        "sample_count": samples,
        "first_timestamp": (
            first_timestamp.isoformat() if first_timestamp else None
        ),
        "last_timestamp": (
            last_timestamp.isoformat() if last_timestamp else None
        ),
        "wall_span_seconds": (
            (last_timestamp - first_timestamp).total_seconds()
            if first_timestamp and last_timestamp
            else 0.0
        ),
        "monitored_seconds": monitored_seconds,
        "peak_commit_percent": peak_commit,
        "peak_process_tree_working_set_bytes": peak_tree,
        "peak_process_tree_working_set_gib": peak_tree / 2**30,
        "minimum_available_bytes": minimum_available,
        "minimum_available_gib": (
            minimum_available / 2**30
            if minimum_available is not None
            else None
        ),
        "peak_gpu_used_mib": peak_gpu,
    }


def attempt_summary(directory: Path) -> list[dict[str, Any]]:
    stdout_paths = sorted(directory.glob("training*.stdout.log"))
    attempts: list[dict[str, Any]] = []
    for stdout in stdout_paths:
        prefix = stdout.name.removesuffix(".stdout.log")
        stderr = directory / f"{prefix}.stderr.log"
        attempts.append(
            {
                "name": prefix,
                "stdout": str(stdout),
                "stdout_bytes": stdout.stat().st_size,
                "stdout_sha256": sha256_file(stdout),
                "stderr": str(stderr) if stderr.exists() else None,
                "stderr_bytes": stderr.stat().st_size if stderr.exists() else None,
                "stderr_sha256": (
                    sha256_file(stderr) if stderr.exists() else None
                ),
                "is_resume": ".resume-" in prefix,
            }
        )
    return attempts


def artifact_hash(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def summarize_candidate(
    directory: Path,
    candidate: int,
    *,
    allow_incomplete: bool,
    selected_result: dict[str, Any] | None,
) -> dict[str, Any]:
    history_path = directory / "history.json"
    experiment_path = directory / "EXPERIMENT.json"
    if not history_path.is_file() or not experiment_path.is_file():
        if allow_incomplete:
            return {
                "candidate": candidate,
                "complete": False,
                "directory": str(directory),
                "reason": "history or experiment metadata missing",
            }
        raise FileNotFoundError(f"candidate-{candidate} is incomplete")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"candidate-{candidate} history is invalid")
    epochs = [int(record["epoch"]) for record in history]
    if epochs != list(range(len(history))):
        raise ValueError(f"candidate-{candidate} history is not contiguous")
    complete = len(history) == 150
    if not complete and not allow_incomplete:
        raise ValueError(f"candidate-{candidate} has {len(history)}/150 epochs")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    best_index = min(
        range(len(history)),
        key=lambda index: (
            -finite_float(
                history[index]["selection_score"],
                "selection score",
            ),
            index,
        ),
    )
    epoch_seconds = [
        finite_float(record["seconds"], "epoch seconds") for record in history
    ]
    checkpoint_count = len(
        list((directory / "checkpoints").glob("epoch-*.pt"))
    )
    best_artifact = artifact_hash(directory / "best.pt")
    last_artifact = artifact_hash(directory / "last.pt")
    best_checkpoint_epoch: int | None = None
    last_checkpoint_epoch: int | None = None
    if best_artifact is not None:
        best_checkpoint_epoch = int(
            torch.load(
                directory / "best.pt",
                map_location="cpu",
                weights_only=False,
            )["epoch"]
        )
    if last_artifact is not None:
        last_checkpoint_epoch = int(
            torch.load(
                directory / "last.pt",
                map_location="cpu",
                weights_only=False,
            )["epoch"]
        )
    attempts = attempt_summary(directory)
    result: dict[str, Any] = {
        "candidate": candidate,
        "complete": complete,
        "directory": str(directory),
        "changed_parameter_group": experiment["changed_parameter_group"],
        "recipe": experiment["recipe"],
        "epochs_completed": len(history),
        "checkpoint_count": checkpoint_count,
        "raw_best_epoch": best_index,
        "raw_best_validation": validation_snapshot(history[best_index]),
        "last_validation": validation_snapshot(history[-1]),
        "best_checkpoint_epoch": best_checkpoint_epoch,
        "last_checkpoint_epoch": last_checkpoint_epoch,
        "overfit": overfit_assessment(history, best_index),
        "training_time": {
            "epoch_seconds_total": sum(epoch_seconds),
            "epoch_seconds_mean": sum(epoch_seconds) / len(epoch_seconds),
            "epoch_seconds_min": min(epoch_seconds),
            "epoch_seconds_max": max(epoch_seconds),
            "epoch_seconds_p95": float(np_percentile(epoch_seconds, 95)),
        },
        "resources": resource_summary(directory),
        "attempts": attempts,
        "retry_count": sum(attempt["is_resume"] for attempt in attempts),
        "nonempty_stderr_count": sum(
            bool(attempt["stderr_bytes"]) for attempt in attempts
        ),
        "artifacts": {
            "experiment": artifact_hash(experiment_path),
            "history": artifact_hash(history_path),
            "training_summary": artifact_hash(
                directory / "training_summary.json"
            ),
            "best": best_artifact,
            "last": last_artifact,
        },
    }
    if selected_result is not None:
        result["validation_selection"] = selected_result
    return result


def np_percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    selection: dict[str, Any] | None = None
    selected_by_candidate: dict[int, dict[str, Any]] = {}
    if args.selection:
        selection_path = Path(args.selection).resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_by_candidate = {
            int(result["candidate"]): result
            for result in selection["candidate_results"]
        }
    candidates = [
        summarize_candidate(
            artifact_root / f"candidate-{candidate}",
            candidate,
            allow_incomplete=args.allow_incomplete,
            selected_result=selected_by_candidate.get(candidate),
        )
        for candidate in args.candidates
    ]
    report = {
        "schema_version": 1,
        "artifact_root": str(artifact_root),
        "selection": selection,
        "candidates": candidates,
    }
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite summary: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="x",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
        temporary_path = Path(stream.name)
    temporary_path.replace(output_path)
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
