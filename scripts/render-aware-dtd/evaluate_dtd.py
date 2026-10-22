from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aec_dataset import AecChallengeDtdDataset
from dtd_event_metrics import event_metrics
from dtd_metrics import (
    binary_metrics,
    low_fpr_operating_points,
    select_exact_threshold,
    select_threshold,
)
from dtd_model import DtdModelConfig, RenderAwareDtd
from dtd_training_objectives import apply_stream_resets
from enriched_dataset import DatasetItem, create_evaluation_suites


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def evaluation_seed_metadata(
    base_seed: int,
    *,
    enriched: bool,
) -> dict[str, object]:
    if not enriched:
        return {"legacy_aec_seed": base_seed}
    return {
        "base_seed": base_seed,
        "suite_seeds": {
            "aec_official": base_seed,
            "wenet_unseen": base_seed + 1,
            "aec_real_unseen": base_seed + 2,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a render-aware DTD checkpoint on fixed test suites."
    )
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--dataset-root",
        default=r"E:\Projects\render-aware-dtd\datasets\AEC-Challenge",
        help="Legacy AEC-only evaluation root.",
    )
    parser.add_argument(
        "--enriched-root",
        help="Manifest dataset root; enables AEC, Wenet, and real suites.",
    )
    parser.add_argument(
        "--suite",
        choices=("all", "aec_official", "wenet_unseen", "aec_real_unseen"),
        default="all",
    )
    parser.add_argument("--output")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--online-examples", type=int, default=2_000)
    parser.add_argument("--sequence-seconds", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--evaluation-seed",
        type=nonnegative_int,
        default=20260731,
        help=(
            "Base seed for deterministic evaluation suite rendering. "
            "Wenet and AEC Real use base+1 and base+2 respectively."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


def configure_cuda_backend(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.enabled = False
    F.conv1d(
        torch.zeros(1, 1, 640, device=device),
        torch.zeros(2, 1, 32, device=device),
    )
    torch.cuda.synchronize(device)


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[RenderAwareDtd, dict[str, object]]:
    resolved = Path(checkpoint_path).resolve()
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    model = RenderAwareDtd(DtdModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def first_sustained_run(
    mask: np.ndarray,
    minimum_frames: int,
) -> tuple[int, int] | None:
    start = -1
    for index, active in enumerate(mask):
        if active and start < 0:
            start = index
        elif not active:
            if start >= 0 and index - start >= minimum_frames:
                return start, index
            start = -1
    if start >= 0 and len(mask) - start >= minimum_frames:
        return start, len(mask)
    return None


def probability_only_event(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
    *,
    minimum_frames: int = 15,
    early_tolerance_frames: int = 5,
) -> dict[str, object]:
    oracle = first_sustained_run(target >= 0.5, minimum_frames)
    trigger = first_sustained_run(probability >= threshold, minimum_frames)
    if oracle is None:
        return {
            "eligible": False,
            "detected": trigger is not None,
            "early_false_trigger": trigger is not None,
            "latency_ms": None,
        }
    if trigger is None:
        return {
            "eligible": True,
            "detected": False,
            "early_false_trigger": False,
            "latency_ms": None,
        }
    oracle_start, _ = oracle
    _, trigger_end = trigger
    early = trigger_end < oracle_start - early_tolerance_frames
    return {
        "eligible": True,
        "detected": not early,
        "early_false_trigger": early,
        "latency_ms": (max(0, trigger_end - oracle_start) * 20 if not early else None),
    }


@torch.inference_mode()
def infer_dataset(
    model: RenderAwareDtd,
    dataset: Dataset[DatasetItem],
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    strata: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for batch in loader:
        microphone = batch["microphone"].to(device, non_blocking=True)
        render = batch["render"].to(device, non_blocking=True)
        target_tensor = batch["target"].to(device, non_blocking=True)
        reset_value = batch.get("reset_before")
        reset_tensor = (
            torch.zeros_like(target_tensor, dtype=torch.bool)
            if reset_value is None
            else reset_value.to(device, non_blocking=True, dtype=torch.bool)
        )
        logits, _ = model(microphone, render)
        logits = apply_stream_resets(
            model,
            logits,
            microphone,
            render,
            reset_tensor,
            getattr(getattr(model, "config", None), "hop_length", 320),
        )
        batch_probability = torch.sigmoid(logits).cpu().numpy()
        batch_target = batch["target"].numpy()
        probabilities.extend(batch_probability)
        targets.extend(batch_target)
        sources = batch.get(
            "source",
            ["aec_legacy"] * batch_probability.shape[0],
        )
        modes = batch.get("mode", ["mixture"] * batch_probability.shape[0])
        for index, (source, mode) in enumerate(zip(sources, modes)):
            pair = (batch_probability[index], batch_target[index])
            strata[f"source:{source}"].append(pair)
            strata[f"mode:{mode}"].append(pair)
    return (
        np.concatenate([value.reshape(-1) for value in probabilities]),
        np.concatenate([value.reshape(-1) for value in targets]),
        strata,
    )


def event_summary(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    threshold: float,
) -> dict[str, object]:
    events = [
        probability_only_event(probability, target, threshold)
        for probability, target in zip(probabilities, targets)
    ]
    eligible = [event for event in events if bool(event["eligible"])]
    detected = [event for event in eligible if bool(event["detected"])]
    early = [event for event in events if bool(event["early_false_trigger"])]
    latencies = np.asarray(
        [
            float(event["latency_ms"])
            for event in detected
            if event["latency_ms"] is not None
        ],
        dtype=np.float64,
    )
    return {
        "definition": "15 consecutive positive 20ms frames",
        "eligible_cases": len(eligible),
        "detected_cases": len(detected),
        "early_false_triggers": len(early),
        "recall": len(detected) / max(1, len(eligible)),
        "latency_p50_ms": (
            float(np.percentile(latencies, 50)) if len(latencies) else None
        ),
        "latency_p95_ms": (
            float(np.percentile(latencies, 95)) if len(latencies) else None
        ),
    }


@torch.inference_mode()
def evaluate_suite(
    model: RenderAwareDtd,
    dataset: Dataset[DatasetItem],
    device: torch.device,
    *,
    threshold: float,
    batch_size: int,
    num_workers: int,
) -> dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    per_case_probability: list[np.ndarray] = []
    per_case_target: list[np.ndarray] = []
    per_case_valid: list[np.ndarray] = []
    per_case_reset: list[np.ndarray] = []
    per_case_mode: list[str] = []
    per_case_cluster: list[str] = []
    strata: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for batch in loader:
        microphone = batch["microphone"].to(device, non_blocking=True)
        render = batch["render"].to(device, non_blocking=True)
        target_tensor = batch["target"].to(device, non_blocking=True)
        reset_value = batch.get("reset_before")
        reset_tensor = (
            torch.zeros_like(target_tensor, dtype=torch.bool)
            if reset_value is None
            else reset_value.to(device, non_blocking=True, dtype=torch.bool)
        )
        logits, _ = model(microphone, render)
        logits = apply_stream_resets(
            model,
            logits,
            microphone,
            render,
            reset_tensor,
            getattr(getattr(model, "config", None), "hop_length", 320),
        )
        batch_probability = torch.sigmoid(logits).cpu().numpy()
        batch_target = batch["target"].numpy()
        valid_value = batch.get("valid_mask")
        batch_valid = (
            np.ones_like(batch_target, dtype=bool)
            if valid_value is None
            else valid_value.numpy().astype(bool, copy=False)
        )
        batch_reset = reset_tensor.cpu().numpy().astype(bool, copy=False)
        sources = batch.get(
            "source",
            ["aec_legacy"] * batch_probability.shape[0],
        )
        modes = batch.get("mode", ["mixture"] * batch_probability.shape[0])
        clusters = batch.get("cluster_id", batch.get("file_id"))
        if clusters is None:
            clusters = [f"stream:{index}" for index in range(len(modes))]
        elif isinstance(clusters, torch.Tensor):
            clusters = clusters.cpu().tolist()
        for index, (source, mode, cluster_id) in enumerate(
            zip(sources, modes, clusters)
        ):
            probability = batch_probability[index].reshape(-1)
            target = batch_target[index].reshape(-1)
            valid = batch_valid[index].reshape(-1)
            reset = batch_reset[index].reshape(-1)
            per_case_probability.append(probability)
            per_case_target.append(target)
            per_case_valid.append(valid)
            per_case_reset.append(reset)
            per_case_mode.append(str(mode))
            per_case_cluster.append(str(cluster_id))
            pair = (probability[valid], target[valid])
            strata[f"source:{source}"].append(pair)
            strata[f"mode:{mode}"].append(pair)

    probability = np.concatenate(
        [value[valid] for value, valid in zip(per_case_probability, per_case_valid)]
    )
    target = np.concatenate(
        [value[valid] for value, valid in zip(per_case_target, per_case_valid)]
    )
    selected_threshold, selected_metrics = select_threshold(probability, target)
    _, exact_selected_metrics = select_exact_threshold(probability, target)
    report: dict[str, object] = {
        "examples": len(dataset),
        "checkpoint_threshold": binary_metrics(
            probability,
            target,
            threshold,
        ),
        "best_f1_operating_point": selected_metrics,
        "exact_best_f1_operating_point": exact_selected_metrics,
        "low_fpr_operating_points": low_fpr_operating_points(
            probability,
            target,
        ),
        "probability_only_event": event_summary(
            per_case_probability,
            per_case_target,
            selected_threshold,
        ),
        "production_event": event_metrics(
            per_case_probability,
            per_case_target,
            threshold,
            modes=per_case_mode,
            cluster_ids=per_case_cluster,
            valid_masks=per_case_valid,
            reset_masks=per_case_reset,
        ),
        "production_event_at_best_f1_threshold": event_metrics(
            per_case_probability,
            per_case_target,
            selected_threshold,
            modes=per_case_mode,
            cluster_ids=per_case_cluster,
            valid_masks=per_case_valid,
            reset_masks=per_case_reset,
        ),
        "strata": {},
    }
    report_strata = report["strata"]
    if not isinstance(report_strata, dict):
        raise AssertionError("report strata must be a dictionary")
    for key, values in sorted(strata.items()):
        stratum_probability = np.concatenate([value[0] for value in values])
        stratum_target = np.concatenate([value[1] for value in values])
        report_strata[key] = binary_metrics(
            stratum_probability,
            stratum_target,
            selected_threshold,
        )
    return report


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    configure_cuda_backend(device)
    checkpoint_path = Path(args.checkpoint).resolve()
    model, checkpoint = load_model(checkpoint_path, device)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint["decision_threshold"])
    )

    if args.enriched_root:
        suites = create_evaluation_suites(
            args.enriched_root,
            sequence_seconds=args.sequence_seconds,
            online_examples=args.online_examples,
            seed=args.evaluation_seed,
            limit=args.limit,
        )
        if args.suite != "all":
            suites = {args.suite: suites[args.suite]}
    else:
        legacy: Dataset[DatasetItem] = AecChallengeDtdDataset(
            args.dataset_root,
            "test",
            sequence_seconds=None,
            augment=False,
            seed=args.evaluation_seed,
            limit=args.limit,
        )
        suites = {"aec_official_legacy": legacy}

    # Legacy AEC evaluation keeps variable-length utterances. The default
    # collate function cannot stack those, so evaluate one case at a time.
    evaluation_batch_size = args.batch_size if args.enriched_root else 1
    evaluation_num_workers = args.num_workers if args.enriched_root else 0
    report = {
        "schema_version": "2.0",
        "checkpoint": str(checkpoint_path),
        "threshold_source": (
            "command_line" if args.threshold is not None else "validation_checkpoint"
        ),
        "evaluation": evaluation_seed_metadata(
            args.evaluation_seed,
            enriched=bool(args.enriched_root),
        ),
        "model_config": model.config.to_dict(),
        "complexity": model.complexity().to_dict(),
        "suites": {
            name: evaluate_suite(
                model,
                dataset,
                device,
                threshold=threshold,
                batch_size=evaluation_batch_size,
                num_workers=evaluation_num_workers,
            )
            for name, dataset in suites.items()
        },
    }
    safe_report = json_safe(report)
    output = (
        Path(args.output).resolve()
        if args.output
        else checkpoint_path.with_name("test_metrics.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(safe_report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "checkpoint": str(checkpoint_path),
                "suites": {
                    name: {
                        "grid": suite["best_f1_operating_point"],
                        "exact": suite["exact_best_f1_operating_point"],
                    }
                    for name, suite in safe_report["suites"].items()
                },
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
