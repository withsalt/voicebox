from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_dtd import (
    configure_cuda_backend,
    evaluate_suite,
    json_safe,
    load_model,
    nonnegative_int,
)
from enriched_dataset import create_evaluation_suites


LOW_FPR_KEYS = ("0.100%", "0.500%", "1.000%", "2.000%", "5.000%")
AEC_METRIC_TOLERANCE = 0.002
AEC_RECALL_TOLERANCE = 0.015
WENET_TARGETS = {
    "average_precision": 0.850,
    "roc_auc": 0.920,
    "best_f1": 0.750,
    "recall_at_1.000%_fpr": 0.450,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a candidate DTD checkpoint with the existing production "
            "baseline on identical fixed evaluation suites."
        )
    )
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--enriched-root", required=True)
    parser.add_argument("--output", required=True)
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


def compared_metrics(suite: dict[str, object]) -> dict[str, float]:
    best = suite.get("exact_best_f1_operating_point")
    if best is None:
        best = suite["best_f1_operating_point"]
    low_fpr = suite["low_fpr_operating_points"]
    if not isinstance(best, dict) or not isinstance(low_fpr, dict):
        raise ValueError("suite metrics have an invalid shape")
    metrics = {
        "average_precision": float(best["average_precision"]),
        "roc_auc": float(best["roc_auc"]),
        "best_f1": float(best["f1"]),
    }
    for key in LOW_FPR_KEYS:
        point = low_fpr[key]
        if not isinstance(point, dict):
            raise ValueError(f"low-FPR point has an invalid shape: {key}")
        metrics[f"recall_at_{key}_fpr"] = float(point["recall"])
    return metrics


def best_f1_selector(suite: dict[str, object]) -> str:
    return (
        "exact_all_distinct_scores"
        if suite.get("exact_best_f1_operating_point") is not None
        else "legacy_0.01_grid"
    )


def compare_suite(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    policy: str,
) -> dict[str, object]:
    baseline_selector = best_f1_selector(baseline)
    candidate_selector = best_f1_selector(candidate)
    if baseline_selector != candidate_selector:
        raise ValueError(
            "baseline and candidate must use the same best-F1 selector: "
            f"baseline={baseline_selector}, candidate={candidate_selector}"
        )
    baseline_metrics = compared_metrics(baseline)
    candidate_metrics = compared_metrics(candidate)
    deltas = {
        name: candidate_metrics[name] - baseline_metrics[name]
        for name in baseline_metrics
    }
    report: dict[str, object] = {
        "best_f1_selector": baseline_selector,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": deltas,
    }
    if policy == "aec_non_regression":
        tolerances = {
            name: (
                AEC_RECALL_TOLERANCE
                if name.startswith("recall_at_")
                else AEC_METRIC_TOLERANCE
            )
            for name in deltas
        }
        checks = {
            name: delta + tolerances[name] >= -1e-12
            for name, delta in deltas.items()
        }
        report.update(
            {
                "policy": policy,
                "passed": all(checks.values()),
                "tolerances": tolerances,
                "checks": checks,
            }
        )
        return report
    if policy != "wenet_targets_or_pareto":
        raise ValueError(f"unsupported comparison policy: {policy}")

    target_checks = {
        name: candidate_metrics[name] >= target
        for name, target in WENET_TARGETS.items()
    }
    pareto_checks = {name: delta >= 0.0 for name, delta in deltas.items()}
    pareto_improved = all(pareto_checks.values()) and any(
        delta > 0.0 for delta in deltas.values()
    )
    targets_passed = all(target_checks.values())
    report.update(
        {
            "policy": policy,
            "passed": targets_passed or pareto_improved,
            "targets": WENET_TARGETS,
            "target_checks": target_checks,
            "targets_passed": targets_passed,
            "pareto_checks": pareto_checks,
            "pareto_improved": pareto_improved,
            "acceptance_basis": (
                "explicit_targets"
                if targets_passed
                else "strict_pareto"
                if pareto_improved
                else "failed"
            ),
        }
    )
    return report


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    configure_cuda_backend(device)
    baseline_model, baseline_checkpoint = load_model(args.baseline, device)
    candidate_model, candidate_checkpoint = load_model(args.candidate, device)
    suites = create_evaluation_suites(
        args.enriched_root,
        sequence_seconds=args.sequence_seconds,
        online_examples=args.online_examples,
        seed=args.evaluation_seed,
    )

    reports: dict[str, dict[str, object]] = {}
    comparisons: dict[str, dict[str, object]] = {}
    for name, dataset in suites.items():
        baseline_report = evaluate_suite(
            baseline_model,
            dataset,
            device,
            threshold=float(baseline_checkpoint["decision_threshold"]),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        candidate_report = evaluate_suite(
            candidate_model,
            dataset,
            device,
            threshold=float(candidate_checkpoint["decision_threshold"]),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        reports[name] = {
            "baseline": baseline_report,
            "candidate": candidate_report,
        }
        comparisons[name] = compare_suite(
            baseline_report,
            candidate_report,
            policy=(
                "wenet_targets_or_pareto"
                if name == "wenet_unseen"
                else "aec_non_regression"
            ),
        )

    baseline_complexity = baseline_model.complexity().to_dict()
    candidate_complexity = candidate_model.complexity().to_dict()
    complexity_delta = {
        key: float(candidate_complexity[key]) - float(baseline_complexity[key])
        for key in baseline_complexity
    }
    report = json_safe(
        {
            "schema_version": "2.0",
            "passed": all(item["passed"] for item in comparisons.values()),
            "baseline": str(Path(args.baseline).resolve()),
            "candidate": str(Path(args.candidate).resolve()),
            "dataset": str(Path(args.enriched_root).resolve()),
            "evaluation": {
                "base_seed": args.evaluation_seed,
                "suite_seeds": {
                    "aec_official": args.evaluation_seed,
                    "wenet_unseen": args.evaluation_seed + 1,
                    "aec_real_unseen": args.evaluation_seed + 2,
                },
                "online_examples": args.online_examples,
                "sequence_seconds": args.sequence_seconds,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            },
            "comparison_policy": {
                "aec_official_and_real": (
                    "AP/AUC/exact best F1 delta >= -0.002 and each "
                    "fixed-FPR recall delta >= -0.015"
                ),
                "wenet_unseen": (
                    "meet explicit AP/AUC/exact best F1/R@1%FPR targets, "
                    "or show strict Pareto improvement over every reported "
                    "baseline metric"
                ),
            },
            "complexity": {
                "informational_only": True,
                "baseline": baseline_complexity,
                "candidate": candidate_complexity,
                "delta": complexity_delta,
            },
            "comparisons": comparisons,
            "reports": reports,
        }
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": report["passed"],
                "comparisons": comparisons,
                "complexity": report["complexity"],
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
