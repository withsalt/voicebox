from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compare_dtd_models import (
    best_f1_selector,
    compare_suite,
    compared_metrics,
)
from evaluate_dtd import (
    configure_cuda_backend,
    evaluate_suite,
    json_safe,
    load_model,
    nonnegative_int,
)
from enriched_dataset import create_evaluation_suites


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate-2, candidate-6, and one validation-locked "
            "candidate on the same three freshly instantiated suites."
        )
    )
    parser.add_argument("candidate_2")
    parser.add_argument("candidate_6")
    parser.add_argument("candidate")
    parser.add_argument("--enriched-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--online-examples", type=int, default=2_000)
    parser.add_argument("--sequence-seconds", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--evaluation-seed",
        type=nonnegative_int,
        default=20260801,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


def strict_pareto_compare(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    baseline_selector = best_f1_selector(baseline)
    candidate_selector = best_f1_selector(candidate)
    if baseline_selector != candidate_selector:
        raise ValueError(
            "incumbent and candidate must use the same best-F1 selector"
        )
    baseline_metrics = compared_metrics(baseline)
    candidate_metrics = compared_metrics(candidate)
    deltas = {
        key: candidate_metrics[key] - baseline_metrics[key]
        for key in baseline_metrics
    }
    checks = {key: delta >= 0.0 for key, delta in deltas.items()}
    materially_improved = any(delta > 0.0 for delta in deltas.values())
    return {
        "policy": "strict_pareto",
        "best_f1_selector": baseline_selector,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": deltas,
        "checks": checks,
        "materially_improved": materially_improved,
        "passed": all(checks.values()) and materially_improved,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary_output.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {output}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    configure_cuda_backend(device)

    checkpoint_paths = {
        "candidate_2": Path(args.candidate_2).resolve(),
        "candidate_6": Path(args.candidate_6).resolve(),
        "candidate": Path(args.candidate).resolve(),
    }
    models: dict[str, torch.nn.Module] = {}
    checkpoints: dict[str, dict[str, object]] = {}
    for label, path in checkpoint_paths.items():
        model, checkpoint = load_model(path, device)
        models[label] = model
        checkpoints[label] = checkpoint

    suites = create_evaluation_suites(
        args.enriched_root,
        sequence_seconds=args.sequence_seconds,
        online_examples=args.online_examples,
        seed=args.evaluation_seed,
    )
    reports: dict[str, dict[str, object]] = {}
    comparisons: dict[str, dict[str, object]] = {}
    for suite_name, dataset in suites.items():
        suite_reports: dict[str, object] = {}
        for label in ("candidate_2", "candidate_6", "candidate"):
            suite_reports[label] = evaluate_suite(
                models[label],
                dataset,
                device,
                threshold=float(
                    checkpoints[label]["decision_threshold"]
                ),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        reports[suite_name] = suite_reports
        comparisons[suite_name] = {
            "candidate_vs_candidate_2": compare_suite(
                suite_reports["candidate_2"],
                suite_reports["candidate"],
                policy=(
                    "wenet_targets_or_pareto"
                    if suite_name == "wenet_unseen"
                    else "aec_non_regression"
                ),
            ),
            "candidate_vs_candidate_6": (
                strict_pareto_compare(
                    suite_reports["candidate_6"],
                    suite_reports["candidate"],
                )
                if suite_name == "wenet_unseen"
                else compare_suite(
                    suite_reports["candidate_6"],
                    suite_reports["candidate"],
                    policy="aec_non_regression",
                )
            ),
        }

    all_checks = [
        comparison["passed"]
        for suite in comparisons.values()
        for comparison in suite.values()
    ]
    complexity = {
        label: models[label].complexity().to_dict()
        for label in models
    }
    report = json_safe(
        {
            "schema_version": "1.0",
            "passed": all(all_checks),
            "checkpoints": {
                label: str(path) for label, path in checkpoint_paths.items()
            },
            "dataset": str(Path(args.enriched_root).resolve()),
            "evaluation": {
                "single_suite_instantiation": True,
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
                "selection_or_threshold_tuning_from_test": False,
            },
            "acceptance_policy": {
                "candidate_vs_candidate_6_wenet": (
                    "strict Pareto improvement across AP, AUC, exact best F1, "
                    "and recall at .1/.5/1/2/5% FPR"
                ),
                "candidate_vs_candidate_6_aec": (
                    "AP/AUC/exact best F1 delta >= -0.002 and each recall "
                    "delta >= -0.015"
                ),
                "candidate_vs_candidate_2": (
                    "original candidate-2 non-regression and Wenet target "
                    "policy"
                ),
            },
            "complexity": complexity,
            "comparisons": comparisons,
            "reports": reports,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": report["passed"],
                "comparisons": comparisons,
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
