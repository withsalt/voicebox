from __future__ import annotations

import pytest

from compare_dtd_models import (
    LOW_FPR_KEYS,
    compare_suite,
    compared_metrics,
    parse_args,
)
from compare_dtd_round import strict_pareto_compare
from evaluate_dtd import (
    evaluation_seed_metadata,
    parse_args as parse_evaluation_args,
)


def suite(
    *,
    average_precision: float,
    roc_auc: float,
    best_f1: float,
    recalls: dict[str, float],
) -> dict[str, object]:
    return {
        "best_f1_operating_point": {
            "average_precision": average_precision,
            "roc_auc": roc_auc,
            "f1": best_f1,
        },
        "low_fpr_operating_points": {
            key: {"recall": recalls[key]} for key in LOW_FPR_KEYS
        },
    }


def baseline_suite() -> dict[str, object]:
    return suite(
        average_precision=0.90,
        roc_auc=0.92,
        best_f1=0.80,
        recalls={key: 0.60 for key in LOW_FPR_KEYS},
    )


def test_evaluation_seed_is_explicit_and_preserves_historical_default():
    required = [
        "baseline.pt",
        "candidate.pt",
        "--enriched-root",
        "dataset",
        "--output",
        "comparison.json",
    ]

    assert parse_args(required).evaluation_seed == 20260731
    assert (
        parse_args(required + ["--evaluation-seed", "20260801"]).evaluation_seed
        == 20260801
    )
    assert parse_evaluation_args(["candidate.pt"]).evaluation_seed == 20260731
    assert (
        parse_evaluation_args(
            ["candidate.pt", "--evaluation-seed", "20260801"]
        ).evaluation_seed
        == 20260801
    )
    with pytest.raises(SystemExit):
        parse_args(required + ["--evaluation-seed", "-1"])
    with pytest.raises(SystemExit):
        parse_evaluation_args(
            ["candidate.pt", "--evaluation-seed", "-1"]
        )
    assert evaluation_seed_metadata(7, enriched=False) == {
        "legacy_aec_seed": 7
    }
    assert evaluation_seed_metadata(7, enriched=True)["suite_seeds"] == {
        "aec_official": 7,
        "wenet_unseen": 8,
        "aec_real_unseen": 9,
    }


def test_compared_metrics_prefers_exact_f1_and_keeps_grid_fallback():
    report = baseline_suite()

    assert compared_metrics(report)["best_f1"] == 0.80
    report["exact_best_f1_operating_point"] = {
        "average_precision": 0.91,
        "roc_auc": 0.93,
        "f1": 0.81,
    }
    metrics = compared_metrics(report)
    assert metrics["average_precision"] == 0.91
    assert metrics["roc_auc"] == 0.93
    assert metrics["best_f1"] == 0.81


def test_comparison_rejects_mixed_grid_and_exact_f1_selectors():
    baseline = baseline_suite()
    candidate = baseline_suite()
    candidate["exact_best_f1_operating_point"] = {
        "average_precision": 0.90,
        "roc_auc": 0.92,
        "f1": 0.80,
    }

    with pytest.raises(ValueError, match="same best-F1 selector"):
        compare_suite(
            baseline,
            candidate,
            policy="aec_non_regression",
        )


def test_aec_policy_allows_only_requested_tolerances():
    candidate = suite(
        average_precision=0.898,
        roc_auc=0.918,
        best_f1=0.798,
        recalls={key: 0.585 for key in LOW_FPR_KEYS},
    )

    report = compare_suite(
        baseline_suite(),
        candidate,
        policy="aec_non_regression",
    )

    assert report["passed"]

    candidate["low_fpr_operating_points"]["1.000%"]["recall"] = 0.5849
    report = compare_suite(
        baseline_suite(),
        candidate,
        policy="aec_non_regression",
    )
    assert not report["passed"]


def test_wenet_explicit_targets_are_sufficient():
    candidate = suite(
        average_precision=0.850,
        roc_auc=0.920,
        best_f1=0.750,
        recalls={
            "0.100%": 0.20,
            "0.500%": 0.35,
            "1.000%": 0.450,
            "2.000%": 0.55,
            "5.000%": 0.65,
        },
    )

    report = compare_suite(
        baseline_suite(),
        candidate,
        policy="wenet_targets_or_pareto",
    )

    assert report["passed"]
    assert report["acceptance_basis"] == "explicit_targets"


def test_wenet_fallback_requires_strict_pareto_across_all_metrics():
    baseline = suite(
        average_precision=0.80,
        roc_auc=0.90,
        best_f1=0.70,
        recalls={key: 0.40 for key in LOW_FPR_KEYS},
    )
    candidate = suite(
        average_precision=0.801,
        roc_auc=0.90,
        best_f1=0.70,
        recalls={key: 0.40 for key in LOW_FPR_KEYS},
    )
    report = compare_suite(
        baseline,
        candidate,
        policy="wenet_targets_or_pareto",
    )
    assert report["passed"]
    assert report["acceptance_basis"] == "strict_pareto"

    candidate["low_fpr_operating_points"]["0.100%"]["recall"] = 0.399
    report = compare_suite(
        baseline,
        candidate,
        policy="wenet_targets_or_pareto",
    )
    assert not report["passed"]


def test_incumbent_round_comparison_requires_strict_pareto():
    baseline = baseline_suite()
    candidate = suite(
        average_precision=0.901,
        roc_auc=0.92,
        best_f1=0.80,
        recalls={key: 0.60 for key in LOW_FPR_KEYS},
    )

    report = strict_pareto_compare(baseline, candidate)

    assert report["passed"]
    candidate["low_fpr_operating_points"]["0.100%"]["recall"] = 0.5999
    assert not strict_pareto_compare(baseline, candidate)["passed"]
