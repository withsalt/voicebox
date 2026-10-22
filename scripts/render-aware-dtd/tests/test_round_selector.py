from __future__ import annotations

from copy import deepcopy

import pytest

from select_256x3_round import (
    CHECKPOINT_FORMAT_VERSION,
    INCUMBENT_SHA256,
    MATERIAL_IMPROVEMENT,
    SELECTION_POLICY_VERSION,
    choose_winner,
    promotion_gate_reasons,
    validate_checkpoint,
    validate_incumbent_checkpoint,
    validation_gate_reasons,
)


def metrics() -> dict[str, object]:
    point = {
        "recall": 0.80,
    }
    source = {
        "average_precision": 0.97,
        "roc_auc": 0.98,
    }
    return {
        "average_precision": 0.97,
        "roc_auc": 0.98,
        "f1": 0.91,
        "low_fpr_operating_points": {
            "0.100%": deepcopy(point),
            "0.500%": deepcopy(point),
            "1.000%": deepcopy(point),
            "2.000%": deepcopy(point),
            "5.000%": deepcopy(point),
        },
        "strata": {
            "source:online_validation": deepcopy(source),
            "source:aec_narrow": deepcopy(source),
            "source:aec_fullband": deepcopy(source),
            "source:aec_real:real": deepcopy(source),
            "source:aec_real:real_doubled": deepcopy(source),
        },
    }


def test_validation_gate_accepts_exact_incumbent_metrics():
    incumbent = metrics()

    assert validation_gate_reasons(deepcopy(incumbent), incumbent) == []


def test_promotion_requires_score_online_and_low_fpr_improvement():
    incumbent = metrics()
    candidate = deepcopy(incumbent)
    candidate["strata"]["source:online_validation"]["average_precision"] += (
        MATERIAL_IMPROVEMENT * 2
    )
    candidate["low_fpr_operating_points"]["1.000%"]["recall"] += (
        MATERIAL_IMPROVEMENT * 2
    )

    assert (
        promotion_gate_reasons(
            candidate,
            score=1.5 + MATERIAL_IMPROVEMENT * 2,
            incumbent=incumbent,
            incumbent_score=1.5,
        )
        == []
    )


def test_winner_tie_band_prefers_online_ap_then_low_fpr_recall():
    results = [
        {
            "candidate": 7,
            "qualified": True,
            "selection_score": 1.5003,
            "online_average_precision": 0.9700,
            "recall_at_fpr": {"1.000%": 0.81},
        },
        {
            "candidate": 8,
            "qualified": True,
            "selection_score": 1.5002,
            "online_average_precision": 0.9710,
            "recall_at_fpr": {"1.000%": 0.80},
        },
        {
            "candidate": 9,
            "qualified": False,
            "selection_score": 1.6,
            "online_average_precision": 1.0,
            "recall_at_fpr": {"1.000%": 1.0},
        },
    ]

    assert choose_winner(results)["candidate"] == 8


def test_incumbent_checkpoint_hash_is_fail_closed():
    with pytest.raises(ValueError, match="incumbent checkpoint hash mismatch"):
        validate_incumbent_checkpoint({}, "0" * len(INCUMBENT_SHA256))


def test_checkpoint_validation_metrics_must_match_external_history():
    history_metrics = metrics()
    checkpoint_metrics = deepcopy(history_metrics)
    checkpoint_metrics["average_precision"] -= 0.01
    history = [
        {
            "epoch": 0,
            "selection_score": 1.25,
            "validation": history_metrics,
        }
    ]
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "epoch": 0,
        "model_config": {
            "sample_rate": 16_000,
            "frame_length": 640,
            "hop_length": 320,
            "frequency_bands": 48,
            "projection_size": 256,
            "hidden_size": 256,
            "gru_layers": 3,
            "dropout": 0.1,
        },
        "selection_score": 1.25,
        "history": history,
        "validation_metrics": checkpoint_metrics,
        "training_args": {},
    }

    with pytest.raises(ValueError, match="validation metrics"):
        validate_checkpoint(
            checkpoint,
            candidate=7,
            epoch=0,
            history=history,
            canonical_recipe=None,
            check_parameters=False,
        )
