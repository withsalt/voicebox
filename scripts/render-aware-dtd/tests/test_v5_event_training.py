from __future__ import annotations

import random
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dtd_event_metrics import event_metrics, production_trigger_mask
from dtd_training_objectives import (
    EventObjectiveConfig,
    apply_stream_resets,
    differentiable_event_support,
    event_loss,
    hard_negative_loss,
    onset_loss,
    reset_cold_start_loss,
)
from enriched_dataset import HOP_LENGTH, splice_stream_discontinuity, standard_item
from evaluate_dtd import evaluate_suite
from select_v5 import load_candidates, select_candidate


def objective_config(**overrides) -> EventObjectiveConfig:
    values = {
        "hard_negative_weight": 1.0,
        "onset_weight": 1.0,
        "event_weight": 1.0,
        "reset_weight": 1.0,
        "event_temperature": 0.08,
    }
    values.update(overrides)
    return EventObjectiveConfig(**values)


def test_hard_negative_loss_targets_high_tail_and_persistent_scores():
    target = torch.zeros((1, 24))
    valid = torch.ones_like(target, dtype=torch.bool)
    reset = torch.zeros_like(valid)
    far_only = torch.tensor([True])
    config = objective_config()

    low = hard_negative_loss(
        torch.full_like(target, -4.0),
        target,
        valid,
        reset,
        far_only,
        config,
    )
    persistent = hard_negative_loss(
        torch.full_like(target, 2.0),
        target,
        valid,
        reset,
        far_only,
        config,
    )

    assert persistent > low


def test_onset_loss_never_rewards_or_penalizes_oracle_pretrigger():
    target = torch.zeros((1, 30))
    target[:, 10:] = 1.0
    valid = torch.ones_like(target, dtype=torch.bool)
    reset = torch.zeros_like(valid)
    config = objective_config()
    baseline = torch.zeros_like(target)
    pretrigger = baseline.clone()
    pretrigger[:, :10] = 12.0

    baseline_loss = onset_loss(baseline, target, valid, reset, config)
    pretrigger_loss = onset_loss(pretrigger, target, valid, reset, config)

    assert pretrigger_loss == pytest.approx(baseline_loss)


def test_onset_loss_treats_a_continuing_target_across_reset_as_two_events():
    logits = torch.zeros((1, 24))
    valid = torch.ones_like(logits, dtype=torch.bool)
    reset = torch.zeros_like(valid)
    reset[:, 10] = True
    first_target = torch.zeros_like(logits)
    first_target[:, 5:10] = 1.0
    second_target = torch.zeros_like(logits)
    second_target[:, 10:15] = 1.0
    combined_target = first_target + second_target
    config = objective_config()

    combined = onset_loss(logits, combined_target, valid, reset, config)
    independent = 0.5 * (
        onset_loss(logits, first_target, valid, reset, config)
        + onset_loss(logits, second_target, valid, reset, config)
    )

    assert combined == pytest.approx(independent)


def test_event_loss_rewards_timely_positive_and_penalizes_far_only_trigger():
    target = torch.zeros((2, 40))
    target[0, 5:35] = 1.0
    valid = torch.ones_like(target, dtype=torch.bool)
    reset = torch.zeros_like(valid)
    far_only = torch.tensor([False, True])
    config = objective_config()
    good = torch.full_like(target, -6.0)
    good[0, 5:35] = 6.0
    bad = good.clone()
    bad[0, 5:35] = -2.0
    bad[1] = 4.0

    good_loss = event_loss(good, target, valid, reset, far_only, config)
    bad_loss = event_loss(bad, target, valid, reset, far_only, config)

    assert bad_loss > good_loss


def test_differentiable_event_evidence_does_not_cross_reset():
    logits = torch.full((1, 18), 8.0, requires_grad=True)
    valid = torch.ones_like(logits, dtype=torch.bool)
    reset = torch.zeros_like(valid)
    reset[:, 8] = True
    support = differentiable_event_support(
        logits,
        objective_config(),
        valid,
        reset,
    )

    assert support[0, 7] > support[0, 8]
    support.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_hard_trigger_evidence_does_not_cross_reset_or_invalid_gap():
    probability = np.full(20, 0.99)
    reset = np.zeros(20, dtype=bool)
    reset[8] = True
    valid = np.ones(20, dtype=bool)
    trigger = production_trigger_mask(
        probability,
        threshold=0.55,
        valid_mask=valid,
        reset_before=reset,
    )

    assert not np.any(trigger[:8])
    assert not np.any(trigger[8:16])
    assert trigger[16]

    valid[8] = False
    reset.fill(False)
    trigger_with_gap = production_trigger_mask(
        probability,
        threshold=0.55,
        valid_mask=valid,
        reset_before=reset,
    )
    assert not trigger_with_gap[8]
    assert trigger_with_gap[17]


def test_event_metrics_are_cluster_level_and_report_censoring_and_false_rate():
    positive_probability = np.full(40, 0.01)
    positive_probability[10:35] = 0.99
    positive_target = np.zeros(40)
    positive_target[10:35] = 1.0
    missed_probability = np.full(40, 0.01)
    missed_target = positive_target.copy()
    far_probability = np.full(40, 0.99)
    far_target = np.zeros(40)

    metrics = event_metrics(
        [positive_probability, missed_probability, far_probability],
        [positive_target, missed_target, far_target],
        0.55,
        modes=["mixture", "mixture", "far_only"],
    )

    assert metrics["positive_events"] == 2
    assert metrics["detected_events"] == 1
    assert metrics["censored_events"] == 1
    assert metrics["false_trigger_streams"] == 1
    assert metrics["latency_p50_ms"] == 180.0
    assert metrics["latency_p95_ms"] is None
    assert metrics["false_triggers_per_hour"] > 0
    assert (
        metrics["false_triggers_per_hour_upper95"] > metrics["false_triggers_per_hour"]
    )


def test_event_confidence_collapses_correlated_variants_by_cluster():
    target = np.zeros(40)
    target[5:35] = 1.0
    detected = np.full(40, 0.01)
    detected[5:35] = 0.99
    missed = np.full(40, 0.01)

    metrics = event_metrics(
        [detected, missed],
        [target, target],
        0.55,
        modes=["mixture", "mixture"],
        cluster_ids=["speaker-1", "speaker-1"],
    )

    assert metrics["positive_events"] == 2
    assert metrics["detected_events"] == 1
    assert metrics["positive_clusters"] == 1
    assert metrics["detected_clusters"] == 0
    assert metrics["censored_clusters"] == 1
    assert metrics["recall_lower95"] == 0.0


@pytest.mark.parametrize("boundary", ["reset", "invalid"])
def test_event_early_trigger_does_not_cross_stream_boundary(boundary):
    probability = np.full(60, 0.01)
    probability[:15] = 0.99
    probability[25:55] = 0.99
    target = np.zeros(60)
    target[25:55] = 1.0
    valid = np.ones(60, dtype=bool)
    reset = np.zeros(60, dtype=bool)
    if boundary == "reset":
        reset[20] = True
    else:
        valid[20] = False

    metrics = event_metrics(
        [probability],
        [target],
        0.55,
        modes=["mixture"],
        valid_masks=[valid],
        reset_masks=[reset],
    )

    assert metrics["detected_events"] == 1
    assert metrics["early_triggers"] == 0
    assert metrics["early_trigger_clusters"] == 0


def test_evaluation_uses_frozen_checkpoint_threshold_for_production_event():
    class FixedModel:
        config = SimpleNamespace(hop_length=HOP_LENGTH)

        def __call__(self, microphone, render, state=None):
            probabilities = torch.tensor([0.60] * 10 + [0.80] * 20)
            logits = torch.logit(probabilities).repeat(microphone.shape[0], 1)
            return logits, torch.zeros(1)

    target = np.zeros(30, dtype=np.float32)
    target[10:] = 1.0
    samples = np.zeros(HOP_LENGTH * 31, dtype=np.float32)
    dataset = [
        standard_item(
            samples,
            samples,
            target,
            source="synthetic",
            mode="mixture",
            file_id="case-1",
            ser_db=0.0,
            farend_nonlinear=False,
            farend_noisy=False,
            nearend_noisy=False,
        )
    ]

    report = evaluate_suite(
        FixedModel(),
        dataset,
        torch.device("cpu"),
        threshold=0.55,
        batch_size=1,
        num_workers=0,
    )

    assert report["production_event"]["threshold"] == 0.55
    assert report["production_event"]["early_triggers"] == 1
    assert report["production_event_at_best_f1_threshold"]["threshold"] > 0.60
    assert report["production_event_at_best_f1_threshold"]["early_triggers"] == 0


def test_legacy_evaluation_defaults_missing_enriched_metadata():
    class FixedModel:
        config = SimpleNamespace(hop_length=HOP_LENGTH)

        def __call__(self, microphone, render, state=None):
            logits = torch.linspace(-2.0, 2.0, 30).repeat(microphone.shape[0], 1)
            return logits, torch.zeros(1)

    dataset = [
        {
            "microphone": torch.zeros(HOP_LENGTH * 31),
            "render": torch.zeros(HOP_LENGTH * 31),
            "target": torch.cat((torch.zeros(15), torch.ones(15))),
            "file_id": 1,
            "mode": "mixture",
        }
    ]

    report = evaluate_suite(
        FixedModel(),
        dataset,
        torch.device("cpu"),
        threshold=0.55,
        batch_size=1,
        num_workers=0,
    )

    assert report["examples"] == 1
    assert "source:aec_legacy" in report["strata"]


def test_stream_discontinuity_omits_gap_instead_of_zero_filling_it():
    signal = np.arange(HOP_LENGTH * 30, dtype=np.float32)
    (spliced,), reset_frame, gap_hops = splice_stream_discontinuity(
        (signal,),
        random.Random(7),
        maximum_gap_hops=4,
    )
    boundary = reset_frame * HOP_LENGTH

    assert reset_frame > 0
    assert 1 <= gap_hops <= 4
    assert len(spliced) == len(signal)
    assert spliced[boundary] == signal[boundary + gap_hops * HOP_LENGTH]
    assert np.count_nonzero(spliced[-gap_hops * HOP_LENGTH :]) == 0


def test_stream_discontinuity_masks_cross_boundary_and_padding_frames():
    samples = np.zeros(HOP_LENGTH * 30, dtype=np.float32)
    target = np.zeros(29, dtype=np.float32)
    item = standard_item(
        samples,
        samples,
        target,
        source="synthetic",
        mode="far_only",
        file_id="case-1",
        ser_db=-99.0,
        farend_nonlinear=False,
        farend_noisy=False,
        nearend_noisy=False,
        reset_frame=10,
        gap_hops=3,
    )

    valid = item["valid_mask"].numpy()
    reset = item["reset_before"].numpy()
    assert not valid[9]
    assert valid[10]
    assert not np.any(valid[-3:])
    assert reset[10]


def test_reset_loss_weights_only_valid_post_boundary_frames():
    logits = torch.zeros((1, 29), requires_grad=True)
    target = torch.ones((1, 29))
    valid = torch.ones_like(target, dtype=torch.bool)
    reset = torch.zeros_like(valid)
    reset[:, 10] = True
    valid[:, 13] = False

    loss = reset_cold_start_loss(
        logits,
        target,
        valid,
        reset,
        positive_weight=1.5,
        config=objective_config(reset_warmup_hops=4),
    )
    loss.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, :10]) == 0
    assert torch.count_nonzero(logits.grad[:, 10:13]) == 3
    assert logits.grad[0, 13] == 0
    assert torch.count_nonzero(logits.grad[:, 14:]) == 0


def test_post_gap_logits_are_recomputed_without_pre_gap_state():
    class StatefulModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, microphone, render, state=None):
            self.calls += 1
            frames = 1 + (microphone.shape[1] - 640) // 320
            initial = 0.0 if state is None else float(state)
            logits = torch.arange(
                initial,
                initial + frames,
                dtype=microphone.dtype,
            ).repeat(microphone.shape[0], 1)
            return logits, torch.tensor(initial + frames)

    model = StatefulModel()
    microphone = torch.zeros((2, 320 * 20))
    render = torch.zeros_like(microphone)
    full_logits = torch.arange(19, dtype=torch.float32).repeat(2, 1) + 100.0
    reset = torch.zeros_like(full_logits, dtype=torch.bool)
    reset[0, 10] = True
    reset[1, 5] = True

    actual = apply_stream_resets(
        model,
        full_logits,
        microphone,
        render,
        reset,
        hop_length=320,
    )

    assert torch.equal(actual[0, :10], full_logits[0, :10])
    assert actual[0, 10] == 0.0
    assert actual[0, 18] == 8.0
    assert torch.equal(actual[1, :5], full_logits[1, :5])
    assert actual[1, 5] == 0.0
    assert actual[1, 18] == 13.0
    assert model.calls == 1


def selection_metrics(
    *,
    frame: float = 0.9,
    recall_lower95: float = 0.8,
    p95: float | None = 600.0,
    false_upper: float = 2.0,
) -> dict[str, object]:
    return {
        "average_precision": frame,
        "roc_auc": frame,
        "f1": frame,
        "low_fpr_operating_points": {
            key: {"recall": frame}
            for key in ("0.100%", "0.500%", "1.000%", "2.000%", "5.000%")
        },
        "event": {
            "recall": recall_lower95 + 0.05,
            "recall_lower95": recall_lower95,
            "specificity": 0.99,
            "cluster_specificity": 0.99,
            "false_triggers_per_hour": 1.0,
            "false_triggers_per_hour_upper95": false_upper,
            "early_triggers": 0,
            "early_trigger_clusters": 0,
            "censored_events": 0,
            "censored_clusters": 0,
            "cluster_recall": recall_lower95 + 0.05,
            "error_count": 0,
            "latency_p50_ms": 300.0,
            "latency_p95_ms": p95,
        },
    }


def test_v5_selector_rejects_any_frame_or_event_regression_before_ranking():
    anchor = selection_metrics()
    better_recall = selection_metrics(recall_lower95=0.85)
    frame_regression = selection_metrics(frame=0.89, recall_lower95=0.99)
    fp_regression = selection_metrics(recall_lower95=0.99, false_upper=2.01)

    result = select_candidate(
        anchor,
        [
            ("better.json", better_recall),
            ("frame-regression.json", frame_regression),
            ("fp-regression.json", fp_regression),
        ],
    )

    assert result["selected"] == "better.json"
    assert result["candidates"][1]["accepted"] is False
    assert result["candidates"][2]["accepted"] is False


def test_v5_selector_treats_new_unbounded_latency_as_regression():
    result = select_candidate(
        selection_metrics(p95=600.0),
        [("unbounded.json", selection_metrics(p95=None))],
    )

    assert result["selected"] is None
    assert "latency_p95_ms" in " ".join(result["candidates"][0]["rejection_reasons"])


def test_v5_selector_rejects_event_recall_regression_even_when_other_gates_hold():
    result = select_candidate(
        selection_metrics(recall_lower95=0.80),
        [("recall-regression.json", selection_metrics(recall_lower95=0.79))],
    )

    assert result["selected"] is None
    reasons = " ".join(result["candidates"][0]["rejection_reasons"])
    assert "recall" in reasons


def test_v5_selector_expands_every_epoch_from_training_history(tmp_path):
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps(
            [
                {"epoch": 3, "validation": selection_metrics()},
                {"epoch": 4, "validation": selection_metrics(recall_lower95=0.81)},
            ]
        ),
        encoding="utf-8",
    )

    candidates = load_candidates(history)

    assert [path.rsplit("#epoch=", 1)[1] for path, _ in candidates] == ["3", "4"]
    assert candidates[1][1]["event"]["recall_lower95"] == 0.81
