from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import torch

from dtd_metrics import (
    binary_metrics,
    select_exact_threshold,
    select_threshold,
)
from evaluate_dtd import evaluate_suite


def test_optimized_threshold_scan_matches_direct_metric_scan():
    rng = np.random.default_rng(43)
    probability = rng.choice(
        np.linspace(0.0, 1.0, 101),
        size=10_000,
        replace=True,
    )
    target = rng.integers(0, 2, size=10_000)
    candidates = np.linspace(0.05, 0.95, 91)
    reference = max(
        (
            binary_metrics(probability, target, float(threshold))
            for threshold in candidates
        ),
        key=lambda item: (
            float(item["f1"]),
            float(item["precision"]),
            float(item["recall"]),
        ),
    )

    threshold, actual = select_threshold(probability, target)

    assert threshold == reference["threshold"]
    assert actual == reference


def exact_reference(
    probability: np.ndarray,
    target: np.ndarray,
) -> tuple[float, dict[str, float | int]]:
    thresholds = [
        float(np.nextafter(np.max(probability), np.inf)),
        *sorted({float(value) for value in probability}, reverse=True),
    ]
    results = [
        binary_metrics(probability, target, threshold)
        for threshold in thresholds
    ]

    def exact_key(item):
        true_positive = int(item["true_positive"])
        false_positive = int(item["false_positive"])
        false_negative = int(item["false_negative"])
        return (
            Fraction(
                2 * true_positive,
                max(
                    1,
                    2 * true_positive
                    + false_positive
                    + false_negative,
                ),
            ),
            Fraction(
                true_positive,
                max(1, true_positive + false_positive),
            ),
            Fraction(
                true_positive,
                max(1, true_positive + false_negative),
            ),
            float(item["threshold"]),
        )

    best = max(results, key=exact_key)
    return float(best["threshold"]), best


def test_exact_threshold_finds_off_grid_optimum():
    probability = np.asarray([0.504, 0.503, 0.502])
    target = np.asarray([1, 1, 0])

    threshold, metrics = select_exact_threshold(probability, target)
    _, grid_metrics = select_threshold(probability, target)

    assert threshold == 0.503
    assert metrics["f1"] == 1.0
    assert grid_metrics["f1"] == 0.8


def test_evaluation_report_includes_grid_and_exact_best_f1():
    class TinyDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "microphone": torch.zeros(640),
                "render": torch.zeros(640),
                "target": torch.tensor([1.0, 1.0, 0.0]),
                "source": "unit",
                "mode": "mixture",
                "file_id": "unit-0",
            }

    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            probability = torch.tensor([0.504, 0.503, 0.502])
            self.register_buffer(
                "fixed_logits",
                torch.logit(probability),
            )

        def forward(self, microphone, render):
            batch = microphone.shape[0]
            return self.fixed_logits.repeat(batch, 1), None

    report = evaluate_suite(
        FixedModel(),
        TinyDataset(),
        torch.device("cpu"),
        threshold=0.5,
        batch_size=1,
        num_workers=0,
    )

    assert report["best_f1_operating_point"]["f1"] == 0.8
    assert report["exact_best_f1_operating_point"]["f1"] == 1.0


@pytest.mark.parametrize(
    ("probability", "target", "expected_threshold"),
    [
        ([0.04, 0.03, 0.02], [1, 1, 0], 0.03),
        ([0.99, 0.98, 0.97], [1, 1, 0], 0.98),
    ],
)
def test_exact_threshold_can_select_outside_legacy_grid(
    probability,
    target,
    expected_threshold,
):
    threshold, metrics = select_exact_threshold(
        np.asarray(probability),
        np.asarray(target),
    )

    assert threshold == expected_threshold
    assert metrics["f1"] == 1.0


def test_exact_threshold_never_splits_equal_scores():
    pairs = [(0.8, 1), (0.8, 0), (0.4, 1), (0.1, 0)]
    expected = None
    for order in (
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [3, 2, 1, 0],
    ):
        probability = np.asarray([pairs[index][0] for index in order])
        target = np.asarray([pairs[index][1] for index in order])
        threshold, metrics = select_exact_threshold(probability, target)
        result = (
            threshold,
            metrics["true_positive"],
            metrics["false_positive"],
            metrics["false_negative"],
            metrics["f1"],
        )
        expected = result if expected is None else expected
        assert result == expected


def test_exact_threshold_handles_single_class_targets():
    negative_threshold, negative = select_exact_threshold(
        np.asarray([0.8, 0.2]),
        np.asarray([0, 0]),
    )
    positive_threshold, positive = select_exact_threshold(
        np.asarray([0.8, 0.2]),
        np.asarray([1, 1]),
    )

    assert negative_threshold > 0.8
    assert negative["false_positive"] == 0
    assert positive_threshold == 0.2
    assert positive["f1"] == 1.0


@pytest.mark.parametrize(
    ("probability", "target", "message"),
    [
        ([], [], "cannot be empty"),
        ([0.1, 0.2], [1], "same length"),
        ([np.nan], [1], "finite"),
        ([np.inf], [1], "finite"),
        ([-0.1], [1], r"\[0, 1\]"),
        ([1.1], [1], r"\[0, 1\]"),
        ([0.5], [np.nan], "target.*finite"),
        ([0.5], [np.inf], "target.*finite"),
    ],
)
def test_exact_threshold_rejects_invalid_input(
    probability,
    target,
    message,
):
    with pytest.raises(ValueError, match=message):
        select_exact_threshold(
            np.asarray(probability),
            np.asarray(target),
        )


def test_exact_threshold_matches_brute_force_reference():
    rng = np.random.default_rng(43)
    scores = np.asarray([0.01, 0.04, 0.2, 0.5, 0.8, 0.97, 0.99])
    for _ in range(200):
        size = int(rng.integers(1, 25))
        probability = rng.choice(scores, size=size, replace=True)
        target = rng.integers(0, 2, size=size)
        expected_threshold, expected = exact_reference(probability, target)
        threshold, actual = select_exact_threshold(probability, target)

        assert threshold == expected_threshold
        for key in (
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
            "precision",
            "recall",
        ):
            assert actual[key] == expected[key]
        true_positive = int(actual["true_positive"])
        denominator = (
            2 * true_positive
            + int(actual["false_positive"])
            + int(actual["false_negative"])
        )
        assert actual["f1"] == float(
            Fraction(2 * true_positive, max(1, denominator))
        )
