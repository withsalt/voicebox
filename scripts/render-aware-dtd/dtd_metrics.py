from __future__ import annotations

import numpy as np


def binary_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1) >= 0.5
    return _binary_metrics_with_ranking(
        probability,
        target,
        threshold,
        auc=roc_auc(probability, target),
        ap=average_precision(probability, target),
    )


def _binary_metrics_with_ranking(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
    *,
    auc: float,
    ap: float,
) -> dict[str, float | int]:
    predicted = probability >= threshold
    true_positive = int(np.count_nonzero(predicted & target))
    false_positive = int(np.count_nonzero(predicted & ~target))
    true_negative = int(np.count_nonzero(~predicted & ~target))
    false_negative = int(np.count_nonzero(~predicted & target))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    false_positive_rate = false_positive / max(1, false_positive + true_negative)
    return {
        "threshold": float(threshold),
        "frames": int(len(target)),
        "positive_frames": int(np.count_nonzero(target)),
        "negative_frames": int(np.count_nonzero(~target)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "roc_auc": float(auc),
        "average_precision": float(ap),
    }


def roc_auc(probability: np.ndarray, target: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    positives = int(np.count_nonzero(target))
    negatives = int(np.count_nonzero(~target))
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-np.asarray(probability), kind="stable")
    sorted_target = target[order]
    true_positive = np.concatenate(([0.0], np.cumsum(sorted_target)))
    false_positive = np.concatenate(([0.0], np.cumsum(~sorted_target)))
    true_positive /= positives
    false_positive /= negatives
    return float(np.trapezoid(true_positive, false_positive))


def average_precision(probability: np.ndarray, target: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    positives = int(np.count_nonzero(target))
    if positives == 0:
        return float("nan")
    order = np.argsort(-np.asarray(probability), kind="stable")
    sorted_target = target[order]
    cumulative_positive = np.cumsum(sorted_target)
    precision = cumulative_positive / np.arange(1, len(sorted_target) + 1)
    return float(np.sum(precision[sorted_target]) / positives)


def select_threshold(
    probability: np.ndarray,
    target: np.ndarray,
) -> tuple[float, dict[str, float | int]]:
    """Select the best operating point on the frozen 0.01 threshold grid."""
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1) >= 0.5
    auc = roc_auc(probability, target)
    ap = average_precision(probability, target)
    candidates = np.linspace(0.05, 0.95, 91)
    results = [
        _binary_metrics_with_ranking(
            probability,
            target,
            float(value),
            auc=auc,
            ap=ap,
        )
        for value in candidates
    ]
    best = max(
        results,
        key=lambda item: (
            float(item["f1"]),
            float(item["precision"]),
            float(item["recall"]),
        ),
    )
    return float(best["threshold"]), best


def select_exact_threshold(
    probability: np.ndarray,
    target: np.ndarray,
) -> tuple[float, dict[str, float | int]]:
    """Select the exact best-F1 operating point across all distinct scores."""
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    raw_target = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(probability) != len(raw_target):
        raise ValueError("probability and target must have the same length")
    if len(probability) == 0:
        raise ValueError("probability and target cannot be empty")
    if not np.all(np.isfinite(probability)):
        raise ValueError("probability must contain only finite values")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probability values must be in [0, 1]")
    if not np.all(np.isfinite(raw_target)):
        raise ValueError("target must contain only finite values")

    target_mask = raw_target >= 0.5
    auc = roc_auc(probability, target_mask)
    ap = average_precision(probability, target_mask)
    order = np.argsort(-probability, kind="stable")
    sorted_probability = probability[order]
    sorted_target = target_mask[order]
    cumulative_true_positive = np.cumsum(sorted_target, dtype=np.int64)
    group_ends = np.concatenate(
        (
            np.flatnonzero(
                sorted_probability[:-1] != sorted_probability[1:]
            ),
            [len(sorted_probability) - 1],
        )
    )

    positives = int(np.count_nonzero(target_mask))
    best_true_positive = 0
    best_predicted = 0
    best_threshold = float(
        np.nextafter(sorted_probability[0], np.inf)
    )
    for group_end in group_ends:
        predicted = int(group_end) + 1
        true_positive = int(cumulative_true_positive[group_end])

        # F1 = 2*TP/(P + predicted), so integer cross-products are exact.
        f1_comparison = (
            true_positive * (positives + best_predicted)
            - best_true_positive * (positives + predicted)
        )
        replace = f1_comparison > 0
        if f1_comparison == 0:
            precision_comparison = (
                true_positive * max(1, best_predicted)
                - best_true_positive * predicted
            )
            replace = precision_comparison > 0
            if precision_comparison == 0:
                replace = true_positive > best_true_positive
        if replace:
            best_true_positive = true_positive
            best_predicted = predicted
            best_threshold = float(sorted_probability[group_end])

    best = _binary_metrics_with_ranking(
        probability,
        target_mask,
        best_threshold,
        auc=auc,
        ap=ap,
    )
    true_positive = int(best["true_positive"])
    false_positive = int(best["false_positive"])
    false_negative = int(best["false_negative"])
    best["f1"] = float(
        2 * true_positive
        / max(
            1,
            2 * true_positive + false_positive + false_negative,
        )
    )
    return best_threshold, best


def low_fpr_operating_points(
    probability: np.ndarray,
    target: np.ndarray,
    maximum_false_positive_rates: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05),
) -> dict[str, dict[str, float | int]]:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1) >= 0.5
    negative_probability = np.sort(probability[~target])[::-1]
    if len(negative_probability) == 0:
        return {}
    auc = roc_auc(probability, target)
    ap = average_precision(probability, target)

    points: dict[str, dict[str, float | int]] = {}
    for maximum_fpr in maximum_false_positive_rates:
        allowed_false_positives = int(
            np.floor(maximum_fpr * len(negative_probability))
        )
        if allowed_false_positives <= 0:
            threshold = np.nextafter(negative_probability[0], np.inf)
        elif allowed_false_positives >= len(negative_probability):
            threshold = 0.0
        else:
            threshold = np.nextafter(
                negative_probability[allowed_false_positives],
                np.inf,
            )
        metrics = _binary_metrics_with_ranking(
            probability,
            target,
            float(threshold),
            auc=auc,
            ap=ap,
        )
        points[f"{maximum_fpr:.3%}"] = metrics
    return points
