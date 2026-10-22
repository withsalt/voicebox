from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class EventDecisionConfig:
    hop_ms: int = 20
    smoothing_hops: int = 3
    evidence_window_hops: int = 18
    required_evidence_hops: int = 9
    tail_hops: int = 4
    minimum_tail_fraction: float = 0.50

    def validate(self) -> None:
        for name in (
            "hop_ms",
            "smoothing_hops",
            "evidence_window_hops",
            "required_evidence_hops",
            "tail_hops",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.required_evidence_hops > self.evidence_window_hops:
            raise ValueError(
                "required_evidence_hops cannot exceed evidence_window_hops"
            )
        if self.tail_hops > self.evidence_window_hops:
            raise ValueError("tail_hops cannot exceed evidence_window_hops")
        if not 0.0 <= self.minimum_tail_fraction <= 1.0:
            raise ValueError("minimum_tail_fraction must be in [0, 1]")


def _as_streams(
    values: list[np.ndarray] | np.ndarray,
    *,
    dtype: np.dtype | type,
) -> list[np.ndarray]:
    if isinstance(values, np.ndarray) and values.ndim == 2:
        return [np.asarray(row, dtype=dtype).reshape(-1) for row in values]
    if isinstance(values, np.ndarray) and values.ndim == 1:
        return [np.asarray(values, dtype=dtype).reshape(-1)]
    return [np.asarray(row, dtype=dtype).reshape(-1) for row in values]


def causal_smoothed_probability(
    probability: np.ndarray,
    valid_mask: np.ndarray,
    reset_before: np.ndarray,
    smoothing_hops: int,
) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    reset_before = np.asarray(reset_before, dtype=bool).reshape(-1)
    if not (len(probability) == len(valid_mask) == len(reset_before)):
        raise ValueError("probability, valid_mask, and reset_before must align")
    if smoothing_hops <= 0:
        raise ValueError("smoothing_hops must be positive")
    if not np.all(np.isfinite(probability)):
        raise ValueError("probability must contain only finite values")

    output = np.full(len(probability), np.nan, dtype=np.float64)
    history: list[float] = []
    for index, value in enumerate(probability):
        if reset_before[index] or not valid_mask[index]:
            history.clear()
        if not valid_mask[index]:
            continue
        history.append(float(value))
        if len(history) > smoothing_hops:
            history.pop(0)
        output[index] = float(np.mean(history))
    return output


def production_trigger_mask(
    probability: np.ndarray,
    threshold: float,
    valid_mask: np.ndarray | None = None,
    reset_before: np.ndarray | None = None,
    config: EventDecisionConfig | None = None,
) -> np.ndarray:
    config = config or EventDecisionConfig()
    config.validate()
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    valid = (
        np.ones(len(probability), dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool).reshape(-1)
    )
    reset = (
        np.zeros(len(probability), dtype=bool)
        if reset_before is None
        else np.asarray(reset_before, dtype=bool).reshape(-1)
    )
    if not (len(probability) == len(valid) == len(reset)):
        raise ValueError("event inputs must have identical lengths")

    smoothed = causal_smoothed_probability(
        probability,
        valid,
        reset,
        config.smoothing_hops,
    )
    accepted = np.isfinite(smoothed) & (smoothed >= threshold)
    trigger = np.zeros(len(probability), dtype=bool)
    window: list[bool] = []
    for index in range(len(probability)):
        if reset[index] or not valid[index]:
            window.clear()
        if not valid[index]:
            continue
        window.append(bool(accepted[index]))
        if len(window) > config.evidence_window_hops:
            window.pop(0)
        tail = window[-config.tail_hops :]
        trigger[index] = (
            accepted[index]
            and sum(window) >= config.required_evidence_hops
            and len(tail) > 0
            and sum(tail) >= math.ceil(len(tail) * config.minimum_tail_fraction)
        )
    return trigger


def wilson_lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return float("nan")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return max(0.0, (centre - margin) / denominator)


def wilson_upper_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return float("nan")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return min(1.0, (centre + margin) / denominator)


def poisson_rate_upper_bound(
    count: int,
    exposure_hours: float,
    confidence: float = 0.95,
) -> float:
    if count < 0:
        raise ValueError("count cannot be negative")
    if exposure_hours <= 0.0:
        return float("nan")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    upper_count = 0.5 * float(chi2.ppf(confidence, 2 * (count + 1)))
    return upper_count / exposure_hours


def _event_intervals(
    target: np.ndarray,
    valid: np.ndarray,
    reset_before: np.ndarray,
) -> list[tuple[int, int]]:
    active = (target >= 0.5) & valid
    # A continuing target across a stream reset is a new independently
    # observable event; recurrent evidence must never bridge the boundary.
    split_points = np.flatnonzero(reset_before & active)
    intervals: list[tuple[int, int]] = []
    segment_start = 0
    for split in [*split_points.tolist(), len(active)]:
        if split > segment_start:
            segment = active[segment_start:split]
            padded = np.pad(segment.astype(np.int8), (1, 1))
            changes = np.diff(padded)
            starts = np.flatnonzero(changes == 1) + segment_start
            stops = np.flatnonzero(changes == -1) + segment_start
            intervals.extend(zip(starts.tolist(), stops.tolist()))
        segment_start = split
    if not intervals and np.any(active):
        padded = np.pad(active.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        intervals.extend(zip(starts.tolist(), stops.tolist()))
    return intervals


def _censored_percentile(
    observed_ms: list[float],
    censored: int,
    percentile: float,
) -> float | None:
    total = len(observed_ms) + censored
    if total == 0 or censored / total > 1.0 - percentile:
        return None
    return float(np.quantile(np.asarray(observed_ms), percentile, method="higher"))


def event_metrics(
    probabilities: list[np.ndarray] | np.ndarray,
    targets: list[np.ndarray] | np.ndarray,
    threshold: float,
    *,
    modes: list[str] | None = None,
    cluster_ids: list[str] | None = None,
    valid_masks: list[np.ndarray] | np.ndarray | None = None,
    reset_masks: list[np.ndarray] | np.ndarray | None = None,
    config: EventDecisionConfig | None = None,
) -> dict[str, float | int | None]:
    """Compute cluster-level production-timing metrics without flattening streams."""

    config = config or EventDecisionConfig()
    config.validate()
    probability_streams = _as_streams(probabilities, dtype=np.float64)
    target_streams = _as_streams(targets, dtype=np.float64)
    if len(probability_streams) != len(target_streams):
        raise ValueError("probabilities and targets must contain the same streams")
    stream_count = len(probability_streams)
    if modes is None:
        modes = ["unknown"] * stream_count
    if len(modes) != stream_count:
        raise ValueError("modes must contain one value per stream")
    if cluster_ids is None:
        cluster_ids = [f"stream:{index}" for index in range(stream_count)]
    if len(cluster_ids) != stream_count:
        raise ValueError("cluster_ids must contain one value per stream")
    valid_streams = (
        [np.ones(len(row), dtype=bool) for row in probability_streams]
        if valid_masks is None
        else _as_streams(valid_masks, dtype=bool)
    )
    reset_streams = (
        [np.zeros(len(row), dtype=bool) for row in probability_streams]
        if reset_masks is None
        else _as_streams(reset_masks, dtype=bool)
    )
    if not (len(valid_streams) == stream_count and len(reset_streams) == stream_count):
        raise ValueError("mask collections must contain one value per stream")

    positive_events = detected_events = censored_events = 0
    early_triggers = 0
    negative_streams = false_trigger_streams = 0
    negative_valid_hops = 0
    error_count = 0
    positive_cluster_records: dict[
        str,
        list[tuple[bool, bool, float | None]],
    ] = defaultdict(list)
    negative_cluster_records: dict[str, list[tuple[bool, int]]] = defaultdict(list)
    for probability, target, mode, cluster_id, valid, reset in zip(
        probability_streams,
        target_streams,
        modes,
        cluster_ids,
        valid_streams,
        reset_streams,
    ):
        if not (len(probability) == len(target) == len(valid) == len(reset)):
            raise ValueError("all arrays within an event stream must align")
        try:
            trigger = production_trigger_mask(
                probability,
                threshold,
                valid,
                reset,
                config,
            )
        except (ValueError, FloatingPointError):
            error_count += 1
            continue

        intervals = _event_intervals(target, valid, reset)
        is_far_only = mode == "far_only" or not intervals
        if is_far_only:
            negative_streams += 1
            valid_hops = int(np.count_nonzero(valid))
            false_trigger = bool(np.any(trigger & valid))
            negative_valid_hops += valid_hops
            false_trigger_streams += int(false_trigger)
            negative_cluster_records[str(cluster_id)].append(
                (false_trigger, valid_hops)
            )
            continue

        previous_stop = 0
        for onset, stop in intervals:
            positive_events += 1
            boundaries = np.flatnonzero((reset | ~valid)[: onset + 1])
            early_start = previous_stop
            if len(boundaries):
                early_start = max(early_start, int(boundaries[-1]))
            if np.any(trigger[early_start:onset]):
                early_triggers += 1
                early = True
            else:
                early = False
            detected = np.flatnonzero(trigger[onset:stop])
            if len(detected):
                detected_events += 1
                latency_ms = float(detected[0] * config.hop_ms)
            else:
                censored_events += 1
                latency_ms = None
            positive_cluster_records[str(cluster_id)].append(
                (bool(len(detected)), early, latency_ms)
            )
            previous_stop = stop

    recall = detected_events / max(1, positive_events)
    specificity = (negative_streams - false_trigger_streams) / max(1, negative_streams)
    positive_clusters = len(positive_cluster_records)
    detected_clusters = sum(
        all(record[0] for record in records)
        for records in positive_cluster_records.values()
    )
    censored_clusters = positive_clusters - detected_clusters
    early_trigger_clusters = sum(
        any(record[1] for record in records)
        for records in positive_cluster_records.values()
    )
    cluster_latencies_ms = [
        max(float(record[2]) for record in records if record[2] is not None)
        for records in positive_cluster_records.values()
        if all(record[0] for record in records)
    ]
    negative_clusters = len(negative_cluster_records)
    false_trigger_clusters = sum(
        any(record[0] for record in records)
        for records in negative_cluster_records.values()
    )
    cluster_specificity = (negative_clusters - false_trigger_clusters) / max(
        1, negative_clusters
    )
    exposure_hours = negative_valid_hops * config.hop_ms / 3_600_000.0
    false_triggers_per_hour = false_trigger_streams / max(
        exposure_hours,
        np.finfo(np.float64).tiny,
    )
    poisson_upper = poisson_rate_upper_bound(
        false_trigger_streams,
        exposure_hours,
    )
    clustered_upper = (
        wilson_upper_bound(false_trigger_clusters, negative_clusters)
        * negative_streams
        / exposure_hours
        if negative_clusters > 0 and exposure_hours > 0.0
        else float("nan")
    )
    conservative_upper = (
        max(poisson_upper, clustered_upper)
        if math.isfinite(clustered_upper)
        else poisson_upper
    )
    return {
        "threshold": float(threshold),
        "streams": stream_count,
        "positive_events": positive_events,
        "detected_events": detected_events,
        "censored_events": censored_events,
        "early_triggers": early_triggers,
        "recall": float(recall),
        "positive_clusters": positive_clusters,
        "detected_clusters": detected_clusters,
        "censored_clusters": censored_clusters,
        "early_trigger_clusters": early_trigger_clusters,
        "cluster_recall": detected_clusters / max(1, positive_clusters),
        "recall_lower95": float(
            wilson_lower_bound(detected_clusters, positive_clusters)
        ),
        "negative_streams": negative_streams,
        "false_trigger_streams": false_trigger_streams,
        "specificity": float(specificity),
        "negative_clusters": negative_clusters,
        "false_trigger_clusters": false_trigger_clusters,
        "cluster_specificity": float(cluster_specificity),
        "negative_exposure_hours": float(exposure_hours),
        "false_triggers_per_hour": float(false_triggers_per_hour),
        "false_triggers_per_hour_poisson_upper95": float(poisson_upper),
        "false_triggers_per_hour_cluster_upper95": float(clustered_upper),
        "false_triggers_per_hour_upper95": float(conservative_upper),
        "latency_p50_ms": _censored_percentile(
            cluster_latencies_ms,
            censored_clusters,
            0.50,
        ),
        "latency_p95_ms": _censored_percentile(
            cluster_latencies_ms,
            censored_clusters,
            0.95,
        ),
        "error_count": error_count,
    }
