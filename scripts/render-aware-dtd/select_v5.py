from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

POLICY_VERSION = "v5_event_lexicographic_v1"
LOW_FPR_KEYS = ("0.100%", "0.500%", "1.000%", "2.000%", "5.000%")


@dataclass(frozen=True)
class CandidateDecision:
    path: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    selection_key: tuple[float, ...] | None


def _metric_object(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("metrics payload must be an object")
    validation = payload.get("validation")
    if isinstance(validation, dict):
        return validation
    return payload


def load_metrics(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raise ValueError("anchor metrics must be a single object, not a history array")
    return _metric_object(payload)


def load_candidates(path: str | Path) -> list[tuple[str, dict[str, object]]]:
    """Load one metrics object or every epoch from a training history."""

    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return [(str(resolved), _metric_object(payload))]
    candidates: list[tuple[str, dict[str, object]]] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"history record {index} must be an object")
        epoch = record.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError(f"history record {index} has no valid epoch")
        candidates.append(
            (
                f"{resolved}#epoch={epoch}",
                _metric_object(record),
            )
        )
    if not candidates:
        raise ValueError("candidate history must not be empty")
    return candidates


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _latency(value: object) -> float:
    return math.inf if value is None else _finite(value, "latency")


def non_regression_reasons(
    anchor: dict[str, object],
    candidate: dict[str, object],
) -> list[str]:
    reasons: list[str] = []
    for name in ("average_precision", "roc_auc", "f1"):
        if _finite(candidate[name], name) < _finite(anchor[name], name):
            reasons.append(f"frame {name} regressed")

    anchor_low_fpr = anchor.get("low_fpr_operating_points")
    candidate_low_fpr = candidate.get("low_fpr_operating_points")
    if not isinstance(anchor_low_fpr, dict) or not isinstance(candidate_low_fpr, dict):
        raise ValueError("low_fpr_operating_points must be objects")
    for key in LOW_FPR_KEYS:
        anchor_point = anchor_low_fpr.get(key)
        candidate_point = candidate_low_fpr.get(key)
        if not isinstance(anchor_point, dict) or not isinstance(candidate_point, dict):
            raise ValueError(f"missing low-FPR operating point: {key}")
        if _finite(candidate_point["recall"], f"recall@{key}") < _finite(
            anchor_point["recall"],
            f"anchor recall@{key}",
        ):
            reasons.append(f"frame recall@{key} regressed")

    anchor_event = anchor.get("event") or anchor.get("production_event")
    candidate_event = candidate.get("event") or candidate.get("production_event")
    if not isinstance(anchor_event, dict) or not isinstance(candidate_event, dict):
        raise ValueError("event metrics must be objects")
    higher_is_better = (
        "recall",
        "cluster_recall",
        "recall_lower95",
        "specificity",
        "cluster_specificity",
    )
    lower_is_better = (
        "false_triggers_per_hour",
        "false_triggers_per_hour_upper95",
        "early_triggers",
        "early_trigger_clusters",
        "censored_events",
        "censored_clusters",
        "error_count",
    )
    for name in higher_is_better:
        if name not in anchor_event and name not in candidate_event:
            continue
        if _finite(candidate_event[name], name) < _finite(anchor_event[name], name):
            reasons.append(f"event {name} regressed")
    for name in lower_is_better:
        if name not in anchor_event and name not in candidate_event:
            continue
        if _finite(candidate_event[name], name) > _finite(anchor_event[name], name):
            reasons.append(f"event {name} regressed")
    for name in ("latency_p50_ms", "latency_p95_ms"):
        if _latency(candidate_event.get(name)) > _latency(anchor_event.get(name)):
            reasons.append(f"event {name} regressed")
    return reasons


def lexicographic_key(metrics: dict[str, object]) -> tuple[float, ...]:
    event = metrics.get("event") or metrics.get("production_event")
    if not isinstance(event, dict):
        raise ValueError("event metrics must be an object")
    return (
        _finite(event["recall_lower95"], "recall_lower95"),
        _finite(event.get("cluster_recall", event["recall"]), "cluster_recall"),
        _finite(event["recall"], "recall"),
        -_latency(event.get("latency_p95_ms")),
        -_latency(event.get("latency_p50_ms")),
        _finite(metrics["average_precision"], "average_precision"),
        _finite(metrics["f1"], "f1"),
    )


def select_candidate(
    anchor: dict[str, object],
    candidates: list[tuple[str, dict[str, object]]],
) -> dict[str, object]:
    decisions: list[CandidateDecision] = []
    accepted: list[tuple[str, dict[str, object], tuple[float, ...]]] = []
    for path, metrics in candidates:
        reasons = tuple(non_regression_reasons(anchor, metrics))
        key = None if reasons else lexicographic_key(metrics)
        decisions.append(CandidateDecision(path, not reasons, reasons, key))
        if key is not None:
            accepted.append((path, metrics, key))
    selected = max(accepted, key=lambda item: item[2]) if accepted else None
    return {
        "policy_version": POLICY_VERSION,
        "selected": selected[0] if selected is not None else None,
        "candidates": [
            {
                "path": decision.path,
                "accepted": decision.accepted,
                "rejection_reasons": list(decision.rejection_reasons),
                "selection_key": (
                    [
                        value if math.isfinite(value) else None
                        for value in decision.selection_key
                    ]
                    if decision.selection_key is not None
                    else None
                ),
            }
            for decision in decisions
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a DTD v5 validation candidate by strict non-regression and lexicographic event metrics."
    )
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    anchor = load_metrics(args.anchor)
    candidates = [
        candidate for path in args.candidate for candidate in load_candidates(path)
    ]
    result = select_candidate(anchor, candidates)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["selected"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
