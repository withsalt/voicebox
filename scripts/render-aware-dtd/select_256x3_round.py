from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import torch

from train_dtd import (
    CHECKPOINT_FORMAT_VERSION,
    SELECTION_POLICY_VERSION,
    json_safe,
    normalized_training_args,
    validation_selection_score,
)


AGGREGATE_TOLERANCE = 0.0005
LOW_FPR_TOLERANCE = 0.003
ONLINE_TOLERANCE = 0.0005
AEC_STRATUM_TOLERANCE = 0.001
MATERIAL_IMPROVEMENT = 0.0002
SCORE_TIE_BAND = 0.0002
LOW_FPR_KEYS = ("0.100%", "0.500%", "1.000%", "2.000%", "5.000%")
PRIMARY_LOW_FPR_KEYS = LOW_FPR_KEYS[:3]
AEC_STRATA = (
    "source:aec_narrow",
    "source:aec_fullband",
    "source:aec_real:real",
    "source:aec_real:real_doubled",
)
RECIPE_ORCHESTRATION_KEYS = frozenset({"resume", "output_dir", "num_workers"})
INCUMBENT_SHA256 = (
    "5CDA490330BE1D4D4C83A8C8DC9F4EB7B1E173711F866EED8066C6ECA03DEFC9"
)
INCUMBENT_EPOCH = 88
INCUMBENT_SELECTION_SCORE = 1.4301377221279903
DATASET_SUMMARY_SHA256 = (
    "347ED8D7935FEDF49708FCB3B828BCB94A6270B2825282A4F7D1DBF6D87613DB"
)
DATASET_ROOT = r"E:\Projects\render-aware-dtd\datasets\AEC-Challenge"
ENRICHED_ROOT = (
    r"E:\Projects\render-aware-dtd\datasets\render-aware-dtd-v4"
)

EXPECTED_CANDIDATE_OVERRIDES: dict[int, dict[str, object]] = {
    7: {},
    8: {"learning_rate": 6e-4, "warmup_epochs": 8},
    9: {"aec_narrow_weight": 0.35, "wenet_weight": 0.25},
    10: {"positive_weight": 1.75},
    11: {"dropout": 0.15},
    12: {"weight_decay": 2e-4},
}

COMMON_RECIPE: dict[str, object] = {
    "dataset_root": DATASET_ROOT,
    "enriched_root": ENRICHED_ROOT,
    "epochs": 150,
    "batch_size": 128,
    "sequence_seconds": 8.0,
    "learning_rate": 8e-4,
    "lr_schedule": "cosine",
    "warmup_epochs": 5,
    "min_learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "positive_weight": 1.5,
    "model_size": 256,
    "gru_layers": 3,
    "dropout": 0.1,
    "aec_narrow_weight": 0.40,
    "aec_fullband_weight": 0.20,
    "wenet_weight": 0.20,
    "aec_real_weight": 0.20,
    "wenet_far_only_probability": 0.40,
    "examples_per_epoch": 12_000,
    "validation_examples": 1_500,
    "validation_modulo": 20,
    "train_limit": 0,
    "validation_limit": 0,
    "seed": 43,
    "device": "cuda",
    "early_stopping_patience": 0,
    "early_stopping_min_delta": 0.0002,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and select the 256x3 round using validation data only."
        )
    )
    parser.add_argument(
        "--artifact-root",
        default=r"E:\Projects\render-aware-dtd\artifacts-v4",
    )
    parser.add_argument(
        "--incumbent-checkpoint",
        default=(
            r"E:\Projects\render-aware-dtd\artifacts-v4"
            r"\candidate-7\incumbent-candidate-6-validation-selected.pt"
        ),
    )
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=list(range(7, 13)),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--deep-audit",
        action="store_true",
        help="Load and validate every immutable epoch checkpoint.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require_finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_incumbent_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], float]:
    if checkpoint_sha256 != INCUMBENT_SHA256:
        raise ValueError(
            "incumbent checkpoint hash mismatch: "
            f"expected={INCUMBENT_SHA256}, actual={checkpoint_sha256}"
        )
    if int(checkpoint["format_version"]) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("incumbent checkpoint format mismatch")
    if checkpoint.get("selection_policy_version") != SELECTION_POLICY_VERSION:
        raise ValueError("incumbent checkpoint selection policy mismatch")
    if int(checkpoint["epoch"]) != INCUMBENT_EPOCH:
        raise ValueError("incumbent checkpoint epoch mismatch")
    expected_config = {
        "sample_rate": 16_000,
        "frame_length": 640,
        "hop_length": 320,
        "frequency_bands": 48,
        "projection_size": 256,
        "hidden_size": 256,
        "gru_layers": 2,
        "dropout": 0.1,
    }
    if checkpoint["model_config"] != expected_config:
        raise ValueError("incumbent checkpoint model config mismatch")
    metrics = checkpoint["validation_metrics"]
    score = require_finite(
        checkpoint["selection_score"],
        "incumbent score",
    )
    if abs(score - INCUMBENT_SELECTION_SCORE) > 1e-12:
        raise ValueError("incumbent checkpoint selection score mismatch")
    recomputed_score = validation_selection_score(metrics)
    if abs(score - recomputed_score) > 1e-12:
        raise ValueError("incumbent checkpoint score cannot be reproduced")
    model_state = checkpoint["model_state"]
    if not all(
        bool(torch.isfinite(tensor).all())
        for tensor in model_state.values()
    ):
        raise ValueError("incumbent checkpoint has non-finite weights")
    return metrics, score


def low_fpr_recall(metrics: dict[str, Any], key: str) -> float:
    return require_finite(
        metrics["low_fpr_operating_points"][key]["recall"],
        f"recall@{key}",
    )


def validation_gate_reasons(
    metrics: dict[str, Any],
    incumbent: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    for key in ("average_precision", "roc_auc", "f1"):
        candidate_value = require_finite(metrics[key], key)
        incumbent_value = require_finite(incumbent[key], f"incumbent {key}")
        if candidate_value < incumbent_value - AGGREGATE_TOLERANCE:
            reasons.append(f"{key} below incumbent tolerance")

    for key in LOW_FPR_KEYS:
        if (
            low_fpr_recall(metrics, key)
            < low_fpr_recall(incumbent, key) - LOW_FPR_TOLERANCE
        ):
            reasons.append(f"recall@{key} below incumbent tolerance")

    candidate_online = metrics["strata"]["source:online_validation"]
    incumbent_online = incumbent["strata"]["source:online_validation"]
    for key in ("average_precision", "roc_auc"):
        if (
            require_finite(candidate_online[key], f"online {key}")
            < require_finite(incumbent_online[key], f"incumbent online {key}")
            - ONLINE_TOLERANCE
        ):
            reasons.append(f"online {key} below incumbent tolerance")

    for stratum in AEC_STRATA:
        candidate_stratum = metrics["strata"][stratum]
        incumbent_stratum = incumbent["strata"][stratum]
        for key in ("average_precision", "roc_auc"):
            if (
                require_finite(
                    candidate_stratum[key],
                    f"{stratum} {key}",
                )
                < require_finite(
                    incumbent_stratum[key],
                    f"incumbent {stratum} {key}",
                )
                - AEC_STRATUM_TOLERANCE
            ):
                reasons.append(
                    f"{stratum} {key} below incumbent tolerance"
                )
    return reasons


def promotion_gate_reasons(
    metrics: dict[str, Any],
    score: float,
    incumbent: dict[str, Any],
    incumbent_score: float,
) -> list[str]:
    reasons = validation_gate_reasons(metrics, incumbent)
    if score <= incumbent_score + MATERIAL_IMPROVEMENT:
        reasons.append("selection score does not clear strict improvement")

    candidate_online = metrics["strata"]["source:online_validation"]
    incumbent_online = incumbent["strata"]["source:online_validation"]
    online_improved = any(
        require_finite(candidate_online[key], f"online {key}")
        > require_finite(incumbent_online[key], f"incumbent online {key}")
        + MATERIAL_IMPROVEMENT
        for key in ("average_precision", "roc_auc")
    )
    if not online_improved:
        reasons.append("neither online AP nor AUC materially improves")

    low_fpr_improved = any(
        low_fpr_recall(metrics, key)
        > low_fpr_recall(incumbent, key) + MATERIAL_IMPROVEMENT
        for key in PRIMARY_LOW_FPR_KEYS
    )
    if not low_fpr_improved:
        reasons.append("no primary low-FPR recall materially improves")
    return reasons


def expected_recipe(candidate: int) -> dict[str, object]:
    if candidate not in EXPECTED_CANDIDATE_OVERRIDES:
        raise ValueError(f"no preregistered recipe for candidate-{candidate}")
    return {
        **COMMON_RECIPE,
        **EXPECTED_CANDIDATE_OVERRIDES[candidate],
    }


def recipe_without_orchestration(values: dict[str, object]) -> dict[str, object]:
    normalized = normalized_training_args(values)
    return {
        key: value
        for key, value in normalized.items()
        if key not in RECIPE_ORCHESTRATION_KEYS
    }


def validate_expected_recipe(
    values: dict[str, object],
    candidate: int,
    label: str,
) -> None:
    normalized = normalized_training_args(values)
    for key, expected in expected_recipe(candidate).items():
        actual = normalized.get(key, "<missing>")
        if actual != expected:
            raise ValueError(
                f"{label} recipe mismatch for {key}: "
                f"expected={expected!r}, actual={actual!r}"
            )


def validate_checkpoint(
    checkpoint: dict[str, Any],
    *,
    candidate: int,
    epoch: int,
    history: list[dict[str, Any]],
    canonical_recipe: dict[str, object] | None,
    check_parameters: bool,
) -> dict[str, object]:
    if int(checkpoint["format_version"]) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"candidate-{candidate} epoch {epoch} format mismatch"
        )
    if checkpoint.get("selection_policy_version") != SELECTION_POLICY_VERSION:
        raise ValueError(
            f"candidate-{candidate} epoch {epoch} policy mismatch"
        )
    if int(checkpoint["epoch"]) != epoch:
        raise ValueError(
            f"candidate-{candidate} checkpoint epoch mismatch at {epoch}"
        )
    config = checkpoint["model_config"]
    expected_config = {
        "sample_rate": 16_000,
        "frame_length": 640,
        "hop_length": 320,
        "frequency_bands": 48,
        "projection_size": 256,
        "hidden_size": 256,
        "gru_layers": 3,
        "dropout": expected_recipe(candidate)["dropout"],
    }
    if config != expected_config:
        raise ValueError(
            f"candidate-{candidate} epoch {epoch} model config mismatch"
        )

    score = require_finite(
        checkpoint["selection_score"],
        f"candidate-{candidate} epoch {epoch} checkpoint score",
    )
    history_score = require_finite(
        history[epoch]["selection_score"],
        f"candidate-{candidate} epoch {epoch} history score",
    )
    if abs(score - history_score) > 1e-12:
        raise ValueError(
            f"candidate-{candidate} epoch {epoch} score mismatch"
        )

    embedded_history = checkpoint.get("history")
    if not isinstance(embedded_history, list):
        raise ValueError(
            f"candidate-{candidate} epoch {epoch} has no embedded history"
        )
    embedded_epochs = [
        int(record["epoch"]) for record in embedded_history
    ]
    if embedded_epochs != list(range(epoch + 1)):
        raise ValueError(
            f"candidate-{candidate} epoch {epoch} embedded history mismatch"
        )
    if embedded_history != history[: epoch + 1]:
        raise ValueError(
            f"candidate-{candidate} epoch {epoch} embedded history content "
            "does not match external history"
        )
    if (
        json_safe(checkpoint["validation_metrics"])
        != history[epoch]["validation"]
    ):
        raise ValueError(
            f"candidate-{candidate} epoch {epoch} validation metrics do not "
            "match external history"
        )

    training_args = checkpoint["training_args"]
    if not isinstance(training_args, dict):
        raise ValueError(
            f"candidate-{candidate} epoch {epoch} training args missing"
        )
    validate_expected_recipe(
        training_args,
        candidate,
        f"candidate-{candidate} epoch {epoch}",
    )
    recipe = recipe_without_orchestration(training_args)
    if canonical_recipe is not None and recipe != canonical_recipe:
        raise ValueError(
            f"candidate-{candidate} epoch {epoch} recipe changed mid-run"
        )

    if check_parameters:
        model_state = checkpoint["model_state"]
        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in model_state.values()
        ):
            raise ValueError(
                f"candidate-{candidate} epoch {epoch} has non-finite weights"
            )
    return recipe


def audit_candidate(
    candidate_directory: Path,
    candidate: int,
    incumbent_metrics: dict[str, Any],
    incumbent_score: float,
    *,
    deep_audit: bool,
) -> dict[str, Any]:
    history_path = candidate_directory / "history.json"
    summary_path = candidate_directory / "training_summary.json"
    if not history_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"candidate-{candidate} is incomplete")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != 150:
        raise ValueError(
            f"candidate-{candidate} must contain exactly 150 history records"
        )
    if [int(record["epoch"]) for record in history] != list(range(150)):
        raise ValueError(f"candidate-{candidate} history is not contiguous")

    max_score_error = 0.0
    eligible: list[dict[str, Any]] = []
    for record in history:
        epoch = int(record["epoch"])
        metrics = record["validation"]
        stored_score = require_finite(
            record["selection_score"],
            f"candidate-{candidate} epoch {epoch} score",
        )
        recomputed_score = validation_selection_score(metrics)
        max_score_error = max(
            max_score_error,
            abs(stored_score - recomputed_score),
        )
        reasons = validation_gate_reasons(metrics, incumbent_metrics)
        if not reasons:
            eligible.append(
                {
                    "epoch": epoch,
                    "score": stored_score,
                    "metrics": metrics,
                }
            )
    if max_score_error > 1e-12:
        raise ValueError(
            f"candidate-{candidate} stored validation score is inconsistent"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["epochs_completed"]) != 150:
        raise ValueError(f"candidate-{candidate} did not complete 150 epochs")
    validate_expected_recipe(
        summary["training_args"],
        candidate,
        f"candidate-{candidate} summary",
    )

    checkpoint_paths = sorted(
        (candidate_directory / "checkpoints").glob("epoch-*.pt")
    )
    expected_names = [f"epoch-{epoch:03d}.pt" for epoch in range(150)]
    if [path.name for path in checkpoint_paths] != expected_names:
        raise ValueError(
            f"candidate-{candidate} immutable checkpoints are incomplete"
        )

    canonical_recipe: dict[str, object] | None = None
    if deep_audit:
        for epoch, path in enumerate(checkpoint_paths):
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            recipe = validate_checkpoint(
                checkpoint,
                candidate=candidate,
                epoch=epoch,
                history=history,
                canonical_recipe=canonical_recipe,
                check_parameters=True,
            )
            canonical_recipe = recipe
            del checkpoint

    if not eligible:
        return {
            "candidate": candidate,
            "eligible_epoch_count": 0,
            "qualified": False,
            "qualification_reasons": ["no validation-eligible epoch"],
            "score_recompute_max_abs_error": max_score_error,
            "checkpoint_count": len(checkpoint_paths),
            "deep_audit": deep_audit,
        }

    selected = min(
        eligible,
        key=lambda item: (-float(item["score"]), int(item["epoch"])),
    )
    selected_epoch = int(selected["epoch"])
    selected_path = checkpoint_paths[selected_epoch]
    selected_checkpoint = torch.load(
        selected_path,
        map_location="cpu",
        weights_only=False,
    )
    validate_checkpoint(
        selected_checkpoint,
        candidate=candidate,
        epoch=selected_epoch,
        history=history,
        canonical_recipe=canonical_recipe,
        check_parameters=True,
    )
    del selected_checkpoint

    selected_metrics = selected["metrics"]
    selected_score = float(selected["score"])
    promotion_reasons = promotion_gate_reasons(
        selected_metrics,
        selected_score,
        incumbent_metrics,
        incumbent_score,
    )
    online = selected_metrics["strata"]["source:online_validation"]
    return {
        "candidate": candidate,
        "eligible_epoch_count": len(eligible),
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "selection_score": selected_score,
        "score_margin_vs_candidate_6": selected_score - incumbent_score,
        "average_precision": selected_metrics["average_precision"],
        "roc_auc": selected_metrics["roc_auc"],
        "grid_best_f1": selected_metrics["f1"],
        "recall_at_fpr": {
            key: low_fpr_recall(selected_metrics, key)
            for key in LOW_FPR_KEYS
        },
        "online_average_precision": online["average_precision"],
        "online_roc_auc": online["roc_auc"],
        "qualified": not promotion_reasons,
        "qualification_reasons": promotion_reasons,
        "score_recompute_max_abs_error": max_score_error,
        "checkpoint_count": len(checkpoint_paths),
        "deep_audit": deep_audit,
    }


def choose_winner(
    candidate_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    qualified = [result for result in candidate_results if result["qualified"]]
    if not qualified:
        return None
    best_score = max(float(result["selection_score"]) for result in qualified)
    tie_pool = [
        result
        for result in qualified
        if float(result["selection_score"]) >= best_score - SCORE_TIE_BAND
    ]
    return min(
        tie_pool,
        key=lambda result: (
            -float(result["online_average_precision"]),
            -float(result["recall_at_fpr"]["1.000%"]),
            int(result["candidate"]),
        ),
    )


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    incumbent_path = Path(args.incumbent_checkpoint).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite selection: {output_path}")
    if not args.deep_audit:
        raise ValueError("formal round selection requires --deep-audit")

    dataset_summary_path = Path(ENRICHED_ROOT) / "dataset_summary.json"
    dataset_summary_sha256 = sha256_file(dataset_summary_path)
    if dataset_summary_sha256 != DATASET_SUMMARY_SHA256:
        raise ValueError(
            "dataset summary hash mismatch: "
            f"expected={DATASET_SUMMARY_SHA256}, "
            f"actual={dataset_summary_sha256}"
        )

    incumbent_sha256 = sha256_file(incumbent_path)
    incumbent_checkpoint = torch.load(
        incumbent_path,
        map_location="cpu",
        weights_only=False,
    )
    incumbent_metrics, incumbent_score = validate_incumbent_checkpoint(
        incumbent_checkpoint,
        incumbent_sha256,
    )
    del incumbent_checkpoint

    results = [
        audit_candidate(
            artifact_root / f"candidate-{candidate}",
            candidate,
            incumbent_metrics,
            incumbent_score,
            deep_audit=args.deep_audit,
        )
        for candidate in args.candidates
    ]
    winner = choose_winner(results)
    decision = {
        "schema_version": 1,
        "selection_scope": "validation_only",
        "fresh_holdout_opened": False,
        "dataset_summary_sha256": dataset_summary_sha256,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "incumbent": {
            "candidate": "candidate-6",
            "selection_score": incumbent_score,
            "checkpoint": str(incumbent_path),
            "checkpoint_sha256": incumbent_sha256,
        },
        "promotion_policy": {
            "strict_score_boundary": (
                incumbent_score + MATERIAL_IMPROVEMENT
            ),
            "aggregate_tolerance": AGGREGATE_TOLERANCE,
            "low_fpr_tolerance": LOW_FPR_TOLERANCE,
            "online_tolerance": ONLINE_TOLERANCE,
            "aec_stratum_tolerance": AEC_STRATUM_TOLERANCE,
            "material_improvement": MATERIAL_IMPROVEMENT,
            "score_tie_band": SCORE_TIE_BAND,
        },
        "candidate_results": results,
        "decision": (
            f"candidate-{winner['candidate']}"
            if winner is not None
            else "candidate-6"
        ),
        "selected_epoch": (
            int(winner["selected_epoch"]) if winner is not None else 88
        ),
        "selected_checkpoint": (
            winner["selected_checkpoint"]
            if winner is not None
            else str(incumbent_path)
        ),
        "selected_checkpoint_sha256": (
            winner["selected_checkpoint_sha256"]
            if winner is not None
            else incumbent_sha256
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="x",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(json.dumps(decision, indent=2, ensure_ascii=True) + "\n")
        temporary_path = Path(stream.name)
    temporary_path.replace(output_path)
    print(json.dumps(decision, ensure_ascii=True))


if __name__ == "__main__":
    main()
