from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from check_data_determinism import parse_args as parse_determinism_args
import train_dtd
from train_dtd import (
    assert_fresh_output_directory,
    atomic_torch_save,
    atomic_write_json,
    load_resume_history,
    parse_args as parse_training_args,
    reconcile_resume_artifacts,
    resolve_resume_checkpoint,
    validate_resume_checkpoint_format,
    validate_resume_recipe,
    windows_commit_memory,
)


def test_determinism_cli_accepts_candidate_source_weights():
    args = parse_determinism_args(
        [
            "--enriched-root",
            "dataset",
            "--aec-narrow-weight",
            "0.40",
            "--aec-fullband-weight",
            "0.20",
            "--wenet-weight",
            "0.20",
            "--aec-real-weight",
            "0.20",
        ]
    )

    assert (
        args.aec_narrow_weight,
        args.aec_fullband_weight,
        args.wenet_weight,
        args.aec_real_weight,
    ) == (0.40, 0.20, 0.20, 0.20)


def test_training_cli_accepts_three_gru_layers():
    args = parse_training_args(["--model-size", "256", "--gru-layers", "3"])

    assert args.model_size == 256
    assert args.gru_layers == 3


def test_training_cli_defaults_match_candidate_01_recipe():
    args = parse_training_args([])

    assert args.enriched_root == (
        r"E:\Projects\render-aware-dtd\datasets\render-aware-dtd-v5"
    )
    assert (args.epochs, args.batch_size, args.sequence_seconds) == (230, 128, 8.0)
    assert (
        args.learning_rate,
        args.lr_schedule,
        args.warmup_epochs,
        args.min_learning_rate,
    ) == (8e-4, "cosine", 5, 1e-5)
    assert (args.weight_decay, args.dropout, args.positive_weight) == (
        2e-4,
        0.1,
        1.5,
    )
    assert (args.model_size, args.gru_layers, args.frequency_bands) == (288, 3, 80)
    assert (
        args.aec_narrow_weight,
        args.aec_fullband_weight,
        args.wenet_weight,
        args.aec_real_weight,
    ) == (0.40, 0.20, 0.20, 0.20)
    assert (args.examples_per_epoch, args.validation_examples) == (12_000, 1_500)
    assert (args.seed, args.device, args.num_workers) == (43, "cuda", 6)
    assert (
        args.hard_negative_loss_weight,
        args.onset_loss_weight,
        args.event_loss_weight,
        args.reset_loss_weight,
    ) == (0.15, 0.10, 0.0, 0.0)
    assert (
        args.v5_reset_probability,
        args.v5_hard_far_only_probability,
        args.v5_maximum_gap_hops,
    ) == (0.2, 0.5, 8)
    assert (
        args.early_stopping_patience,
        args.early_stopping_min_delta,
    ) == (30, 0.0002)


def test_atomic_json_failure_preserves_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "history.json"
    target.write_text('{"old": true}\n', encoding="utf-8")

    def fail_dump(*args, **kwargs):
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(train_dtd.json, "dump", fail_dump)
    with pytest.raises(RuntimeError, match="simulated"):
        atomic_write_json(target, {"new": True})

    assert target.read_text(encoding="utf-8") == '{"old": true}\n'
    assert not list(tmp_path.glob(".history.json.*.tmp"))


def test_atomic_checkpoint_failure_preserves_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "last.pt"
    target.write_bytes(b"old checkpoint")

    def fail_save(*args, **kwargs):
        raise RuntimeError("simulated checkpoint failure")

    monkeypatch.setattr(train_dtd.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="simulated"):
        atomic_torch_save({"epoch": 1}, target)

    assert target.read_bytes() == b"old checkpoint"
    assert not list(tmp_path.glob(".last.pt.*.tmp"))


def test_atomic_checkpoint_round_trip_and_no_replace(tmp_path):
    target = tmp_path / "epoch-000.pt"
    atomic_torch_save({"epoch": 0}, target, replace=False)

    assert torch.load(target, weights_only=False)["epoch"] == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        atomic_torch_save({"epoch": 1}, target, replace=False)
    assert torch.load(target, weights_only=False)["epoch"] == 0


def test_atomic_checkpoint_no_replace_survives_creation_race(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "epoch-000.pt"
    real_link = train_dtd.os.link

    def create_competing_checkpoint(source, destination):
        torch.save({"epoch": 99}, destination)
        return real_link(source, destination)

    monkeypatch.setattr(
        train_dtd.os,
        "link",
        create_competing_checkpoint,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        atomic_torch_save({"epoch": 0}, target, replace=False)

    assert torch.load(target, weights_only=False)["epoch"] == 99
    assert not list(tmp_path.glob(".epoch-000.pt.*.tmp"))


def test_embedded_resume_history_is_authoritative(tmp_path):
    external = tmp_path / "history.json"
    external.write_text("not valid JSON", encoding="utf-8")
    checkpoint = {"history": [{"epoch": 0}, {"epoch": 1}]}

    assert load_resume_history(external, 2, checkpoint) == [
        {"epoch": 0},
        {"epoch": 1},
    ]


def test_resume_history_must_be_contiguous_and_complete(tmp_path):
    checkpoint = {"history": [{"epoch": 0}, {"epoch": 2}]}

    with pytest.raises(ValueError, match="do not match"):
        load_resume_history(tmp_path / "missing.json", 3, checkpoint)


def test_legacy_resume_history_requires_external_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="legacy"):
        load_resume_history(tmp_path / "missing.json", 1, {})


def test_resume_recipe_allows_only_orchestration_changes():
    stored = {
        "seed": 43,
        "learning_rate": 8e-4,
        "model_size": 192,
        "wenet_far_only_probability": 0.50,
        "num_workers": 8,
        "output_dir": "candidate-4",
        "resume": None,
    }
    current = dict(stored)
    current.update(
        num_workers=4,
        output_dir="different-spelling",
        resume="candidate-4/last.pt",
    )
    validate_resume_recipe({"training_args": stored}, Namespace(**current))

    current["seed"] = 44
    with pytest.raises(ValueError, match="seed"):
        validate_resume_recipe({"training_args": stored}, Namespace(**current))


def test_resume_recipe_normalizes_legacy_far_only_default():
    stored = {"seed": 43, "num_workers": 8, "output_dir": "old", "resume": None}
    current = dict(stored)
    current["wenet_far_only_probability"] = 0.40
    validate_resume_recipe({"training_args": stored}, Namespace(**current))

    current["wenet_far_only_probability"] = 0.50
    with pytest.raises(ValueError, match="wenet_far_only_probability"):
        validate_resume_recipe({"training_args": stored}, Namespace(**current))


def test_resume_recipe_normalizes_all_legacy_training_defaults():
    current = parse_training_args(
        [
            "--lr-schedule",
            "constant",
            "--warmup-epochs",
            "0",
            "--min-learning-rate",
            "1e-5",
            "--model-size",
            "192",
            "--gru-layers",
            "2",
            "--dropout",
            "0.1",
            "--wenet-far-only-probability",
            "0.40",
        ]
    )
    stored = dict(vars(current))
    for name in (
        "lr_schedule",
        "warmup_epochs",
        "min_learning_rate",
        "model_size",
        "gru_layers",
        "dropout",
        "wenet_far_only_probability",
    ):
        del stored[name]

    validate_resume_recipe({"training_args": stored}, current)


def test_three_layer_resume_recipe_is_exact_except_worker_orchestration():
    stored_args = parse_training_args(
        [
            "--model-size",
            "256",
            "--gru-layers",
            "3",
            "--epochs",
            "150",
            "--num-workers",
            "8",
            "--output-dir",
            "candidate-7",
        ]
    )
    stored = dict(vars(stored_args))
    current = dict(stored)
    current.update(
        num_workers=4,
        output_dir="candidate-7-renamed",
        resume="candidate-7-renamed/last.pt",
    )

    validate_resume_recipe({"training_args": stored}, Namespace(**current))

    current["gru_layers"] = 2
    with pytest.raises(ValueError, match="gru_layers"):
        validate_resume_recipe({"training_args": stored}, Namespace(**current))


def recovery_checkpoint(
    epoch: int,
    scores: list[float],
    training_args: dict[str, object],
) -> dict[str, object]:
    return {
        "format_version": 3,
        "epoch": epoch,
        "selection_score": scores[epoch],
        "training_args": training_args,
        "history": [
            {"epoch": index, "selection_score": score}
            for index, score in enumerate(scores[: epoch + 1])
        ],
    }


def test_format_v2_resume_is_rejected_before_mixing_selection_policies():
    checkpoint = recovery_checkpoint(0, [0.5], {})
    checkpoint["format_version"] = 2

    with pytest.raises(ValueError, match="selection score policy"):
        validate_resume_checkpoint_format(checkpoint, "candidate-2/best.pt")


def test_format_v4_resume_requires_explicit_selection_policy():
    checkpoint = recovery_checkpoint(0, [0.5], {})
    checkpoint["format_version"] = 4

    with pytest.raises(ValueError, match="selection policy"):
        validate_resume_checkpoint_format(checkpoint, "candidate-6/last.pt")

    checkpoint["selection_policy_version"] = "low_fpr_grid_v1"
    validate_resume_checkpoint_format(checkpoint, "candidate-6/last.pt")


def test_resume_advances_stale_last_and_repairs_stale_best(tmp_path):
    stored = {
        "seed": 43,
        "num_workers": 8,
        "output_dir": str(tmp_path),
        "resume": None,
    }
    args = Namespace(**{**stored, "resume": str(tmp_path / "last.pt")})
    checkpoint_zero = recovery_checkpoint(0, [0.4], stored)
    checkpoint_one = recovery_checkpoint(1, [0.4, 0.6], stored)
    atomic_torch_save(checkpoint_zero, tmp_path / "last.pt")
    atomic_torch_save(checkpoint_zero, tmp_path / "best.pt")
    atomic_torch_save(
        checkpoint_zero,
        tmp_path / "checkpoints" / "epoch-000.pt",
        replace=False,
    )
    atomic_torch_save(
        checkpoint_one,
        tmp_path / "checkpoints" / "epoch-001.pt",
        replace=False,
    )

    resolved_path, resolved, resolution_actions = resolve_resume_checkpoint(
        tmp_path,
        tmp_path / "last.pt",
        args,
    )
    best, actions = reconcile_resume_artifacts(
        tmp_path,
        resolved,
        resolved["history"],
        args,
    )

    assert resolved_path.name == "epoch-001.pt"
    assert resolved["epoch"] == 1
    assert resolution_actions == []
    assert torch.load(tmp_path / "last.pt", weights_only=False)["epoch"] == 1
    assert best["epoch"] == 1
    assert torch.load(tmp_path / "best.pt", weights_only=False)["epoch"] == 1
    assert actions == [
        "advanced last.pt to epoch 1",
        "restored best.pt from epoch 1",
    ]


def test_resume_recovers_when_first_last_and_best_are_missing(tmp_path):
    stored = {
        "seed": 43,
        "num_workers": 8,
        "output_dir": str(tmp_path),
        "resume": None,
    }
    args = Namespace(**{**stored, "resume": str(tmp_path / "last.pt")})
    checkpoint = recovery_checkpoint(0, [0.5], stored)
    atomic_torch_save(
        checkpoint,
        tmp_path / "checkpoints" / "epoch-000.pt",
        replace=False,
    )

    _, resolved, resolution_actions = resolve_resume_checkpoint(
        tmp_path,
        tmp_path / "last.pt",
        args,
    )
    best, actions = reconcile_resume_artifacts(
        tmp_path,
        resolved,
        resolved["history"],
        args,
    )

    assert best["epoch"] == 0
    assert resolution_actions == []
    assert torch.load(tmp_path / "last.pt", weights_only=False)["epoch"] == 0
    assert torch.load(tmp_path / "best.pt", weights_only=False)["epoch"] == 0
    assert actions == [
        "advanced last.pt to epoch 0",
        "restored best.pt from epoch 0",
    ]


def test_resume_restores_missing_immutable_checkpoint_from_last(tmp_path):
    stored = {
        "seed": 43,
        "num_workers": 8,
        "output_dir": str(tmp_path),
        "resume": None,
    }
    args = Namespace(**{**stored, "resume": str(tmp_path / "last.pt")})
    checkpoint = recovery_checkpoint(0, [0.5], stored)
    atomic_torch_save(checkpoint, tmp_path / "last.pt")

    _, resolved, resolution_actions = resolve_resume_checkpoint(
        tmp_path,
        tmp_path / "last.pt",
        args,
    )
    _, actions = reconcile_resume_artifacts(
        tmp_path,
        resolved,
        resolved["history"],
        args,
    )

    immutable = tmp_path / "checkpoints" / "epoch-000.pt"
    assert resolution_actions == []
    assert torch.load(immutable, weights_only=False)["epoch"] == 0
    assert actions == [
        "restored immutable checkpoint epoch-000.pt",
        "restored best.pt from epoch 0",
    ]


def test_resume_rejects_explicit_rollback_with_newer_checkpoints(tmp_path):
    stored = {
        "seed": 43,
        "num_workers": 8,
        "output_dir": str(tmp_path),
        "resume": None,
    }
    checkpoint_zero = recovery_checkpoint(0, [0.4], stored)
    checkpoint_one = recovery_checkpoint(1, [0.4, 0.6], stored)
    old_path = tmp_path / "checkpoints" / "epoch-000.pt"
    atomic_torch_save(checkpoint_zero, old_path, replace=False)
    atomic_torch_save(
        checkpoint_one,
        tmp_path / "checkpoints" / "epoch-001.pt",
        replace=False,
    )
    args = Namespace(**{**stored, "resume": str(old_path)})

    with pytest.raises(ValueError, match="explicit rollback"):
        resolve_resume_checkpoint(tmp_path, old_path, args)


def test_resume_archives_corrupt_mutable_aliases_and_repairs_them(tmp_path):
    stored = {
        "seed": 43,
        "num_workers": 8,
        "output_dir": str(tmp_path),
        "resume": None,
    }
    args = Namespace(**{**stored, "resume": str(tmp_path / "last.pt")})
    checkpoint = recovery_checkpoint(0, [0.5], stored)
    atomic_torch_save(
        checkpoint,
        tmp_path / "checkpoints" / "epoch-000.pt",
        replace=False,
    )
    (tmp_path / "last.pt").write_bytes(b"truncated-last")
    (tmp_path / "best.pt").write_bytes(b"truncated-best")

    _, resolved, resolution_actions = resolve_resume_checkpoint(
        tmp_path,
        tmp_path / "last.pt",
        args,
    )
    best, reconciliation_actions = reconcile_resume_artifacts(
        tmp_path,
        resolved,
        resolved["history"],
        args,
    )

    assert best["epoch"] == 0
    assert (tmp_path / "last.pt.corrupt-001").read_bytes() == b"truncated-last"
    assert (tmp_path / "best.pt.corrupt-001").read_bytes() == b"truncated-best"
    assert torch.load(tmp_path / "last.pt", weights_only=False)["epoch"] == 0
    assert torch.load(tmp_path / "best.pt", weights_only=False)["epoch"] == 0
    assert resolution_actions[0].startswith(
        "archived unreadable last.pt as last.pt.corrupt-001"
    )
    assert reconciliation_actions[0] == "advanced last.pt to epoch 0"
    assert reconciliation_actions[1].startswith(
        "archived unreadable best.pt as best.pt.corrupt-001"
    )
    assert reconciliation_actions[2] == "restored best.pt from epoch 0"


def test_corrupt_last_without_immutable_fallback_is_preserved(tmp_path):
    stored = {
        "seed": 43,
        "num_workers": 8,
        "output_dir": str(tmp_path),
        "resume": None,
    }
    last_path = tmp_path / "last.pt"
    last_path.write_bytes(b"only-corrupt-copy")
    args = Namespace(**{**stored, "resume": str(last_path)})

    with pytest.raises(ValueError, match="no immutable"):
        resolve_resume_checkpoint(tmp_path, last_path, args)

    assert last_path.read_bytes() == b"only-corrupt-copy"
    assert not list(tmp_path.glob("last.pt.corrupt-*"))


@pytest.mark.parametrize(
    "relative_path",
    [
        "best.pt",
        "last.pt",
        "history.json",
        "training_summary.json",
        "checkpoints/epoch-000.pt",
    ],
)
def test_fresh_run_rejects_training_artifacts(tmp_path, relative_path):
    artifact = tmp_path / relative_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        assert_fresh_output_directory(tmp_path)


def test_fresh_run_allows_logs_and_experiment_notes(tmp_path):
    (tmp_path / "EXPERIMENT.md").write_text("planned", encoding="utf-8")
    (tmp_path / "training.stdout.log").write_text("", encoding="utf-8")
    (tmp_path / "resource-monitor.csv").write_text("", encoding="utf-8")

    assert_fresh_output_directory(tmp_path)


@pytest.mark.skipif(train_dtd.os.name != "nt", reason="Windows-only accounting")
def test_windows_commit_accounting_is_bounded():
    committed, limit = windows_commit_memory()

    assert 0 < committed <= limit
