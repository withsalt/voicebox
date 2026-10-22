from __future__ import annotations

import json
from pathlib import Path

import pytest

from train_dtd import parse_args as parse_training_args
from training_preset import (
    build_training_command,
    load_training_preset,
    parse_args as parse_preset_args,
    validate_output_directory,
    verify_frozen_inputs,
)

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
PRESET_PATH = SCRIPT_ROOT / "configs" / "candidate-12.json"
DEFAULT_PRESET_PATH = SCRIPT_ROOT / "configs" / "default-candidate-16.json"
CANDIDATE_16_PRESET_PATH = SCRIPT_ROOT / "configs" / "candidate-16.json"


def load_candidate_12():
    return load_training_preset(PRESET_PATH)


def training_namespace(command: list[str]):
    return parse_training_args(command[2:])


def test_default_preset_remains_explicit_and_bare_cli_uses_candidate_01_defaults(tmp_path):
    preset = load_training_preset(DEFAULT_PRESET_PATH)
    candidate_16 = load_training_preset(CANDIDATE_16_PRESET_PATH)
    preset_args = training_namespace(
        build_training_command(
            preset,
            tmp_path / "candidate-new",
            python_executable="python",
        )
    )
    cli_args = parse_training_args([])

    assert preset.name == "default-candidate-16"
    assert preset.training_arguments == candidate_16.training_arguments
    assert preset.initial_num_workers == candidate_16.initial_num_workers
    assert preset.recovery_num_workers == candidate_16.recovery_num_workers
    assert preset_args.enriched_root == (
        r"E:\Projects\render-aware-dtd\datasets\render-aware-dtd-v4"
    )
    assert (preset_args.epochs, preset_args.batch_size) == (200, 128)
    assert cli_args.enriched_root == (
        r"E:\Projects\render-aware-dtd\datasets\render-aware-dtd-v5"
    )
    assert (cli_args.epochs, cli_args.batch_size, cli_args.num_workers) == (
        230,
        128,
        6,
    )
    assert verify_frozen_inputs(preset)


def test_preset_launcher_defaults_to_candidate_16(tmp_path):
    args = parse_preset_args(["--output-dir", str(tmp_path / "candidate-new")])

    assert Path(args.preset).resolve() == DEFAULT_PRESET_PATH.resolve()


def test_candidate_12_preset_matches_frozen_training_recipe(tmp_path):
    preset = load_candidate_12()
    output_dir = tmp_path / "candidate-new"
    args = training_namespace(
        build_training_command(
            preset,
            output_dir,
            python_executable="python",
        )
    )

    assert preset.name == "candidate-12"
    assert preset.selection_policy_version == "low_fpr_grid_v1"
    assert (
        args.epochs,
        args.batch_size,
        args.sequence_seconds,
        args.seed,
    ) == (150, 128, 8.0, 43)
    assert (
        args.model_size,
        args.gru_layers,
        args.dropout,
    ) == (256, 3, 0.1)
    assert (
        args.learning_rate,
        args.lr_schedule,
        args.warmup_epochs,
        args.min_learning_rate,
    ) == (8e-4, "cosine", 5, 1e-5)
    assert (
        args.weight_decay,
        args.positive_weight,
    ) == (2e-4, 1.5)
    assert (
        args.aec_narrow_weight,
        args.aec_fullband_weight,
        args.wenet_weight,
        args.aec_real_weight,
    ) == (0.40, 0.20, 0.20, 0.20)
    assert args.wenet_far_only_probability == 0.40
    assert (
        args.examples_per_epoch,
        args.validation_examples,
        args.validation_modulo,
    ) == (12_000, 1_500, 20)
    assert (args.train_limit, args.validation_limit) == (0, 0)
    assert args.num_workers == 8
    assert args.device == "cuda"
    assert (
        args.early_stopping_patience,
        args.early_stopping_min_delta,
    ) == (0, 0.0002)
    assert args.output_dir == str(output_dir.resolve())
    assert args.resume is None


def test_candidate_12_preset_uses_recovery_workers(tmp_path):
    preset = load_candidate_12()
    output_dir = tmp_path / "candidate-new"
    output_dir.mkdir()
    checkpoint = output_dir / "last.pt"
    checkpoint.write_bytes(b"checkpoint")

    args = training_namespace(
        build_training_command(
            preset,
            output_dir,
            resume=checkpoint,
            python_executable="python",
        )
    )

    assert args.num_workers == 4
    assert args.resume == str(checkpoint.resolve())


def test_candidate_12_preset_allows_trainer_to_recover_missing_last(tmp_path):
    preset = load_candidate_12()
    output_dir = tmp_path / "candidate-new"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "epoch-000.pt").write_bytes(b"checkpoint")
    missing_last = output_dir / "last.pt"

    args = training_namespace(
        build_training_command(
            preset,
            output_dir,
            resume=missing_last,
            python_executable="python",
        )
    )

    assert args.num_workers == 4
    assert args.resume == str(missing_last.resolve())


def test_candidate_12_preset_rejects_missing_last_without_epoch_checkpoint(tmp_path):
    preset = load_candidate_12()
    output_dir = tmp_path / "candidate-new"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="resume checkpoint is missing"):
        build_training_command(
            preset,
            output_dir,
            resume=output_dir / "last.pt",
        )


def test_candidate_12_preset_ignores_checkpoint_named_directory(tmp_path):
    preset = load_candidate_12()
    output_dir = tmp_path / "candidate-new"
    (output_dir / "checkpoints" / "epoch-000.pt").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="resume checkpoint is missing"):
        build_training_command(
            preset,
            output_dir,
            resume=output_dir / "last.pt",
        )


def test_candidate_12_preset_refuses_existing_fresh_output(tmp_path):
    preset = load_candidate_12()
    output_dir = tmp_path / "candidate-new"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        build_training_command(preset, output_dir)


def test_candidate_12_preset_refuses_external_resume_checkpoint(tmp_path):
    preset = load_candidate_12()
    output_dir = tmp_path / "candidate-new"
    output_dir.mkdir()
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="must be inside"):
        build_training_command(preset, output_dir, resume=checkpoint)


def test_candidate_12_preset_refuses_frozen_candidate_directory():
    preset = load_candidate_12()
    frozen_candidate = next(
        path
        for path in preset.protected_output_directories
        if path.name == "candidate-12"
    )

    with pytest.raises(ValueError, match="protected candidate"):
        validate_output_directory(preset, frozen_candidate, resume=None)


def test_candidate_12_preset_refuses_frozen_candidate_subdirectory():
    preset = load_candidate_12()
    frozen_candidate = next(
        path
        for path in preset.protected_output_directories
        if path.name == "candidate-12"
    )

    with pytest.raises(ValueError, match="protected candidate"):
        validate_output_directory(
            preset,
            frozen_candidate / "new-run",
            resume=None,
        )


def test_candidate_12_frozen_inputs_detect_v5_code_drift():
    preset = load_candidate_12()

    with pytest.raises(ValueError, match="frozen input hash mismatch"):
        verify_frozen_inputs(preset)


def test_v5_validation_defaults_match_parameter_lock():
    lock = json.loads(
        (SCRIPT_ROOT / "configs" / "v5-parameter-lock.json").read_text(encoding="utf-8")
    )
    args = parse_training_args([])

    assert (
        args.v5_validation_reset_probability
        == lock["v5_added_parameters"]["v5_validation_reset_probability"][
            "locked_value_for_all_ablations"
        ]
    )
    assert (
        args.v5_validation_hard_far_only_probability
        == lock["v5_added_parameters"]["v5_validation_hard_far_only_probability"][
            "locked_value_for_all_ablations"
        ]
    )
    assert (
        args.v5_validation_maximum_gap_hops
        == lock["v5_added_parameters"]["v5_validation_maximum_gap_hops"][
            "locked_value_for_all_ablations"
        ]
    )


def test_preset_rejects_runtime_owned_training_arguments(tmp_path):
    raw = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
    raw["training_arguments"].append(["--output-dir", "forbidden"])
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="controlled"):
        load_training_preset(invalid_path)
