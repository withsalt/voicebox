from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PRESET_SCHEMA_VERSION = 1
RUNTIME_OWNED_OPTIONS = frozenset(
    {
        "--num-workers",
        "--output-dir",
        "--resume",
    }
)


@dataclass(frozen=True)
class TrainingPreset:
    path: Path
    name: str
    selection_policy_version: str
    trainer: Path
    initial_num_workers: int
    recovery_num_workers: int
    protected_output_directories: tuple[Path, ...]
    frozen_inputs: tuple[tuple[Path, str], ...]
    training_arguments: tuple[tuple[str, str], ...]


def _resolve_preset_path(preset_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if not path.is_absolute():
        path = preset_path.parent / path
    return path.resolve()


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_training_preset(path: str | Path) -> TrainingPreset:
    preset_path = Path(path).resolve()
    raw = json.loads(preset_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("training preset must contain a JSON object")
    if raw.get("schema_version") != PRESET_SCHEMA_VERSION:
        raise ValueError(
            "unsupported training preset schema_version: "
            f"{raw.get('schema_version')!r}"
        )

    raw_arguments = raw.get("training_arguments")
    if not isinstance(raw_arguments, list) or not raw_arguments:
        raise ValueError("training_arguments must be a non-empty array")
    training_arguments: list[tuple[str, str]] = []
    seen_options: set[str] = set()
    for index, pair in enumerate(raw_arguments):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(
                f"training_arguments[{index}] must contain option and value"
            )
        option = _require_string(pair[0], f"training_arguments[{index}][0]")
        value = _require_string(pair[1], f"training_arguments[{index}][1]")
        if not option.startswith("--"):
            raise ValueError(f"invalid training option: {option}")
        if option in seen_options:
            raise ValueError(f"duplicate training option: {option}")
        if option in RUNTIME_OWNED_OPTIONS:
            raise ValueError(
                f"{option} is controlled by the preset launcher at runtime"
            )
        seen_options.add(option)
        training_arguments.append((option, value))

    raw_frozen_inputs = raw.get("frozen_inputs")
    if not isinstance(raw_frozen_inputs, list) or not raw_frozen_inputs:
        raise ValueError("frozen_inputs must be a non-empty array")
    frozen_inputs: list[tuple[Path, str]] = []
    for index, entry in enumerate(raw_frozen_inputs):
        if not isinstance(entry, dict):
            raise ValueError(f"frozen_inputs[{index}] must be an object")
        configured_path = _require_string(
            entry.get("path"),
            f"frozen_inputs[{index}].path",
        )
        expected_hash = _require_string(
            entry.get("sha256"),
            f"frozen_inputs[{index}].sha256",
        ).upper()
        if len(expected_hash) != 64 or any(
            character not in "0123456789ABCDEF"
            for character in expected_hash
        ):
            raise ValueError(
                f"frozen_inputs[{index}].sha256 must be a SHA-256 digest"
            )
        frozen_inputs.append(
            (
                _resolve_preset_path(preset_path, configured_path),
                expected_hash,
            )
        )

    raw_protected = raw.get("protected_output_directories")
    if not isinstance(raw_protected, list) or not raw_protected:
        raise ValueError(
            "protected_output_directories must be a non-empty array"
        )
    protected = tuple(
        _resolve_preset_path(
            preset_path,
            _require_string(
                configured_path,
                f"protected_output_directories[{index}]",
            ),
        )
        for index, configured_path in enumerate(raw_protected)
    )

    return TrainingPreset(
        path=preset_path,
        name=_require_string(raw.get("name"), "name"),
        selection_policy_version=_require_string(
            raw.get("selection_policy_version"),
            "selection_policy_version",
        ),
        trainer=_resolve_preset_path(
            preset_path,
            _require_string(raw.get("trainer"), "trainer"),
        ),
        initial_num_workers=_require_positive_int(
            raw.get("initial_num_workers"),
            "initial_num_workers",
        ),
        recovery_num_workers=_require_positive_int(
            raw.get("recovery_num_workers"),
            "recovery_num_workers",
        ),
        protected_output_directories=protected,
        frozen_inputs=tuple(frozen_inputs),
        training_arguments=tuple(training_arguments),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_frozen_inputs(preset: TrainingPreset) -> dict[str, str]:
    verified: dict[str, str] = {}
    for path, expected_hash in preset.frozen_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"frozen input is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                "frozen input hash mismatch: "
                f"path={path}, expected={expected_hash}, actual={actual_hash}"
            )
        verified[str(path)] = actual_hash
    return verified


def _normalized_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _is_same_or_descendant(path: Path, directory: Path) -> bool:
    normalized_path = _normalized_path(path)
    normalized_directory = _normalized_path(directory)
    try:
        return (
            os.path.commonpath((normalized_path, normalized_directory))
            == normalized_directory
        )
    except ValueError:
        return False


def validate_output_directory(
    preset: TrainingPreset,
    output_dir: Path,
    resume: Path | None,
) -> tuple[Path, Path | None]:
    resolved_output = output_dir.resolve()
    if any(
        _is_same_or_descendant(resolved_output, protected)
        for protected in preset.protected_output_directories
    ):
        raise ValueError(
            f"refusing to write protected candidate directory: {resolved_output}"
        )

    if resume is None:
        if resolved_output.exists():
            raise FileExistsError(
                f"fresh output directory already exists: {resolved_output}"
            )
        return resolved_output, None

    resolved_resume = resume.resolve()
    if not resolved_output.is_dir():
        raise FileNotFoundError(
            f"resume output directory is missing: {resolved_output}"
        )
    immutable_checkpoints = resolved_output / "checkpoints"
    can_recover_missing_last = (
        resolved_resume == resolved_output / "last.pt"
        and immutable_checkpoints.is_dir()
        and any(path.is_file() for path in immutable_checkpoints.glob("epoch-*.pt"))
    )
    if not resolved_resume.is_file() and not can_recover_missing_last:
        raise FileNotFoundError(f"resume checkpoint is missing: {resolved_resume}")
    try:
        resolved_resume.relative_to(resolved_output)
    except ValueError as error:
        raise ValueError(
            "resume checkpoint must be inside its output directory: "
            f"checkpoint={resolved_resume}, output={resolved_output}"
        ) from error
    return resolved_output, resolved_resume


def build_training_command(
    preset: TrainingPreset,
    output_dir: Path,
    resume: Path | None = None,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    resolved_output, resolved_resume = validate_output_directory(
        preset,
        output_dir,
        resume,
    )
    workers = (
        preset.recovery_num_workers
        if resolved_resume is not None
        else preset.initial_num_workers
    )
    command = [
        str(python_executable),
        str(preset.trainer),
    ]
    for option, value in preset.training_arguments:
        command.extend((option, value))
    command.extend(
        (
            "--output-dir",
            str(resolved_output),
            "--num-workers",
            str(workers),
        )
    )
    if resolved_resume is not None:
        command.extend(("--resume", str(resolved_resume)))
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run a frozen render-aware DTD training preset."
    )
    parser.add_argument(
        "--preset",
        default=str(
            script_root / "configs" / "default-candidate-16.json"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preset = load_training_preset(args.preset)
    verified_inputs = verify_frozen_inputs(preset)
    command = build_training_command(
        preset,
        Path(args.output_dir),
        Path(args.resume) if args.resume else None,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "preset": preset.name,
                    "selection_policy_version": (
                        preset.selection_policy_version
                    ),
                    "verified_inputs": verified_inputs,
                    "command": command,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
