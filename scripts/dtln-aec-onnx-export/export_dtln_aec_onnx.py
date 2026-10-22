from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx

from dtln_aec_common import (
    DEFAULT_ONNX_DIR,
    UPSTREAM_DIR,
    PART_IO,
    add_metadata,
    iter_model_parts,
    onnx_path,
    parse_units,
    rename_onnx_io,
    tflite_path,
)

DEFAULT_UPSTREAM_URL = "https://github.com/breizhn/DTLN-aec.git"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DTLN-aec streaming TFLite models to ONNX."
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=UPSTREAM_DIR,
        help="Directory for the upstream DTLN-aec repository.",
    )
    parser.add_argument(
        "--upstream-url",
        default=DEFAULT_UPSTREAM_URL,
        help="Git URL used when the upstream DTLN-aec directory is missing.",
    )
    parser.add_argument(
        "--clone-timeout-sec",
        type=int,
        default=300,
        help="Timeout for automatic git clone of the upstream DTLN-aec repository.",
    )
    parser.add_argument(
        "--no-clone",
        action="store_true",
        help="Fail instead of cloning the upstream DTLN-aec repository when it is missing.",
    )
    parser.add_argument(
        "--tflite-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing dtln_aec_{units}_{part}.tflite files. "
            "Defaults to upstream-dir/pretrained_models."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ONNX_DIR,
        help="Directory for renamed ONNX models.",
    )
    parser.add_argument(
        "--units",
        type=parse_units,
        default=list(parse_units("all")),
        help="Unit counts to export: all, 128, 256, 512, or comma-separated values.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset passed to tf2onnx.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep unrenamed tf2onnx outputs under output-dir/raw.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ONNX files.",
    )
    return parser.parse_args()


def ensure_upstream_repo(upstream_dir: Path, upstream_url: str, no_clone: bool, timeout_sec: int) -> None:
    if upstream_dir.exists():
        return
    if no_clone:
        raise FileNotFoundError(
            f"Missing upstream repository: {upstream_dir}. "
            "Run without --no-clone or clone it manually."
        )

    upstream_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"clone upstream DTLN-aec: {upstream_url} -> {upstream_dir}")
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", upstream_url, str(upstream_dir)],
            check=True,
            env=env,
            timeout=max(1, timeout_sec),
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(upstream_dir, ignore_errors=True)
        raise TimeoutError(
            f"Timed out cloning upstream DTLN-aec after {timeout_sec}s: {upstream_url}"
        ) from exc
    except subprocess.CalledProcessError:
        shutil.rmtree(upstream_dir, ignore_errors=True)
        raise


def validate_tflite_models(tflite_dir: Path, units_to_export: list[int]) -> None:
    missing = [
        tflite_path(tflite_dir, units, part)
        for units, part in iter_model_parts(units_to_export)
        if not tflite_path(tflite_dir, units, part).exists()
    ]
    if not missing:
        return

    examples = "\n".join(f"  - {path}" for path in missing[:6])
    raise FileNotFoundError(
        "The upstream DTLN-aec directory exists, but pretrained TFLite models "
        f"are missing under {tflite_dir}:\n{examples}"
    )


def convert_with_tf2onnx(tflite_model: Path, raw_onnx: Path, opset: int) -> None:
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    command = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--tflite",
        str(tflite_model),
        "--output",
        str(raw_onnx),
        "--opset",
        str(opset),
    ]
    subprocess.run(command, check=True, env=env)


def postprocess_onnx(raw_onnx: Path, final_onnx: Path, units: int, part: int, opset: int) -> None:
    model = onnx.load(raw_onnx)
    io = PART_IO[part]
    rename_onnx_io(model, io["inputs"], io["outputs"])
    add_metadata(model, units, part, opset)
    onnx.checker.check_model(model)
    onnx.save(model, final_onnx)


def export_one(
    tflite_dir: Path,
    output_dir: Path,
    raw_dir: Path,
    units: int,
    part: int,
    opset: int,
    force: bool,
) -> None:
    source = tflite_path(tflite_dir, units, part)
    target = onnx_path(output_dir, units, part)
    raw_target = onnx_path(raw_dir, units, part)

    if not source.exists():
        raise FileNotFoundError(f"Missing TFLite model: {source}")
    if target.exists() and not force:
        print(f"skip existing {target}")
        return

    raw_target.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"export dtln_aec_{units}_{part}: {source.name} -> {target.name}")
    convert_with_tf2onnx(source, raw_target, opset)
    postprocess_onnx(raw_target, target, units, part, opset)


def main() -> int:
    args = parse_args()
    upstream_dir = args.upstream_dir.resolve()
    ensure_upstream_repo(upstream_dir, args.upstream_url, args.no_clone, args.clone_timeout_sec)

    tflite_dir = (args.tflite_dir or (upstream_dir / "pretrained_models")).resolve()
    validate_tflite_models(tflite_dir, args.units)
    output_dir = args.output_dir.resolve()

    if args.keep_raw:
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="dtln-aec-onnx-")
        raw_dir = Path(temp_dir.name)

    try:
        for units, part in iter_model_parts(args.units):
            export_one(tflite_dir, output_dir, raw_dir, units, part, args.opset, args.force)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    if args.keep_raw:
        print(f"raw ONNX files kept under {raw_dir}")
    else:
        raw_copy = output_dir / "raw"
        if raw_copy.exists():
            shutil.rmtree(raw_copy)

    print(f"done: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
