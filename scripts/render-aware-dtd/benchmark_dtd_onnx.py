from __future__ import annotations

import argparse
import json
import math
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return parsed


def model_argument(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("model must have the form LABEL=PATH")
    return label, Path(raw_path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark streaming DTD ONNX models with a single CPU thread."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        type=model_argument,
        required=True,
        help="Repeatable LABEL=PATH model specification.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-frames", type=positive_int, default=250)
    parser.add_argument("--frames", type=positive_int, default=2_000)
    parser.add_argument("--trials", type=positive_int, default=5)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--hop-ms", type=positive_float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args(argv)


def static_shape(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} has no static shape")
    result: list[int] = []
    for dimension in value:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"{label} shape is not fully static: {value}")
        result.append(dimension)
    return tuple(result)


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.percentile(values, quantile))


def benchmark_model(
    path: Path,
    *,
    warmup_frames: int,
    frames: int,
    trials: int,
    threads: int,
    hop_ms: float,
    seed: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(
            f"unexpected execution providers: {session.get_providers()}"
        )
    inputs = {item.name: item for item in session.get_inputs()}
    expected_inputs = {"microphone", "render", "state_in"}
    if set(inputs) != expected_inputs:
        raise ValueError(
            f"unexpected model inputs: expected={expected_inputs}, "
            f"actual={set(inputs)}"
        )
    microphone_shape = static_shape(
        inputs["microphone"].shape,
        "microphone",
    )
    render_shape = static_shape(inputs["render"].shape, "render")
    state_shape = static_shape(inputs["state_in"].shape, "state_in")
    if microphone_shape != render_shape:
        raise ValueError("microphone and render shapes differ")

    rng = np.random.default_rng(seed)
    total_frames = warmup_frames + frames * trials
    microphone = rng.normal(
        0.0,
        0.05,
        (total_frames, *microphone_shape),
    ).astype(np.float32)
    render = rng.normal(
        0.0,
        0.05,
        (total_frames, *render_shape),
    ).astype(np.float32)

    state = np.zeros(state_shape, dtype=np.float32)
    for index in range(warmup_frames):
        _, state = session.run(
            None,
            {
                "microphone": microphone[index],
                "render": render[index],
                "state_in": state,
            },
        )

    trial_reports: list[dict[str, float]] = []
    offset = warmup_frames
    for _ in range(trials):
        state = np.zeros(state_shape, dtype=np.float32)
        elapsed_ms = np.empty(frames, dtype=np.float64)
        for frame in range(frames):
            started = time.perf_counter_ns()
            _, state = session.run(
                None,
                {
                    "microphone": microphone[offset + frame],
                    "render": render[offset + frame],
                    "state_in": state,
                },
            )
            elapsed_ms[frame] = (
                time.perf_counter_ns() - started
            ) / 1_000_000.0
        offset += frames
        mean_ms = float(elapsed_ms.mean())
        trial_reports.append(
            {
                "mean_ms": mean_ms,
                "p50_ms": percentile(elapsed_ms, 50),
                "p95_ms": percentile(elapsed_ms, 95),
                "p99_ms": percentile(elapsed_ms, 99),
                "max_ms": float(elapsed_ms.max()),
                "realtime_factor": mean_ms / hop_ms,
                "realtime_margin_x": hop_ms / mean_ms,
                "frames_per_second": 1_000.0 / mean_ms,
            }
        )

    representative = sorted(
        trial_reports,
        key=lambda report: report["mean_ms"],
    )[len(trial_reports) // 2]
    metadata = session.get_modelmeta().custom_metadata_map
    return {
        "path": str(path),
        "provider": "CPUExecutionProvider",
        "intra_op_threads": threads,
        "inter_op_threads": 1,
        "execution_mode": "sequential",
        "audio_shape": list(microphone_shape),
        "state_shape": list(state_shape),
        "warmup_frames": warmup_frames,
        "timed_frames_per_trial": frames,
        "trials": trials,
        "hop_ms": hop_ms,
        "seed": seed,
        "model_metadata": metadata,
        "trial_reports": trial_reports,
        "representative_median_trial": representative,
        "mean_ms_range": [
            min(report["mean_ms"] for report in trial_reports),
            max(report["mean_ms"] for report in trial_reports),
        ],
        "meets_realtime_by_mean": representative["mean_ms"] < hop_ms,
        "meets_realtime_by_p99": representative["p99_ms"] < hop_ms,
    }


def main() -> None:
    args = parse_args()
    labels = [label for label, _ in args.model]
    if len(set(labels)) != len(labels):
        raise ValueError("model labels must be unique")
    report = {
        "schema_version": 1,
        "method": (
            "recursive streaming ONNX Runtime CPU benchmark; session.run "
            "latency includes recurrent-state output allocation"
        ),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "onnxruntime_version": ort.__version__,
            "numpy_version": np.__version__,
        },
        "models": {
            label: benchmark_model(
                path,
                warmup_frames=args.warmup_frames,
                frames=args.frames,
                trials=args.trials,
                threads=args.threads,
                hop_ms=args.hop_ms,
                seed=args.seed,
            )
            for label, path in args.model
        },
    }
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="x",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
        temporary_path = Path(stream.name)
    temporary_path.replace(output_path)
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
