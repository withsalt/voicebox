from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from dtln_aec_common import (
    BLOCK_LEN,
    BLOCK_SHIFT,
    DEFAULT_ONNX_DIR,
    DEFAULT_SAMPLE_MIC,
    DEFAULT_TFLITE_DIR,
    PART_IO,
    iter_model_parts,
    load_checked_model,
    onnx_path,
    parse_units,
    tflite_path,
    zeros_for_shape,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate exported DTLN-aec streaming ONNX models."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--tflite-dir", type=Path, default=DEFAULT_TFLITE_DIR)
    parser.add_argument(
        "--units",
        type=parse_units,
        default=list(parse_units("all")),
        help="Unit counts to validate: all, 128, 256, 512, or comma-separated values.",
    )
    parser.add_argument(
        "--sample-mic",
        type=Path,
        default=DEFAULT_SAMPLE_MIC,
        help="Sample *_mic.wav used for TFLite parity validation.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of streaming frames for TFLite/ONNX parity validation.",
    )
    parser.add_argument(
        "--benchmark-frames",
        type=int,
        default=100,
        help="Zero-input ONNX Runtime frames for a lightweight per-session benchmark.",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Only run ONNX checker, ONNX Runtime load, and zero-input inference.",
    )
    return parser.parse_args()


def ort_session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def validate_onnx_session(path: Path, benchmark_frames: int) -> None:
    model = load_checked_model(path)
    session = ort_session(path)
    expected_inputs = [value.name for value in model.graph.input]
    expected_outputs = [value.name for value in model.graph.output]
    actual_inputs = [value.name for value in session.get_inputs()]
    actual_outputs = [value.name for value in session.get_outputs()]
    if expected_inputs != actual_inputs:
        raise AssertionError(f"{path.name}: ORT inputs differ from ONNX graph.")
    if expected_outputs != actual_outputs:
        raise AssertionError(f"{path.name}: ORT outputs differ from ONNX graph.")

    feed = {value.name: zeros_for_shape(value.shape) for value in session.get_inputs()}
    session.run(None, feed)
    if benchmark_frames > 0:
        for _ in range(5):
            session.run(None, feed)
        started = time.perf_counter()
        for _ in range(benchmark_frames):
            session.run(None, feed)
        elapsed_ms = (time.perf_counter() - started) * 1000 / benchmark_frames
    else:
        elapsed_ms = 0.0

    print(
        f"{path.name}: checked, inputs={actual_inputs}, outputs={actual_outputs}, "
        f"zero-input avg={elapsed_ms:.3f} ms/frame"
    )


def run_tflite_parity(model_dir: Path, tflite_dir: Path, units: list[int], sample_mic: Path, frames: int) -> None:
    try:
        import soundfile as sf
        import tensorflow.lite as tflite
    except ImportError as exc:
        raise RuntimeError(
            "TFLite parity validation requires tensorflow and soundfile. "
            "Install requirements.txt or pass --skip-parity."
        ) from exc

    audio, fs = sf.read(sample_mic, dtype="float32")
    lpb_path = Path(str(sample_mic).replace("mic.wav", "lpb.wav"))
    lpb, fs_lpb = sf.read(lpb_path, dtype="float32")
    if fs != 16_000 or fs_lpb != 16_000:
        raise ValueError("DTLN-aec parity sample must be 16 kHz.")
    if audio.ndim != 1 or lpb.ndim != 1:
        raise ValueError("DTLN-aec parity sample must be mono.")

    length = min(len(audio), len(lpb))
    padding = np.zeros((BLOCK_LEN - BLOCK_SHIFT), dtype=np.float32)
    audio = np.concatenate((padding, audio[:length], padding)).astype(np.float32)
    lpb = np.concatenate((padding, lpb[:length], padding)).astype(np.float32)
    max_frames = (audio.shape[0] - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT
    frames = min(frames, max_frames)
    if frames <= 0:
        raise ValueError("No frames available for parity validation.")

    for unit in units:
        metrics = compare_streaming_unit(model_dir, tflite_dir, unit, audio, lpb, frames, tflite)
        print(
            f"dtln_aec_{unit}: frames={frames}, "
            f"max_mask={metrics['max_mask']:.6g}, "
            f"max_state1={metrics['max_state1']:.6g}, "
            f"max_block={metrics['max_block']:.6g}, "
            f"mean_block={metrics['mean_block']:.6g}, "
            f"max_state2={metrics['max_state2']:.6g}"
        )


def compare_streaming_unit(
    model_dir: Path,
    tflite_dir: Path,
    units: int,
    audio: np.ndarray,
    lpb: np.ndarray,
    frames: int,
    tflite_module,
) -> dict[str, float]:
    tflite_1 = tflite_module.Interpreter(model_path=str(tflite_path(tflite_dir, units, 1)))
    tflite_2 = tflite_module.Interpreter(model_path=str(tflite_path(tflite_dir, units, 2)))
    tflite_1.allocate_tensors()
    tflite_2.allocate_tensors()
    input_1 = tflite_1.get_input_details()
    output_1 = tflite_1.get_output_details()
    input_2 = tflite_2.get_input_details()
    output_2 = tflite_2.get_output_details()

    ort_1 = ort_session(onnx_path(model_dir, units, 1))
    ort_2 = ort_session(onnx_path(model_dir, units, 2))

    state1_t = np.zeros(input_1[1]["shape"], dtype=np.float32)
    state2_t = np.zeros(input_2[1]["shape"], dtype=np.float32)
    state1_o = zeros_for_shape(ort_1.get_inputs()[1].shape)
    state2_o = zeros_for_shape(ort_2.get_inputs()[1].shape)
    mic_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    lpb_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
    block_diffs: list[float] = []
    metrics = {
        "max_mask": 0.0,
        "max_state1": 0.0,
        "max_block": 0.0,
        "max_state2": 0.0,
    }

    for index in range(frames):
        start = index * BLOCK_SHIFT
        stop = start + BLOCK_SHIFT
        mic_buffer[:-BLOCK_SHIFT] = mic_buffer[BLOCK_SHIFT:]
        mic_buffer[-BLOCK_SHIFT:] = audio[start:stop]
        lpb_buffer[:-BLOCK_SHIFT] = lpb_buffer[BLOCK_SHIFT:]
        lpb_buffer[-BLOCK_SHIFT:] = lpb[start:stop]

        mic_fft = np.fft.rfft(mic_buffer).astype(np.complex64)
        mic_mag = np.abs(mic_fft).reshape(1, 1, -1).astype(np.float32)
        lpb_mag = np.abs(np.fft.rfft(lpb_buffer).astype(np.complex64)).reshape(1, 1, -1).astype(np.float32)

        tflite_1.set_tensor(input_1[0]["index"], mic_mag)
        tflite_1.set_tensor(input_1[2]["index"], lpb_mag)
        tflite_1.set_tensor(input_1[1]["index"], state1_t)
        tflite_1.invoke()
        mask_t = tflite_1.get_tensor(output_1[0]["index"])
        state1_t = tflite_1.get_tensor(output_1[1]["index"])

        mask_o, state1_o = ort_1.run(
            None,
            {
                PART_IO[1]["inputs"][0]: mic_mag,
                PART_IO[1]["inputs"][1]: state1_o,
                PART_IO[1]["inputs"][2]: lpb_mag,
            },
        )
        metrics["max_mask"] = max(metrics["max_mask"], max_abs(mask_t, mask_o))
        metrics["max_state1"] = max(metrics["max_state1"], max_abs(state1_t, state1_o))

        estimated_t = np.fft.irfft(mic_fft * mask_t).reshape(1, 1, -1).astype(np.float32)
        estimated_o = np.fft.irfft(mic_fft * mask_o).reshape(1, 1, -1).astype(np.float32)
        lpb_block = lpb_buffer.reshape(1, 1, -1).astype(np.float32)

        tflite_2.set_tensor(input_2[1]["index"], state2_t)
        tflite_2.set_tensor(input_2[0]["index"], estimated_t)
        tflite_2.set_tensor(input_2[2]["index"], lpb_block)
        tflite_2.invoke()
        block_t = tflite_2.get_tensor(output_2[0]["index"])
        state2_t = tflite_2.get_tensor(output_2[1]["index"])

        block_o, state2_o = ort_2.run(
            None,
            {
                PART_IO[2]["inputs"][0]: estimated_o,
                PART_IO[2]["inputs"][1]: state2_o,
                PART_IO[2]["inputs"][2]: lpb_block,
            },
        )
        diff = np.abs(block_t - block_o)
        metrics["max_block"] = max(metrics["max_block"], float(np.max(diff)))
        metrics["max_state2"] = max(metrics["max_state2"], max_abs(state2_t, state2_o))
        block_diffs.append(float(np.mean(diff)))

    metrics["mean_block"] = float(np.mean(block_diffs)) if block_diffs else 0.0
    return metrics


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    tflite_dir = args.tflite_dir.resolve()

    for units, part in iter_model_parts(args.units):
        validate_onnx_session(onnx_path(model_dir, units, part), args.benchmark_frames)

    if not args.skip_parity:
        run_tflite_parity(model_dir, tflite_dir, args.units, args.sample_mic.resolve(), args.frames)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
