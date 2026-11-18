"""Compare a GRE-Net ONNX model with another ONNX file or the upstream checkpoint.

The front end mirrors GRENetDNRService: periodic Hann analysis (400/200),
amplitude compression 0.3, one hop per 200 samples, overlap-add ISTFT with Hann
synthesis and window-square normalisation, and the 200-sample startup delay.
Both sides see identical spectra, so the metrics isolate model changes.

Use ``--torch-source GRE-Net-Streaming`` to run the ``g_00515000`` checkpoint
through the upstream ``StreamMPNet`` streaming forward, or ``--old model.onnx``
to compare two exports.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

import export as exporter

WINDOW = 400
HOP = 200
COMPRESS = 0.3
MAX_MAGNITUDE = 10.0
DELAY = HOP


class TorchReferenceModel:
    """Upstream PyTorch checkpoint (g_00515000) streaming one hop per call."""

    def __init__(self, source):
        import torch

        torch.set_num_threads(1)
        self.torch = torch
        self.stream = exporter.load_stream(source)
        self.state = list(self.stream.init_stream_cache(1, 201))

    def run(self, amplitude, phase):
        torch = self.torch
        amp = torch.from_numpy(amplitude).reshape(1, -1, 1)
        pha = torch.from_numpy(phase).reshape(1, -1, 1)
        with torch.no_grad():
            result = self.stream(amp, pha, *self.state)
        self.state = list(result[3:])
        return result[0].numpy().reshape(-1), result[1].numpy().reshape(-1)


class StreamingModel:
    def __init__(self, path, threads=1):
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = threads
        settings.inter_op_num_threads = 1
        settings.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), settings, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        self.names = [value.name for value in inputs]
        self.state = [np.zeros(value.shape, dtype=np.float32) for value in inputs[2:]]
        outputs = self.session.get_outputs()
        self.deltas = [i for i, value in enumerate(outputs[3:]) if value.name.endswith("_frame")]
        self.history = next(self.state[i].shape[-2] for i, name in enumerate(self.names[2:]) if name.startswith("ts_k_cache"))
        self.frame = 0

    def run(self, amplitude, phase):
        feed = dict(zip(self.names, (amplitude, phase, *self.state), strict=True))
        results = self.session.run(None, feed)
        next_state = results[3:]
        slot = self.frame % self.history
        for index in self.deltas:
            self.state[index][:, slot, :] = next_state[index]
            next_state[index] = self.state[index]
        self.state = next_state
        self.frame += 1
        return results[0].reshape(-1), results[1].reshape(-1)


def denoise(audio, model):
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WINDOW) / WINDOW)
    window_squared = window * window
    ring = np.zeros(WINDOW)
    ola = np.zeros(WINDOW)
    ola_norm = np.zeros(WINDOW)
    frames = audio.size // HOP
    output = []
    for index in range(frames):
        ring = np.concatenate((ring[HOP:], audio[index * HOP : (index + 1) * HOP]))
        spectrum = np.fft.rfft(ring * window)
        magnitude = np.sqrt(spectrum.real**2 + spectrum.imag**2 + 1e-9)
        amplitude = np.power(magnitude, COMPRESS).astype(np.float32)
        phase = np.arctan2(spectrum.imag + 1e-10, spectrum.real + 1e-5).astype(np.float32)
        out_amplitude, out_phase = model.run(amplitude[None, :, None], phase[None, :, None])
        out_magnitude = np.minimum(np.power(np.maximum(out_amplitude, 1e-9), 1.0 / COMPRESS), MAX_MAGNITUDE)
        enhanced = out_magnitude * np.exp(1j * out_phase)
        frame = np.fft.irfft(enhanced, n=WINDOW)
        ola = np.concatenate((ola[HOP:], np.zeros(HOP)))
        ola_norm = np.concatenate((ola_norm[HOP:], np.zeros(HOP)))
        ola += frame * window
        ola_norm += window_squared
        output.append(ola[:HOP] / np.maximum(ola_norm[:HOP], 1e-4))
    return np.concatenate(output)[DELAY:] if output else np.zeros(0)


def snr_db(reference, degraded):
    difference = degraded - reference
    energy = np.sum(reference.astype(np.float64) ** 2)
    noise = np.sum(difference.astype(np.float64) ** 2)
    return float(10 * np.log10(energy / max(noise, 1e-300)))


def segmental_snr(reference, degraded, size=320):
    count = reference.size // size
    values = []
    for index in range(count):
        left = reference[index * size : (index + 1) * size]
        right = degraded[index * size : (index + 1) * size]
        energy = np.sum(left.astype(np.float64) ** 2)
        noise = np.sum((right - left).astype(np.float64) ** 2)
        if energy > 1e-8:
            values.append(10 * np.log10(energy / max(noise, 1e-300)))
    return (float(np.median(values)) if values else float("inf"))


def optional_metrics(reference, degraded, rate):
    results = {}
    try:
        from pesq import pesq

        results["pesq"] = float(pesq(rate, reference.astype(np.float64), degraded.astype(np.float64), "wb"))
    except Exception as error:  # noqa: BLE001 - optional metric
        results["pesq"] = f"n/a ({error})"
    try:
        from pystoi import stoi

        results["stoi"] = float(stoi(reference, degraded, rate, extended=False))
    except Exception as error:  # noqa: BLE001 - optional metric
        results["stoi"] = f"n/a ({error})"
    return results


def main(old_factory, new_factory, audio_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(str(Path(audio_dir) / "*.wav")))
    if not paths:
        raise SystemExit(f"no wav files in {audio_dir}")
    for path in paths:
        audio, rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != 16000:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, 16000, rate).astype(np.float32)
            rate = 16000
        old = denoise(audio, old_factory())
        new = denoise(audio, new_factory())
        length = min(old.size, new.size, audio.size)
        old, new, trimmed = old[:length], new[:length], audio[:length]
        difference = new - old
        peak = float(np.max(np.abs(old))) if old.size else 0.0
        correlation = float(np.corrcoef(old, new)[0, 1]) if old.size else 0.0
        metrics = {
            "file": os.path.basename(path),
            "seconds": round(audio.size / rate, 2),
            "old_rms": round(float(np.sqrt(np.mean(old.astype(np.float64) ** 2))), 6),
            "new_rms": round(float(np.sqrt(np.mean(new.astype(np.float64) ** 2))), 6),
            "input_rms": round(float(np.sqrt(np.mean(trimmed.astype(np.float64) ** 2))), 6),
            "max_abs_diff": float(np.max(np.abs(difference))),
            "diff_rms": float(np.sqrt(np.mean(difference.astype(np.float64) ** 2))),
            "snr_db": round(snr_db(old, new), 2),
            "segmental_snr_db": round(segmental_snr(old, new), 2),
            "input_to_old_snr_db": round(snr_db(trimmed, old), 2),
            "input_to_new_snr_db": round(snr_db(trimmed, new), 2),
            "correlation": round(correlation, 8),
            "peak": round(peak, 4),
        }
        print(metrics)
        if output_dir:
            base = os.path.splitext(os.path.basename(path))[0]
            sf.write(Path(output_dir) / f"{base}_old.wav", old, rate)
            sf.write(Path(output_dir) / f"{base}_new.wav", new, rate)
            gain = 20.0 / max(float(np.max(np.abs(difference))), 1e-12)
            sf.write(Path(output_dir) / f"{base}_diff_x{gain:.0f}.wav", np.clip(difference * gain, -1, 1), rate)
        comparison = optional_metrics(trimmed, old, rate)
        print({f"old_vs_input_{key}": value for key, value in comparison.items()})
        comparison = optional_metrics(trimmed, new, rate)
        print({f"new_vs_input_{key}": value for key, value in comparison.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, help="reference ONNX model")
    parser.add_argument("--torch-source", type=Path, help="upstream GRE-Net-Streaming checkout (uses g_00515000)")
    parser.add_argument("--new", type=Path, required=True, help="candidate ONNX model")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_source is not None:
        source = args.torch_source.resolve()

        def old_factory():
            return TorchReferenceModel(source)
    elif args.old is not None:
        def old_factory():
            return StreamingModel(args.old, args.threads)
    else:
        parser.error("pass --old or --torch-source")

    def new_factory():
        return StreamingModel(args.new, args.threads)

    main(old_factory, new_factory, args.audio, args.output)
