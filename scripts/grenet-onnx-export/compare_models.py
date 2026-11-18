"""Run two GRE-Net streaming ONNX models on identical input and compare outputs.

The harness tracks every cache exactly like the C# runner (attention rings and
frame deltas), optionally measures wall-clock per frame, and reports peak
amplitude, phase and complex errors and the complex-spectrum SNR.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def make_session(model, threads):
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = threads
    settings.inter_op_num_threads = 1
    settings.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model), settings, providers=["CPUExecutionProvider"])


class Harness:
    def __init__(self, model, threads=1):
        self.session = make_session(model, threads)
        inputs = self.session.get_inputs()
        self.names = [value.name for value in inputs]
        self.state = [np.zeros(value.shape, dtype=np.float32) for value in inputs[2:]]
        outputs = self.session.get_outputs()
        self.deltas = [(i, o.name) for i, o in enumerate(outputs[3:]) if o.name.endswith("_frame")]
        self.history = None
        for i, name in enumerate(self.names[2:]):
            if name.startswith("ts_k_cache"):
                self.history = self.state[i].shape[-2]
                break
        self.frame = 0

    def run(self, amp, phase):
        feed = dict(zip(self.names, (amp, phase, *self.state), strict=True))
        results = self.session.run(None, feed)
        next_state = results[3:]
        for index, _ in self.deltas:
            self.state[index][:, self.frame % self.history, :] = next_state[index]
            next_state[index] = self.state[index]
        self.state = next_state
        self.frame += 1
        return results[0], results[1], results[2]


def signal(samples):
    """Deterministic speech-like test signal."""

    generator = np.random.default_rng(20261118)
    t = np.arange(400 + 200 * (samples - 1)) / 16000
    audio = 0.12 * np.sin(2 * np.pi * (140 * t + 45 * t**2)) + 0.025 * generator.standard_normal(t.size)
    audio[:800] = 0
    return audio


def frames(audio):
    window = np.hanning(401)[:-1]
    for index in range((audio.size - 400) // 200 + 1):
        bins = np.fft.rfft(audio[200 * index : 200 * index + 400] * window)
        amp = np.power(np.abs(bins) ** 2 + 1e-9, 0.15).astype(np.float32)[None, :, None]
        phase = np.arctan2(bins.imag + 1e-10, bins.real + 1e-5).astype(np.float32)[None, :, None]
        yield amp, phase


def compare(reference, candidate, threads, count, measure, require_exact=False):
    left, right = Harness(reference, threads), Harness(candidate, threads)
    errors = {name: [] for name in ("amp", "phase", "complex")}
    times = {left: [], right: []}
    signal_energy = error_energy = 0.0
    for index, (amp, phase) in enumerate(frames(signal(count))):
        outputs = {}
        for harness in (left, right):
            start = time.perf_counter()
            outputs[harness] = harness.run(amp, phase)
            times[harness].append((time.perf_counter() - start) * 1000)
        ref_amp, ref_phase, ref_complex = outputs[left]
        new_amp, new_phase, new_complex = outputs[right]
        if require_exact:
            for name, old, new in zip(errors, outputs[left], outputs[right], strict=True):
                np.testing.assert_array_equal(new.view(np.uint32), old.view(np.uint32), err_msg=f"{name}, hop {index}")
        delta_amp = new_amp - ref_amp
        delta_phase = np.angle(np.exp(1j * (new_phase - ref_phase)))
        delta_complex = new_complex - ref_complex
        errors["amp"].append(np.abs(delta_amp).max())
        errors["phase"].append(np.abs(delta_phase).max())
        errors["complex"].append(np.abs(delta_complex).max())
        signal_energy += np.sum(ref_complex.astype(np.float64) ** 2)
        error_energy += np.sum(delta_complex.astype(np.float64) ** 2)
    print(f"frames {count}")
    for name, peaks in errors.items():
        peak = max(peaks)
        print(f"  {name:8s} max|delta| {peak:.3e}")
    snr = np.inf if error_energy == 0 else 10 * np.log10(signal_energy / error_energy)
    print(f"  complex SNR: {snr:.2f} dB")
    if measure:
        for harness, label in ((left, "reference"), (right, "candidate")):
            values = np.sort(times[harness])
            print(f"  {label:9s} mean {values.mean():.3f} ms  p99 {values[int(len(values) * 0.99) - 1]:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--require-exact", action="store_true", help="fail on the first bitwise difference in an audio output")
    args = parser.parse_args()
    compare(args.reference, args.candidate, args.threads, args.frames, args.measure, args.require_exact)
