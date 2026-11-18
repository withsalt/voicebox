"""Check recurrent ONNX outputs against independent PyTorch checkpoint inference."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export import load_stream


@torch.inference_mode()
def validate(source, model):
    torch.set_num_threads(1)
    stream = load_stream(source)
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model), settings, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    names = [value.name for value in inputs]
    ref_state = stream.init_stream_cache(1, 201)
    state = [np.zeros(value.shape, dtype=np.float32) for value in inputs[2:]]
    cursor_index = names[2:].index("ts_attn_cursor")
    history = state[names[2:].index("ts_k_cache")].shape[-2]
    generator = np.random.default_rng(20261118)
    frames = 256
    t = np.arange(400 + 200 * (frames - 1)) / 16000
    samples = 0.12 * np.sin(2 * np.pi * (140 * t + 45 * t**2)) + 0.025 * generator.standard_normal(t.size)
    samples[:800] = 0
    window = np.hanning(401)[:-1]
    peak_error = signal_energy = error_energy = 0.0
    for index in range(frames):
        bins = np.fft.rfft(samples[200 * index:200 * index + 400] * window)
        amp = np.power(np.abs(bins)**2 + 1e-9, 0.15).astype(np.float32)[None, :, None]
        phase = np.arctan2(bins.imag + 1e-10, bins.real + 1e-5).astype(np.float32)[None, :, None]
        expected = stream(torch.from_numpy(amp), torch.from_numpy(phase), *ref_state)
        actual = session.run(None, dict(zip(names, (amp, phase, *state), strict=True)))
        ref_state, state = expected[3:], actual[3:]
        # Compare phase on the unit circle so crossing +/-pi is continuous.
        phase_error = np.angle(np.exp(1j * (actual[1] - expected[1].numpy())))
        np.testing.assert_allclose(phase_error, 0, atol=2e-3)
        np.testing.assert_allclose(actual[0], expected[0].numpy(), atol=1e-4, rtol=1e-4)
        spectrum = expected[2].numpy()
        delta = actual[2] - spectrum
        np.testing.assert_allclose(actual[2], spectrum, atol=1e-4, rtol=1e-4)
        peak_error = max(peak_error, float(np.abs(delta).max()))
        signal_energy += np.sum(spectrum.astype(np.float64)**2)
        error_energy += np.sum(delta.astype(np.float64)**2)
        if index in (0, 1, 98, 99, 100, 199, frames - 1):
            cursor = (index + 1) % history
            np.testing.assert_array_equal(state[cursor_index], [cursor])
            for name, predicted, reference in zip(names[2:], state, ref_state, strict=True):
                if name == "ts_attn_cursor":
                    continue
                if name in ("ts_k_cache", "ts_v_cache"):
                    predicted = np.roll(predicted, -cursor, axis=-2)
                np.testing.assert_allclose(predicted, reference.numpy(), atol=5e-4, rtol=5e-4, err_msg=f"{name}, hop {index}")
    print(f"{frames} recurrent hops, complex spectrum max error {peak_error:.3g}, SNR {10*np.log10(signal_energy/error_energy):.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    validate(args.source, args.model)

