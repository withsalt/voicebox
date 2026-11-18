"""Check recurrent ONNX outputs against independent PyTorch checkpoint inference."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from export import load_stream


def projected_history(layer, cache):
    # Reconstruct the past tap from the upstream raw activation delay line.
    x = cache[0].transpose(0, 1).unsqueeze(2)
    r, i, a = x.split((layer.ang_in_chn, layer.ang_in_chn, layer.amp_in_chn), dim=1)

    def past(x, conv):
        return F.conv2d(x, conv.weight[:, :, :1], padding=(0, 1))

    cr, ci = layer.conv_ang.conv_re, layer.conv_ang.conv_im
    return torch.cat((past(r, cr) - past(i, ci), past(r, ci) + past(i, cr), past(a, layer.conv_amp)), dim=1)


def assert_caches(names, state, ref_state, cursor, history, split_rings, index):
    compared = dict(zip(names, state, strict=True))
    for bank_name, ref in zip(("ts_k_cache", "ts_v_cache"), ref_state[8:10], strict=True):
        if split_rings:
            layers = [compared.pop(f"{bank_name}_{i + 1}") for i in range(ref.shape[0])]
            bank = np.stack(layers).reshape(*ref.shape[:-2], history, ref.shape[-1])
        else:
            bank = compared.pop(bank_name)
        chronological = np.roll(bank, -cursor, axis=-2)[..., -ref.shape[-2]:, :]
        np.testing.assert_allclose(chronological, ref.numpy(), atol=5e-4, rtol=5e-4, err_msg=f"{bank_name}, hop {index}")
    del compared["ts_attn_cursor"]
    other_ref = (*ref_state[:8], *ref_state[11:])
    for (name, predicted), reference in zip(compared.items(), other_ref, strict=True):
        np.testing.assert_allclose(predicted, reference.numpy(), atol=5e-4, rtol=5e-4, err_msg=f"{name}, hop {index}")


def assert_ffn_histories(state, ref_state, index):
    for stage, real, imag in (("in", *ref_state[13:15]), ("mid", *ref_state[15:17])):
        for layer, (layer_real, layer_imag) in enumerate(zip(real, imag, strict=True), start=1):
            name = f"ts_ang_{stage}_history_{layer}"
            # Upstream retains the complete four-tap window. Only its newest three taps enter the next hop.
            expected = torch.cat((layer_real[0, ..., 1:], layer_imag[0, ..., 1:]), dim=1).permute(2, 0, 1)
            np.testing.assert_allclose(state[name], expected.numpy(), atol=5e-4, rtol=5e-4, err_msg=f"{name}, hop {index}")


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
    deltas = [i for i, value in enumerate(session.get_outputs()[3:]) if value.name == names[i + 2] + "_frame"]
    dense = [layer for block in (stream.mpnet.dense_encoder, stream.mpnet.decoder) for _, layer in block.dense_block.dense_block]
    cursor_index = names[2:].index("ts_attn_cursor")
    # Incremental rings contain 99 past frames. Older per-layer full-state rings also hold the current slot.
    split_rings = "ts_k_cache_1" in names
    # The node-minimal export merges paired norm caches, so their per-hop state
    # is no longer directly comparable to the reference module; outputs are.
    merged_norms = "enc_norm_power_cache" in names
    ring_name = "ts_k_cache_1" if split_rings else "ts_k_cache"
    history = state[names[2:].index(ring_name)].shape[-2]
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
        next_state = actual[3:]
        for i in deltas:
            state[i][:, index % history, :] = next_state[i]
            next_state[i] = state[i]
        ref_state, state = expected[3:], next_state
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
        if index in (0, 1, 2, 3, 98, 99, 100, 198, 199, 200, frames - 1):
            cursor = (index + 1) % history
            np.testing.assert_array_equal(state[cursor_index], [cursor])
            if "ts_ang_in_history_1" in names:
                assert_ffn_histories(dict(zip(names[2:], state, strict=True)), ref_state, index)
            elif not merged_norms:
                ref_caches = list(ref_state)
                if deltas:
                    ref_caches[:8] = [projected_history(layer, ref) for layer, ref in zip(dense, ref_state[:8], strict=True)]
                assert_caches(names[2:], state, ref_caches, cursor, history, split_rings, index)
    print(f"{frames} recurrent hops, complex spectrum max error {peak_error:.3g}, SNR {10*np.log10(signal_energy/error_energy):.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    validate(args.source, args.model)

