"""Export the four-block causal GRE-Net checkpoint with functional streaming caches."""

import argparse
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper
from onnxsim import simplify
from torch import nn


MODEL_FILENAME = "stream.onnx"


# State order is shared by StaticStream.forward, init_stream_cache and the VoiceBox runners.
CACHE_NAMES = (
    *(f"enc_cache_{i}" for i in range(1, 5)),
    *(f"dec_cache_{i}" for i in range(1, 5)),
    "ts_k_cache", "ts_v_cache", "ts_attn_cursor", "ts_attn_count", "ts_gru_cache",
    "ts_ang_in_cache_r", "ts_ang_in_cache_i", "ts_ang_mid_cache_r", "ts_ang_mid_cache_i",
    "enc_amp_power_cache", "enc_ang_power_cache", "enc_amp_count_cache", "enc_ang_count_cache",
    "dec_amp_power_cache", "dec_ang_power_cache", "dec_amp_count_cache", "dec_ang_count_cache",
    "input_amp_power_cache", "input_amp_count_cache",
)


def shift_history(history, sample):
    return torch.cat((history[:, :, 1:], sample), dim=2)


def phase_angle(y, x):
    # Range reduction bounds |z| by tan(pi/8). The first omitted Taylor term
    # is below 3e-9 radians, smaller than float32 rounding near pi/4.
    ax, ay = x.abs(), y.abs()
    high = torch.maximum(ax, ay)
    ratio = torch.minimum(ax, ay) / torch.where(high > 0, high, 1.0)
    reduce = ratio > math.tan(math.pi / 8)
    z = torch.where(reduce, (ratio - 1) / (ratio + 1), ratio)
    zz = z.square()
    polynomial = 1.0 / 17
    for degree in range(15, 0, -2):
        polynomial = 1.0 / degree - zz * polynomial
    angle = z * polynomial + torch.where(reduce, math.pi / 4, 0.0)
    angle = torch.where(ay > ax, math.pi / 2 - angle, angle)
    angle = torch.where(x < 0, math.pi - angle, angle)
    return torch.where(y < 0, -angle, angle)


def pack_norms(rows):
    # Each upstream block returns amp power, amp count, angle power, angle count.
    amp_power, amp_count, ang_power, ang_count = map(torch.stack, zip(*rows))
    return amp_power, ang_power, amp_count, ang_count


def apply_block(block, x, norms, index):
    amp_power, ang_power, amp_count, ang_count = norms
    x, *updated = block.forward_stream(x, amp_power[index], amp_count[index], ang_power[index], ang_count[index])
    return x, updated


def dense_layers(dense, x, caches, norms, offset):
    next_caches, next_norms = [], []
    skip = x
    for i, (_, layer) in enumerate(dense.dense_block):
        # The two temporal kernel taps read only the oldest and the newest frame.
        # Their original dilation is replaced with 1 on this two-frame input.
        context = torch.cat((caches[i][:, :, :1], skip), dim=2)
        next_caches.append(shift_history(caches[i], skip))
        x, norm = apply_block(layer, F.pad(context, (1, 1)), norms, offset + i)
        next_norms.append(norm)
        if i + 1 < len(dense.dense_block):
            channels = (dense.ang_in_chn, dense.ang_in_chn, dense.amp_in_chn)
            new = x.split(channels, dim=1)
            old = skip.split([c * (i + 1) for c in channels], dim=1)
            skip = torch.cat([tensor for pair in zip(new, old) for tensor in pair], dim=1)
    return x, next_caches, next_norms


def stream_complex_ffn(ffn, real, imag, in_r, in_i, mid_r, mid_i):
    def append(history, current):
        return torch.cat((history[..., 1:], current), dim=-1)

    in_r = append(in_r, real.transpose(1, 2))
    in_i = append(in_i, imag.transpose(1, 2))

    def single_position_conv(conv, r, i):
        joined = torch.cat((r, i), dim=1).flatten(1)
        result = F.linear(joined, conv.fused_conv.weight.flatten(1))
        return result.unsqueeze(-1).chunk(2, dim=1)

    real, imag = single_position_conv(ffn.conv1d, in_r, in_i)
    real, gate_r = real.chunk(2, dim=1)
    imag, gate_i = imag.chunk(2, dim=1)
    magnitude = torch.sqrt(gate_r.square() + gate_i.square() + 1e-9)
    gate = ffn.gate_act(ffn.gate_ln(magnitude.transpose(1, 2)).transpose(1, 2))
    mid_r = append(mid_r, real * gate)
    mid_i = append(mid_i, imag * gate)
    real, imag = single_position_conv(ffn.causal_conv_out, mid_r, mid_i)
    return real.transpose(1, 2), imag.transpose(1, 2), in_r, in_i, mid_r, mid_i


def time_step(block, x, caches):
    keys, values, count, hidden, *ffn_history = caches
    att = block.att
    real, imag, amp = x.split((block.ang_chn, block.ang_chn, block.amp_chn), dim=-1)
    nr, ni = block.cnorm1(real, imag)
    na = block.norm1(amp)

    def project(linear_ang, linear_amp):
        r, i = linear_ang(nr, ni)
        parts = [p.reshape(x.size(0), att.num_heads, 1, -1) for p in (r, i, linear_amp(na))]
        return torch.cat(parts, dim=-1)

    q = project(att.to_q_ang, att.to_q_amp)
    k = project(att.to_k_ang, att.to_k_amp)
    v = project(att.to_v_ang, att.to_v_amp)
    history = keys.size(2)
    query = q.transpose(-1, -2)
    scores = torch.cat((keys @ query, k @ query), dim=-2).transpose(-1, -2) * block.attn_scale
    # During startup the ring fills from slot 0. Once full, attention is
    # invariant to the physical order of the paired key and value slots.
    valid = torch.arange(history, dtype=count.dtype) < count
    valid = torch.cat((valid, torch.ones_like(count, dtype=torch.bool)), dim=-1)
    weights = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min).softmax(dim=-1)
    attended = weights[..., :history] @ values + weights[..., history:] * v
    parts = attended.split((att.ang_v_head_dim, att.ang_v_head_dim, att.amp_v_head_dim), dim=-1)
    r, i, a = [part.reshape(x.size(0), 1, -1) for part in parts]
    r, i = att.to_out_ang(r, i)
    real, imag, amp = real + r, imag + i, amp + att.to_out_amp(a)
    a, hidden = block.amp_ffn.forward_stream(block.norm2(amp), hidden)
    amp = block.norm3(amp + a)
    nr, ni = block.cnorm2(real, imag)
    r, i, *ffn_history = stream_complex_ffn(block.ang_ffn, nr, ni, *ffn_history)
    real, imag = block.cnorm3(real + r, imag + i)
    state = (k, v, (count + 1).clamp(max=history), hidden, *ffn_history)
    return torch.cat((real, imag, amp), dim=-1), state


def write_attention_frame(cache, frame, cursor):
    # Each row of the flattened tensor holds one key or value vector.
    history, channels = cache.shape[-2:]
    frame = frame.reshape(-1, channels)
    rows = torch.arange(frame.size(0), device=cache.device) * history + cursor.to(torch.long)
    return cache.reshape(-1, channels).index_put((rows,), frame).reshape_as(cache)


def complex_weights(real, imag):
    # (Wr + i Wi)(xr + i xi) as a real block matrix.
    return torch.cat((torch.cat((real, -imag), dim=1), torch.cat((imag, real), dim=1)), dim=0)


class FusedComplexLinear(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.register_buffer("weight", complex_weights(projection.weight_r, projection.weight_i).detach())
        bias = torch.cat((projection.bias_r, projection.bias_i)).detach() if projection.use_bias else None
        self.register_buffer("bias", bias)

    def forward(self, real, imag):
        return F.linear(torch.cat((real, imag), dim=-1), self.weight, self.bias).chunk(2, dim=-1)


class FusedComplexConv(nn.Module):
    def __init__(self, conv):
        super().__init__()
        real, imag = conv.conv_re, conv.conv_im
        self.transpose = isinstance(real, nn.ConvTranspose2d)
        weight = (
            complex_weights(real.weight.transpose(0, 1), imag.weight.transpose(0, 1)).transpose(0, 1).contiguous()
            if self.transpose else complex_weights(real.weight, imag.weight)
        )
        self.register_buffer("weight", weight.detach())
        self.stride, self.padding, self.dilation = real.stride, real.padding, real.dilation
        self.output_padding = real.output_padding

    def forward(self, real, imag):
        x = torch.cat((real, imag), dim=1)
        if self.transpose:
            x = F.conv_transpose2d(
                x.unsqueeze(-2), self.weight, stride=self.stride, padding=self.padding,
                output_padding=self.output_padding, dilation=self.dilation,
            ).squeeze(-2)
        else:
            x = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
        return x.chunk(2, dim=1)


def fuse_complex_layers(module):
    from models.model import ComplexConv
    from models.transformer import ComplexConvTranspose1d, ComplexLinear

    for name, child in module.named_children():
        if isinstance(child, ComplexLinear):
            module.add_module(name, FusedComplexLinear(child))
        elif isinstance(child, (ComplexConv, ComplexConvTranspose1d)):
            module.add_module(name, FusedComplexConv(child))
        else:
            fuse_complex_layers(child)


class StaticStream(nn.Module):
    def __init__(self, stream):
        super().__init__()
        self.net = stream.mpnet
        stream.fuse_inference_kernels()
        for dense in (self.net.dense_encoder.dense_block, self.net.decoder.dense_block):
            for _, layer in dense.dense_block:
                for conv in (layer.conv_ang.conv_re, layer.conv_ang.conv_im, layer.conv_amp):
                    conv.dilation = (1, 1)
        fuse_complex_layers(self.net)

    def forward(self, amp, phase, *state):
        enc = self.net.dense_encoder
        dec = self.net.decoder
        ts = (*state[8:10], *state[11:17])
        cursor = state[10]
        enc_norm, dec_norm = state[17:21], state[21:25]
        amplitude, power, count, gain = self.net.input_rms.forward_stream(amp[:, None].transpose(2, 3), *state[25:], return_gain=True)
        x = torch.cat((phase.cos().transpose(1, 2)[:, None], phase.sin().transpose(1, 2)[:, None], amplitude), dim=1)
        x, first = apply_block(enc.dense_conv_1, x, enc_norm, 0)
        x, enc_cache, middle = dense_layers(enc.dense_block, x, state[:4], enc_norm, 1)
        x, last = apply_block(enc.dense_conv_3, x, enc_norm, 5)
        next_enc_norm = pack_norms([first, *middle, last])

        next_ts = []
        layer_caches = zip(*(cache.unbind(0) for cache in ts))
        for block, layer_state in zip(self.net.TSTransformer, layer_caches):
            # Fixed B=1. The GRU cache carries a separate leading layer axis.
            inputs = [cache.squeeze(1 if j == 3 else 0) for j, cache in enumerate(layer_state)]
            x = x[0].permute(2, 1, 0)
            y, updated = time_step(block.time_transformer, x, inputs)
            next_ts.append([v.unsqueeze(1 if j == 3 else 0) for j, v in enumerate(updated)])
            x = (x + y).transpose(0, 1)
            x = (x + block.freq_transformer(x)).permute(2, 0, 1)[None]

        x, dec_cache, middle = dense_layers(dec.dense_block, x, state[4:8], dec_norm, 0)
        x, last = apply_block(dec.upsample_layer, x, dec_norm, 4)
        next_dec_norm = pack_norms([*middle, last])
        real, imag, amp = x.split((dec.ang_chn, dec.ang_chn, dec.amp_chn), dim=1)
        real, imag = dec.ang_conv(real, imag)
        phase = phase_angle(imag + 1e-9, real + 1e-9).squeeze(1).transpose(1, 2)
        amp = dec.amp_conv_layer(amp).squeeze(1).transpose(1, 2) / gain.squeeze(-1)
        spectrum = torch.stack((amp * phase.cos(), amp * phase.sin()), dim=-1)
        keys, values, *other = map(torch.stack, zip(*next_ts))
        keys = write_attention_frame(ts[0], keys, cursor)
        values = write_attention_frame(ts[1], values, cursor)
        cursor = torch.remainder(cursor + 1, keys.shape[-2])
        return (
            amp, phase, spectrum, *enc_cache, *dec_cache, keys, values, cursor,
            *other, *next_enc_norm, *next_dec_norm, power, count,
        )


def load_stream(source):
    sys.path.insert(0, str(source.resolve()))
    from models import model

    directory = source / "cp_model_casual_whamr"
    cfg = SimpleNamespace(**json.loads((directory / "config.json").read_text()))
    stream = model.StreamMPNet(cfg).eval()
    weights = torch.load(directory / "g_00515000", map_location="cpu", weights_only=True)["generator"]
    target = stream.mpnet.state_dict()
    for name, tensor in weights.items():
        if name in target and "ang_ffn.deconv1d" in name and tensor.ndim + 1 == target[name].ndim:
            weights[name] = tensor.unsqueeze(-2)
    stream.mpnet.load_state_dict(weights, strict=True)
    return stream


def lower_square_ops(graph):
    squares = {
        node.output[0]
        for node in graph.node
        if node.op_type == "Constant" and not node.domain
        for attr in node.attribute
        if attr.name == "value" and not attr.t.dims and numpy_helper.to_array(attr.t).item() == 2
    }
    for node in graph.node:
        if node.op_type == "Pow" and not node.domain and node.input[1] in squares:
            node.op_type = "Mul"
            node.input[1] = node.input[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="GRE-Net-Streaming checkout")
    parser.add_argument("--output", type=Path, default=Path(MODEL_FILENAME), help="ONNX output path (default: stream.onnx)")
    args = parser.parse_args()
    stream = load_stream(args.source)
    from models import model

    names_in = ["noisy_amp_t", "noisy_pha_t", *CACHE_NAMES]
    names_out = ["denoised_amp_t", "denoised_pha_t", "denoised_com_t", *(f"{name}_out" for name in CACHE_NAMES)]
    model._shift_append_cache_dim2 = shift_history
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cache = list(stream.init_stream_cache(1, 201))
    cache[10] = torch.zeros(1, dtype=torch.float32)
    wrapper = StaticStream(stream).eval()
    with TemporaryDirectory() as temp:
        raw = Path(temp) / MODEL_FILENAME
        with torch.no_grad():
            torch.onnx.export(wrapper, (torch.rand(1, 201, 1), torch.randn(1, 201, 1), *cache), raw,
                              input_names=names_in, output_names=names_out, opset_version=17, dynamo=False)
        exported = onnx.load(raw)
    lower_square_ops(exported.graph)
    simplified, valid = simplify(exported)
    if not valid:
        raise RuntimeError("ONNX simplification failed")
    onnx.checker.check_model(simplified)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(simplified, args.output)


if __name__ == "__main__":
    main()

