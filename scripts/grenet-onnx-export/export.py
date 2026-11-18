"""Export the static FP32 GRE-Net streaming graph.

The export uses projected convolution histories, external attention rings and
packed normalization caches. Each time FFN keeps three contiguous tap planes
per convolution, with real and imaginary channels together. Transformer blocks
keep features contiguous across the time and frequency passes, expose RMSNorm
patterns for ONNX Runtime fusion and lower the frequency complex FFN to two plain convolutions.
"""

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

from optimize_attention import fuse_frequency_attention


MODEL_FILENAME = "stream.onnx"
ENC_NORM_COUNT = 6
DEC_NORM_COUNT = 5
ANG_EPS = 1e-7  # hard-coded by CFWiseComplexRMSNorm.forward_stream

# State order is shared by StaticStream.forward and the VoiceBox runners.
CACHE_NAMES = (
    *(f"enc_cache_{i}" for i in range(1, 5)),
    *(f"dec_cache_{i}" for i in range(1, 5)),
    *(f"ts_k_cache_{i}" for i in range(1, 5)),
    *(f"ts_v_cache_{i}" for i in range(1, 5)),
    "ts_attn_cursor", "ts_attn_count", "ts_gru_cache",
    *(f"ts_ang_in_history_{i}" for i in range(1, 5)),
    *(f"ts_ang_mid_history_{i}" for i in range(1, 5)),
    "enc_norm_power_cache", "enc_norm_count_cache",
    "dec_norm_power_cache", "dec_norm_count_cache",
    "input_amp_power_cache", "input_amp_count_cache",
)


def shift_history(history, sample):
    return torch.cat((history[:, :, 1:], sample), dim=2)


def phase_angle(y, x):
    # Range reduction bounds |z| by tan(pi/8). The first omitted Taylor term
    # is below 3e-9 radians, smaller than float32 rounding near pi/4.
    # (ONNX Runtime's CUDA EP has no Atan kernel, so the polynomial stays.)
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


def complex_weights(real, imag):
    # (Wr + i Wi)(xr + i xi) as a real block matrix.
    return torch.cat((torch.cat((real, -imag), dim=1), torch.cat((imag, real), dim=1)), dim=0)


class FusedComplexConv(nn.Module):
    def __init__(self, conv):
        super().__init__()
        real, imag = conv.conv_re, conv.conv_im
        self.register_buffer("weight", complex_weights(real.weight, imag.weight).detach())
        self.stride, self.padding, self.dilation = real.stride, real.padding, real.dilation

    def forward(self, real, imag):
        x = torch.cat((real, imag), dim=1)
        return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation).chunk(2, dim=1)


def fuse_complex_convs(module):
    from models.model import ComplexConv

    for name, child in module.named_children():
        if isinstance(child, ComplexConv):
            module.add_module(name, FusedComplexConv(child))
        else:
            fuse_complex_convs(child)


class PackedAttention(nn.Module):
    """Lay out the checkpoint's projections in head order once at export time."""

    def __init__(self, att):
        super().__init__()
        self.num_heads = att.num_heads

        def pack(ang, amp, output=False):
            weight = torch.block_diag(complex_weights(ang.weight_r, ang.weight_i), amp.weight)
            # Permute rows for Q/K/V and columns for the output projection.
            if output:
                weight = weight.T
            axis = 1 if output else 0
            sizes = (ang.weight_r.size(axis), ang.weight_i.size(axis), amp.weight.size(axis))
            parts = [w.reshape(att.num_heads, -1, weight.size(1)) for w in weight.split(sizes)]
            packed = torch.cat(parts, dim=1).reshape_as(weight)
            return (packed.T if output else packed).contiguous().detach()

        qkv = [pack(getattr(att, f"to_{part}_ang"), getattr(att, f"to_{part}_amp")) for part in ("q", "k", "v")]
        packed = torch.cat(qkv)
        self.projection_size = qkv[0].size(0)
        scale = (self.projection_size // self.num_heads) ** -0.5
        # Fold the attention scale into the query rows so the graph needs no Mul.
        packed[: self.projection_size] *= scale
        self.register_buffer("qkv", packed)
        self.register_buffer("out", pack(att.to_out_ang, att.to_out_amp, output=True))
        bias = att.to_out_amp.bias.detach()
        self.register_buffer("out_bias", F.pad(bias, (2 * att.to_out_ang.out_features, 0)))

    def project(self, x):
        batch, length, _ = x.shape
        projected = F.linear(x, self.qkv).split(self.projection_size, dim=-1)
        if length == 1:
            # Swapping a unit axis preserves linear order, so reshape directly.
            return [part.reshape(batch, self.num_heads, length, -1) for part in projected]
        return [part.reshape(batch, length, self.num_heads, -1).transpose(1, 2) for part in projected]

    def merge(self, attended):
        return F.linear(attended.transpose(1, 2).flatten(2), self.out, self.out_bias)

    def forward(self, x):
        q, k, v = self.project(x)
        weights = (q @ k.transpose(-1, -2)).softmax(dim=-1)
        return self.merge(weights @ v)


def attend_with_external_history(keys, values, q, k, v, mask_bias):
    # Score 99 past slots and the current frame separately. The caller commits
    # each row delta only after inference has finished reading the old rings.
    rows, history, channels = keys.shape
    query = q.reshape(rows, channels, 1)
    key, value = k.reshape(rows, 1, channels), v.reshape(rows, 1, -1)
    scores = torch.cat((keys @ query, key @ query), dim=1).reshape(q.size(0), q.size(1), 1, -1)
    weights = (scores + mask_bias).softmax(dim=-1)
    weights = weights.reshape(rows, 1, -1)
    attended = weights[..., :history] @ values + weights[..., history:] * value
    return attended.reshape_as(v), key.squeeze(1), value.squeeze(1)


def single_step_gru(ffn, x, hidden):
    # Compute the three input and three recurrent gates in two projections.
    # The reset gate applies only to the recurrent part of the candidate.
    gru = ffn.gru
    size = gru.hidden_size
    x = x.squeeze(1)
    states = []
    for layer, previous in enumerate(hidden.unbind()):
        input_projection = F.linear(x, getattr(gru, f"weight_ih_l{layer}"), getattr(gru, f"bias_ih_l{layer}"))
        recurrent_projection = F.linear(previous, getattr(gru, f"weight_hh_l{layer}"), getattr(gru, f"bias_hh_l{layer}"))
        input_gates, input_candidate = input_projection.split((2 * size, size), dim=-1)
        recurrent_gates, recurrent_candidate = recurrent_projection.split((2 * size, size), dim=-1)
        reset, update = (input_gates + recurrent_gates).sigmoid().split(size, dim=-1)
        candidate = (input_candidate + reset * recurrent_candidate).tanh()
        x = candidate + update * (previous - candidate)
        states.append(x)
    return ffn.linear(F.leaky_relu(x)).unsqueeze(1), torch.stack(states)


def stream_complex_ffn(ffn, angle, input_history, mid_history):
    def project(conv, history, current):
        # Append and evict whole [frequency, complex channels] planes. Feed the
        # original [channel, tap] order to the GEMM to preserve FP32 accumulation.
        window = torch.cat((history, current.transpose(0, 1)), dim=0)
        weight = conv.fused_conv.weight.flatten(1)
        return F.linear(window.permute(1, 2, 0).flatten(1), weight), window[1:]

    projected, input_history = project(ffn.conv1d, input_history, angle)
    real, gate_real, imag, gate_imag = projected.split(ffn.chn_inner, dim=-1)
    magnitude = (gate_real.square() + gate_imag.square() + 1e-9).sqrt()
    gate = ffn.gate_act(ffn.gate_ln(magnitude))
    current = torch.cat((real * gate, imag * gate), dim=-1).unsqueeze(1)
    out, mid_history = project(ffn.causal_conv_out, mid_history, current)
    return out.unsqueeze(1), input_history, mid_history


class SeqNormPair(nn.Module):
    """Normalize the packed complex features and amplitude independently."""

    def __init__(self, cnorm, norm):
        super().__init__()
        self.ang_channels = cnorm.gamma.numel()
        self.amp_channels = norm.g.numel()
        # The mean over [real, imag] is half the mean of real² + imag².
        # Halve epsilon and divide the complex gain by sqrt(2) to compensate.
        self.angle_eps = cnorm.eps * 0.5
        self.amp_eps = norm.eps
        self.register_buffer("angle_scale", cnorm.gamma.detach().repeat(2) / math.sqrt(2))
        self.register_buffer("amp_scale", norm.g.detach())

    def forward(self, x):
        angle, amp = x.split((2 * self.ang_channels, self.amp_channels), dim=-1)
        angle = self.rms_norm(angle, self.angle_scale, self.angle_eps)
        amp = self.rms_norm(amp, self.amp_scale, self.amp_eps)
        return angle, amp

    @staticmethod
    def rms_norm(x, scale, eps):
        return (x / (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()) * scale


def time_step(block, x, keys, values, caches):
    mask_bias, hidden, input_history, mid_history = caches
    att = block.att
    angle, amp = block.merged1(x)
    q, k, v = att.project(torch.cat((angle, amp), dim=-1))
    attended, next_key, next_value = attend_with_external_history(keys, values, q, k, v, mask_bias)
    x = x + F.linear(attended.reshape(x.size(0), 1, -1), att.out, att.out_bias)
    angle, amp = block.merged2(x)
    a, hidden = single_step_gru(block.amp_ffn, amp, hidden)
    angle, input_history, mid_history = stream_complex_ffn(block.ang_ffn, angle, input_history, mid_history)
    y = torch.cat(block.merged3(x + torch.cat((angle, a), dim=-1)), dim=-1)
    return y, next_key, next_value, (hidden, input_history, mid_history)


def prepare_frequency_ffn(ffn):
    # The frequency path crops kernel_size - 1 bins from each end of the
    # unit-stride transposed convolution. This is a valid convolution.
    real, imag = ffn.deconv1d.conv_re, ffn.deconv1d.conv_im
    if ffn.conv1d_shift != 1 or ffn.diff_ks != real.kernel_size[-1] - 1:
        raise ValueError("The frequency FFN export requires unit stride and a kernel-width crop")
    weight = complex_weights(real.weight.transpose(0, 1), imag.weight.transpose(0, 1))
    ffn.register_buffer("freq_out_weight", weight.detach().flip(-1).contiguous())


def frequency_complex_ffn(ffn, angle):
    # Explicit [N, C, 1, F] convolutions keep frequency on the width axis for cuDNN's FP32 kernels.
    x = angle.transpose(1, 2).unsqueeze(2)
    weight = ffn.conv1d.fused_conv.weight.unsqueeze(2)
    x = F.conv2d(x, weight, padding=(0, ffn.diff_ks)).squeeze(2).transpose(1, 2)
    real, gate_real, imag, gate_imag = x.split(ffn.chn_inner, dim=-1)
    magnitude = (gate_real.square() + gate_imag.square() + 1e-9).sqrt()
    gate = ffn.gate_act(ffn.gate_ln(magnitude))
    x = torch.cat((real * gate, imag * gate), dim=-1).transpose(1, 2).unsqueeze(2)
    return F.conv2d(x, ffn.freq_out_weight).squeeze(2).transpose(1, 2)


def freq_block(block, x):
    angle, amp = block.merged1(x)
    x = x + block.att(torch.cat((angle, amp), dim=-1))
    angle, amp = block.merged2(x)
    y = torch.cat((frequency_complex_ffn(block.ang_ffn, angle), block.amp_ffn(amp)), dim=-1)
    return torch.cat(block.merged3(x + y), dim=-1)


class FastStreamNorm(nn.Module):
    """Amplitude and complex streaming RMS norms sharing one scalar chain.

    The upstream implementation divides by ``count + 1`` and immediately
    multiplies it back while blending the prior, so both divisions collapse
    into ``(hist + power + prior_power * prior_count) / (count + 1 + prior_count)``.
    """

    def __init__(self, norm_ang, norm_amp):
        super().__init__()
        self.norm_ang = norm_ang
        self.norm_amp = norm_amp
        ang_channels = norm_ang.gain_matrix.size(1)
        amp_channels = norm_amp.gain_matrix.size(1)
        self.ang_channels = ang_channels
        self.amp_channels = amp_channels
        self.prior_gain = float(norm_amp._prior_power) * float(norm_amp._prior_count)
        self.prior_denom = 1.0 + float(norm_amp._prior_count)
        self.min_power = float(norm_amp._min_power)
        self.max_gain = float(norm_amp.max_gain)
        eps = torch.cat((torch.full((1, ang_channels, 1, 1), ANG_EPS), torch.full((1, amp_channels, 1, 1), float(norm_amp.eps))), dim=1)
        self.register_buffer("eps", eps)

    def forward(self, real, imag, amp, hist_sum, count_cache):
        stats = torch.cat((real.square() + imag.square() + 1e-9, amp.square()), dim=1)
        power_t = stats.mean(dim=3, keepdim=True)
        mean = (hist_sum + power_t + self.prior_gain) / (count_cache + self.prior_denom)
        mean = torch.clamp(mean, min=self.min_power)
        gain = (mean + self.eps).pow(-0.5)
        gain = torch.clamp(gain, max=self.max_gain)
        gain_ang, gain_amp = gain.split((self.ang_channels, self.amp_channels), dim=1)
        angle_gain = gain_ang * self.norm_ang.gain_matrix
        real_out = real * angle_gain
        imag_out = imag * angle_gain
        amp_out = amp * (gain_amp * self.norm_amp.gain_matrix) + self.norm_amp.bias_matrix
        return real_out, imag_out, amp_out, power_t


def learnable_sigmoid(module, x):
    return module.beta * torch.sigmoid(module.slope * x)


class FastDenseLayer(nn.Module):
    """One dense layer: projected tap delays, merged norm, gating.

    Each convolution emits the current tap before the delayed tap so a single
    Split separates them; this keeps the original FLOPs while halving the
    slice/concat plumbing.
    """

    def __init__(self, layer, layer_index):
        super().__init__()
        self.ang_out = layer.ang_out_chn
        self.amp_out = layer.amp_out_chn
        ang_in = layer.ang_in_chn
        angle = layer.conv_ang.weight.detach()
        angle = torch.cat((angle[:, :, 1:], angle[:, :, :1]), dim=0)
        permutation = self._angle_permutation(layer_index, ang_in)
        angle = angle[:, permutation].contiguous()
        amp_weight = torch.cat((layer.conv_amp.weight[:, :, 1:], layer.conv_amp.weight[:, :, :1]), dim=0)
        self.register_buffer("angle_weight", angle)
        self.register_buffer("amp_weight", amp_weight.detach())
        bias = torch.cat((layer.conv_amp.bias.detach(), torch.zeros(self.amp_out)))
        self.register_buffer("amp_bias", bias)
        self.layer = layer
        self.norm = FastStreamNorm(layer.norm_ang, layer.norm_amp)
        self.channels = 2 * self.ang_out + self.amp_out

    @staticmethod
    def _angle_permutation(layer_index, ang_in):
        # Canonical columns are [all cos | all sin] ordered newest to oldest; the
        # incremental skip keeps [cos_i, sin_i, cos_{i-1}, sin_{i-1}, ...].
        per_level = ang_in // (layer_index + 1)
        cos = {level: list(range(level * per_level, (level + 1) * per_level)) for level in range(layer_index + 1)}
        sin = {level: [ang_in + index for index in cos[level]] for level in range(layer_index + 1)}
        layout = []
        for level in range(layer_index, -1, -1):
            layout += cos[level] + sin[level]
        canonical = []
        for level in range(layer_index, -1, -1):
            canonical += cos[level]
        for level in range(layer_index, -1, -1):
            canonical += sin[level]
        return torch.tensor([canonical.index(index) for index in layout], dtype=torch.long)

    def forward(self, angle, amp, history, hist_sum, count_cache):
        angle_out = F.conv2d(angle, self.angle_weight, padding=(0, 1))
        amp_out = F.conv2d(amp, self.amp_weight, self.amp_bias, padding=(0, 1))
        now_angle, past_angle = angle_out.split((2 * self.ang_out, 2 * self.ang_out), dim=1)
        now_amp, past_amp = amp_out.split((self.amp_out, self.amp_out), dim=1)
        present = torch.cat((now_angle, now_amp), dim=1) + history[:1]
        real, imag, amp_present = present.split((self.ang_out, self.ang_out, self.amp_out), dim=1)
        real, imag, amp_out, power_t = self.norm(real, imag, amp_present, hist_sum, count_cache)
        amp_out = self.layer.act_amp(amp_out)
        magnitude = (real.square() + imag.square() + 1e-9).sqrt()
        w_amp2ang = learnable_sigmoid(self.layer.act_amp2ang, self.layer.pconv_amp2ang(amp_out))
        w_ang2amp = learnable_sigmoid(self.layer.act_ang2amp, self.layer.pconv_ang2amp(magnitude))
        cos_out = w_amp2ang * real
        sin_out = w_amp2ang * imag
        amp_out = w_ang2amp * amp_out
        past = torch.cat((past_angle, past_amp), dim=1)
        history_out = torch.cat((history[1:], past), dim=0)
        return cos_out, sin_out, amp_out, history_out, power_t


class FastConvBlock(nn.Module):
    """Streaming InteConvBlock (encoder head/tail, decoder upsampling)."""

    def __init__(self, block, simple):
        super().__init__()
        self.block = block
        self.simple = simple
        self.norm = FastStreamNorm(block.norm_ang, block.norm_amp)
        self.ang_out = block.ang_out_chn
        self.amp_out = block.amp_out_chn
        # block.conv_ang is already a FusedComplexConv after fuse_complex_convs,
        # except for the class-transposed decoder block which keeps its module.
        self.transpose_block = hasattr(block, "conv_ang") and hasattr(block.conv_ang, "pad1")

    def forward(self, angle, amp, hist_sum, count_cache):
        block = self.block
        if self.transpose_block:
            real, imag = angle.split((self.ang_out, self.ang_out), dim=1)
            real, imag = block.conv_ang(real, imag)
        else:
            conv = block.conv_ang
            real, imag = F.conv2d(
                angle, conv.weight, stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
            ).chunk(2, dim=1)
        amp_features = block.conv_amp(amp)
        real, imag, amp_features, power_t = self.norm(real, imag, amp_features, hist_sum, count_cache)
        amp_out = block.act_amp(amp_features)
        if self.simple:
            return real, imag, amp_out, power_t
        magnitude = (real.square() + imag.square() + 1e-9).sqrt()
        w_amp2ang = learnable_sigmoid(block.act_amp2ang, block.pconv_amp2ang(amp_out))
        w_ang2amp = learnable_sigmoid(block.act_ang2amp, block.pconv_ang2amp(magnitude))
        return w_amp2ang * real, w_amp2ang * imag, w_ang2amp * amp_out, power_t


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


class StaticStream(nn.Module):
    def __init__(self, stream):
        super().__init__()
        stream.fuse_inference_kernels()
        self.net = stream.mpnet
        net = self.net
        for dense in (net.dense_encoder.dense_block, net.decoder.dense_block):
            for _, layer in dense.dense_block:
                for conv in (layer.conv_ang.conv_re, layer.conv_ang.conv_im, layer.conv_amp):
                    conv.dilation = (1, 1)
        for block in net.TSTransformer:
            for transformer in (block.time_transformer, block.freq_transformer):
                transformer.att = PackedAttention(transformer.att)
                transformer.merged1 = SeqNormPair(transformer.cnorm1, transformer.norm1)
                transformer.merged2 = SeqNormPair(transformer.cnorm2, transformer.norm2)
                transformer.merged3 = SeqNormPair(transformer.cnorm3, transformer.norm3)
        fuse_complex_convs(net)
        for block in net.TSTransformer:
            prepare_frequency_ffn(block.freq_transformer.ang_ffn)
        self.encoder_head = FastConvBlock(net.dense_encoder.dense_conv_1, simple=True)
        self.encoder_tail = FastConvBlock(net.dense_encoder.dense_conv_3, simple=False)
        self.decoder_upsample = FastConvBlock(net.decoder.upsample_layer, simple=False)
        self.dense_encoder = nn.ModuleList([FastDenseLayer(layer, i) for i, (_, layer) in enumerate(net.dense_encoder.dense_block.dense_block)])
        self.dense_decoder = nn.ModuleList([FastDenseLayer(layer, i) for i, (_, layer) in enumerate(net.decoder.dense_block.dense_block)])

    def _dense_block(self, layers, angle, amp, histories, power_sums, count_bank, offset):
        next_histories, powers = [], []
        last = None
        for index, layer in enumerate(layers):
            cos_out, sin_out, amp_out, history_out, power_t = layer(
                angle, amp, histories[index], power_sums[offset + index], count_bank[offset + index])
            next_histories.append(history_out)
            powers.append(power_t)
            last = (cos_out, sin_out, amp_out)
            angle = torch.cat((cos_out, sin_out, angle), dim=1)
            amp = torch.cat((amp_out, amp), dim=1)
        return last, next_histories, powers

    def forward(self, amp, phase, *state):
        net = self.net
        keys, values = state[8:12], state[12:16]
        cursor, counts, hidden = state[16:19]
        in_histories, mid_histories = state[19:23], state[23:27]
        enc_power, enc_count, dec_power, dec_count = state[27:31]
        # Reduce the history axis once per bank, keeping separate sums for every layer and channel.
        enc_sum = enc_power.sum(dim=3, keepdim=True)
        dec_sum = dec_power.sum(dim=3, keepdim=True)
        amplitude, power, count, gain = net.input_rms.forward_stream(
            amp[:, None].transpose(2, 3), *state[31:], return_gain=True)
        cos = phase.cos().transpose(1, 2)[:, None]
        sin = phase.sin().transpose(1, 2)[:, None]
        angle = torch.cat((cos, sin), dim=1)

        real, imag, amplitude, power0 = self.encoder_head(angle, amplitude, enc_sum[0], enc_count[0])
        angle = torch.cat((real, imag), dim=1)
        last, enc_histories, powers = self._dense_block(
            self.dense_encoder, angle, amplitude, state[:4], enc_sum, enc_count, 1)
        angle = torch.cat((last[0], last[1]), dim=1)
        real, imag, amp_out, power5 = self.encoder_tail(angle, last[2], enc_sum[5], enc_count[5])
        x = torch.cat((real, imag, amp_out), dim=1).squeeze(2).transpose(1, 2)
        enc_power_out = torch.cat((enc_power[:, :, :, 1:], torch.stack((power0, *powers, power5))), dim=3)
        enc_count_out = (enc_count + 1).clamp(max=enc_power.size(3))

        next_keys, next_values, next_hidden = [], [], []
        next_in, next_mid = [], []
        past = keys[0].size(1)
        # Bias only unused ring slots. The appended current frame is always valid.
        invalid = torch.arange(past, dtype=counts.dtype) >= counts
        bias = invalid.to(counts.dtype) * torch.finfo(counts.dtype).min
        masks = F.pad(bias, (0, 1)).reshape(len(keys), -1, 1, 1, past + 1)
        for i, block in enumerate(net.TSTransformer):
            x = x.reshape(-1, 1, x.size(-1))
            layer_state = (masks[i], hidden[i, :, 0], in_histories[i], mid_histories[i])
            y, key, value, (new_hidden, new_in, new_mid) = time_step(block.time_transformer, x, keys[i], values[i], layer_state)
            next_keys.append(key)
            next_values.append(value)
            next_hidden.append(new_hidden)
            next_in.append(new_in)
            next_mid.append(new_mid)
            x = (x + y).reshape(1, -1, x.size(-1))
            x = x + freq_block(block.freq_transformer, x)
        x = x.transpose(1, 2).unsqueeze(2)

        angle, amp_out = x.split((2 * net.decoder.ang_chn, net.decoder.amp_chn), dim=1)
        last, dec_histories, powers = self._dense_block(
            self.dense_decoder, angle, amp_out, state[4:8], dec_sum, dec_count, 0)
        angle = torch.cat((last[0], last[1]), dim=1)
        real, imag, amp_out, power4 = self.decoder_upsample(angle, last[2], dec_sum[4], dec_count[4])
        dec_power_out = torch.cat((dec_power[:, :, :, 1:], torch.stack((*powers, power4))), dim=3)
        dec_count_out = (dec_count + 1).clamp(max=dec_power.size(3))

        real, imag = net.decoder.ang_conv(real, imag)
        phase = phase_angle(imag + 1e-9, real + 1e-9).squeeze(1).transpose(1, 2)
        amp_out = net.decoder.amp_conv_layer(amp_out).squeeze(1).transpose(1, 2) / gain.squeeze(-1)
        spectrum = torch.stack((amp_out * phase.cos(), amp_out * phase.sin()), dim=-1)
        counts = (counts + 1).clamp(max=keys[0].size(1))
        hidden = torch.stack(next_hidden).unsqueeze(2)
        cursor = torch.remainder(cursor + 1, keys[0].shape[1])
        return (
            amp_out, phase, spectrum, *enc_histories, *dec_histories, *next_keys, *next_values, cursor,
            counts, hidden, *next_in, *next_mid, enc_power_out, enc_count_out, dec_power_out, dec_count_out, power, count,
        )


def lower_square_ops(graph):
    squares = {
        node.output[0]
        for node in graph.node
        if node.op_type == "Constant" and not node.domain
        for attr in node.attribute
        if attr.name == "value" and not attr.t.dims and numpy_helper.to_array(attr.t).item() == 2
    }
    for node in graph.node:
        # Keep Pow in SeqNormPair so ORT can fuse the complete RMSNorm pattern.
        if node.name.startswith("/merged"):
            continue
        if node.op_type == "Pow" and not node.domain and node.input[1] in squares:
            node.op_type = "Mul"
            node.input[1] = node.input[0]


def build_cache(stream):
    cache = list(stream.init_stream_cache(1, 201))
    # The caller commits frame updates after attention; the ring holds only past frames.
    rings = []
    for bank in cache[8:10]:
        layers, batch, bins, heads, past, channels = bank.shape
        rings.extend(bank.new_zeros(batch * bins * heads, past, channels) for _ in range(layers))
    cache[8:11] = [*rings, torch.zeros(1, dtype=torch.float32)]
    # Dense histories store the projected past tap: 2 * ang + amp channels.
    channels = cache[0].size(1)
    for index, state in enumerate(cache[:8]):
        cache[index] = state.new_zeros(state.size(2), channels, 1, state.size(3))
    cache[23] = torch.zeros(ENC_NORM_COUNT, 1, 64, 99, 1)
    cache[24] = torch.zeros(ENC_NORM_COUNT, 1, 1, 1, 1)
    cache[25] = torch.zeros(DEC_NORM_COUNT, 1, 64, 99, 1)
    cache[26] = torch.zeros(DEC_NORM_COUNT, 1, 1, 1, 1)
    histories = []
    for bank in (cache[19], cache[21]):
        layers, _, bins, channels, taps = bank.shape
        histories.extend(bank.new_zeros(taps - 1, bins, 2 * channels) for _ in range(layers))
    return cache[:19] + histories + cache[23:27] + cache[31:33]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="GRE-Net-Streaming checkout")
    parser.add_argument("--output", type=Path, default=Path(MODEL_FILENAME), help="Output path (defaults to stream.onnx)")
    args = parser.parse_args()
    output = args.output
    stream = load_stream(args.source)
    from models import model

    names_in = ["noisy_amp_t", "noisy_pha_t", *CACHE_NAMES]
    names_out = ["denoised_amp_t", "denoised_pha_t", "denoised_com_t", *(f"{name}_out" for name in CACHE_NAMES)]
    for i in range(8, 16):
        names_out[3 + i] = f"{CACHE_NAMES[i]}_frame"
    model._shift_append_cache_dim2 = shift_history
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cache = build_cache(stream)
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
    count, fused = fuse_frequency_attention(simplified)
    if count != 4:
        raise RuntimeError(f"Expected four frequency attention subgraphs, matched {count}")
    onnx.checker.check_model(fused)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(fused, output)


if __name__ == "__main__":
    main()
