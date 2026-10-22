# Copyright (c) 2026
# Export the Qwen3-TTS 12Hz codec decoder (audio codes -> 24kHz waveform) to ONNX.
#
# Graph:  codes[1, 16, T] (int64)  ->  wav[1, 1, T*1920] (float32, clamped [-1,1])
# Always fp32 (small, numerically sensitive). The C# side drives the chunked-decode
# loop (chunk_size / left_context) exactly like Qwen3TTSTokenizerV2Decoder.chunked_decode.
#
# HF's masking_utils builds the sliding-window causal mask via torch.vmap (even for
# eager attn), which the legacy ONNX tracer cannot export. We bypass it: build the
# mask with plain broadcasting and inject it as a precomputed dict into the
# pre_transformer forward. A pre-export check confirms this reproduces the original
# decoder bit-for-bit before we trace.
import os
import argparse
import numpy as np
import torch
import torch.nn as nn

import _common as C


def _v2_ops():
    from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as M
    return M.apply_rotary_pos_emb, M.repeat_kv


def _patch_causalconv_dynamic():
    """`_get_extra_padding_for_conv1d` computes `ideal_length - length`; the tracer bakes
    `ideal_length` (through math.ceil) to the example length while `length` stays dynamic,
    so every chunk gets padded up to the trace length. For stride-1 convs the extra padding
    is mathematically 0, so return 0 (all CausalConvNet here are stride-1)."""
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2CausalConvNet as _CC)
    _orig = _CC._get_extra_padding_for_conv1d

    def _patched(self, hidden_state):
        if self.stride == 1:
            return 0
        return _orig(self, hidden_state)

    _CC._get_extra_padding_for_conv1d = _patched


def _patch_transconv_dynamic():
    """The causal trans-conv trims `[..., :shape[-1]-right_pad]`; the tracer bakes the
    absolute end from the example length, so the graph mis-sizes other chunk lengths.
    Rewrite as a length-relative negative slice (identical math, ONNX-dynamic-safe)."""
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2CausalTransConvNet as _TC)

    def _fwd(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        if self.right_pad > 0:
            hidden_state = hidden_state[..., : -self.right_pad]
        return hidden_state.contiguous()

    _TC.forward = _fwd


class CodecDecoderExport(nn.Module):
    """Reimplements Qwen3TTSTokenizerV2Decoder.forward for ONNX:
      * sliding-window mask + positions are derived from tensor ops (cumsum of ones)
        so the time axis stays SYMBOLIC under the legacy tracer (no baked length);
      * the 8-layer pre_transformer is run manually to bypass HF's @check_model_inputs
        and its internal `cache_position = arange(inputs_embeds.shape[1])` (which the
        tracer freezes to the example length)."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.window = int(decoder.pre_transformer.window_size)
        self._rope, self._repeat_kv = _v2_ops()

    def _attn(self, attn, x, cos, sin, mask):
        B = x.shape[0]
        D = attn.head_dim
        nh = attn.config.num_attention_heads
        nkv = attn.config.num_key_value_heads
        # dynamic reshape (B,-1,H,D) keeps T symbolic; HF uses .view(*shape[:-1],...) which bakes T
        q = attn.q_norm(attn.q_proj(x).reshape(B, -1, nh, D)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).reshape(B, -1, nkv, D)).transpose(1, 2)
        v = attn.v_proj(x).reshape(B, -1, nkv, D).transpose(1, 2)
        q, k = self._rope(q, k, cos, sin)
        k = self._repeat_kv(k, attn.num_key_value_groups)
        v = self._repeat_kv(v, attn.num_key_value_groups)
        scores = torch.matmul(q, k.transpose(2, 3)) * attn.scaling + mask
        w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        o = torch.matmul(w, v).transpose(1, 2).reshape(B, -1, nh * D)
        return attn.o_proj(o)

    def _run_transformer(self, tr, hidden):
        # dynamic 0..T-1 index derived from the tensor (keeps T symbolic under legacy trace)
        ones = torch.ones_like(hidden[:, :, 0])                 # [B, T]
        pos = torch.cumsum(ones, dim=1) - 1                     # [B, T] float, 0..T-1
        position_ids = pos.to(torch.long)
        neg = torch.finfo(hidden.dtype).min
        qi = pos[:, :, None]
        kj = pos[:, None, :]
        allowed = (kj <= qi) & ((qi - kj) < self.window)
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=hidden.dtype, device=hidden.device),
            torch.full((), neg, dtype=hidden.dtype, device=hidden.device),
        )[:, None]                                              # [B, 1, T, T]

        h = tr.input_proj(hidden)
        cos, sin = tr.rotary_emb(h, position_ids)
        for layer in tr.layers:
            res = h
            h = res + layer.self_attn_layer_scale(self._attn(layer.self_attn, layer.input_layernorm(h), cos, sin, mask))
            res = h
            h = res + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(h)))
        h = tr.norm(h)
        return tr.output_proj(h)

    def forward(self, codes):                       # codes: [1, 16, T] int64
        dec = self.decoder
        codes = torch.clamp(codes, min=0)
        hidden = dec.quantizer.decode(codes)            # [B, codebook_dim, T]
        hidden = dec.pre_conv(hidden).transpose(1, 2)   # [B, T, latent]
        hidden = self._run_transformer(dec.pre_transformer, hidden)
        hidden = hidden.permute(0, 2, 1)
        for blocks in dec.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in dec.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


def _rand_codes(num_q, T, vocab=2048, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (1, num_q, T), generator=g, dtype=torch.long)


def run(tts=None):
    out_path = os.path.join(C.PKG_DIR, "codec_decoder.onnx")
    _patch_causalconv_dynamic()
    _patch_transconv_dynamic()
    if tts is None:
        print("== loading model (fp32) for codec decoder ...")
        tts = C.load_tts(device="cpu", dtype="float32")
    decoder = tts.model.speech_tokenizer.model.decoder.eval()
    num_q = int(decoder.config.num_quantizers)
    up = int(decoder.total_upsample)
    print(f"   num_quantizers={num_q} total_upsample={up} sliding_window={decoder.pre_transformer.window_size}")

    exporter = CodecDecoderExport(decoder).eval()
    codes = _rand_codes(num_q, T=64, seed=1)

    # pre-export check: injected-mask wrapper == original decoder (HF vmap mask)
    with torch.inference_mode():
        wav_ref = decoder(torch.clamp(codes, min=0)).cpu().numpy()
        wav_wrap = exporter(codes).cpu().numpy()
    C.compare("codec_wrapper_vs_original", wav_ref, wav_wrap, atol_warn=1e-4)

    print("== exporting", out_path)
    C.export_onnx(
        exporter, (codes,), out_path,
        input_names=["codes"], output_names=["wav"],
        dynamic_axes={"codes": {2: "T"}, "wav": {2: "L"}},
    )
    sz = os.path.getsize(out_path) / 1e6
    print(f"   wrote {out_path} ({sz:.1f} MB)")

    # parity: ORT vs torch at several chunk lengths; T=128 exercises the sliding-window (>72) region.
    sess = C.ort_session(out_path)
    for T in (128, 64, 41):
        cc = _rand_codes(num_q, T=T, seed=T)
        with torch.inference_mode():
            wav_t = exporter(cc).cpu().numpy()
        wav_o = sess.run(["wav"], {"codes": cc.numpy().astype(np.int64)})[0]
        C.compare(f"codec_onnx_vs_torch_T{T}", wav_t, wav_o, atol_warn=1e-3)

    print("codec decoder export done.")


def main():
    ap = argparse.ArgumentParser(description="Export Qwen3-TTS codec decoder to ONNX.")
    C.add_io_args(ap)
    args = ap.parse_args()
    C.apply_io_args(args)
    run()
    C.gate_and_exit()


if __name__ == "__main__":
    main()
