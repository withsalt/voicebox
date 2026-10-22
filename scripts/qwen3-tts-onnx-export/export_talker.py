# Copyright (c) 2026
# Export the Qwen3-TTS talker (main autoregressive transformer) to ONE ONNX graph that
# serves both prefill (past length P=0) and decode (P>0) -- matching the single-graph KV
# pattern of the C# decoder. Halves package size vs separate prefill/decode graphs.
#
#   talker.onnx
#     in : inputs_embeds[1,S,H], attention_mask[1,P+S], position_ids[3,1,S],
#          past_k[L,1,Hkv,P,D], past_v[L,1,Hkv,P,D]
#     out: logits[1,S,Vc], hidden[1,S,H], present_k[L,1,Hkv,P+S,D], present_v[...]
#
# Export learnings reused from the codec: build the additive mask explicitly (HF builds it
# via torch.vmap -> untraceable); reshape with (B,-1,H,D) not .view(*shape[:-1],...) (which
# bakes S); run the 28 layers manually. mrope -> STANDARD rope: for CustomVoice all three
# mrope position rows are identical, so the interleaved combine == row 0.
import os
import argparse
import numpy as np
import torch
import torch.nn as nn

import _common as C


def _talker_ops():
    from qwen_tts.core.models import modeling_qwen3_tts as M
    return M.apply_rotary_pos_emb, M.repeat_kv


class TalkerStep(nn.Module):
    """Unified manual talker forward (prefill P=0 / decode P>0), ONNX-friendly."""

    def __init__(self, talker):
        super().__init__()
        self.talker = talker
        self.model = talker.model
        self.codec_head = talker.codec_head
        self.rope, self.repeat_kv = _talker_ops()
        c = talker.config
        self.nh, self.nkv, self.D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.groups = self.nh // self.nkv

    def _mask(self, inputs_embeds, attention_mask):
        # built in fp32 regardless of model dtype: CumSum has no bf16 CUDA kernel, and the mask is
        # added to the (already fp32) attention scores anyway.
        dev = inputs_embeds.device
        neg = torch.finfo(torch.float32).min
        ref = inputs_embeds[:, :, 0].float()                                         # [B,S] fp32
        s_idx = torch.cumsum(torch.ones_like(ref), dim=1) - 1                         # [B,S] 0..S-1
        k_ones = torch.ones_like(attention_mask, dtype=torch.float32)                # [B,P+S]
        k_idx = torch.cumsum(k_ones, dim=1) - 1                                       # [B,P+S] 0..total-1
        s_len = s_idx[:, -1:] + 1
        t_len = k_idx[:, -1:] + 1
        qabs = (t_len - s_len) + s_idx                                                # [B,S] absolute query index
        allowed = k_idx[:, None, :] <= qabs[:, :, None]                              # [B,S,total] causal
        mask = torch.where(allowed, torch.zeros((), dtype=torch.float32, device=dev),
                           torch.full((), neg, dtype=torch.float32, device=dev))[:, None]
        keypad = (1.0 - attention_mask.float())[:, None, None, :] * neg              # key padding (batch>1)
        return mask + keypad

    def forward(self, inputs_embeds, attention_mask, position_ids, past_k, past_v):
        m = self.model
        cos, sin = m.rotary_emb(inputs_embeds, position_ids)   # [3,B,S,D]
        cos, sin = cos[0], sin[0]                              # rows equal -> standard rope
        mask = self._mask(inputs_embeds, attention_mask)
        B = inputs_embeds.shape[0]
        h = inputs_embeds
        present_k, present_v = [], []
        for i, layer in enumerate(m.layers):
            attn = layer.self_attn
            res = h
            x = layer.input_layernorm(h)
            q = attn.q_norm(attn.q_proj(x).reshape(B, -1, self.nh, self.D)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(x).reshape(B, -1, self.nkv, self.D)).transpose(1, 2)
            v = attn.v_proj(x).reshape(B, -1, self.nkv, self.D).transpose(1, 2)
            q, k = self.rope(q, k, cos, sin)
            k = torch.cat([past_k[i], k], dim=2)
            v = torch.cat([past_v[i], v], dim=2)
            present_k.append(k)
            present_v.append(v)
            kk = self.repeat_kv(k, self.groups)
            vv = self.repeat_kv(v, self.groups)
            # fp16 q@kᵀ overflows on CUDA Tensor Cores -> compute scores in fp32. fp32 (identity) and
            # bf16 (fp32 exponent range, no overflow) keep the native matmul -> far fewer Cast nodes.
            if q.dtype == torch.float16:
                scores = torch.matmul(q.float(), kk.float().transpose(2, 3)) * attn.scaling + mask.float()
            else:
                scores = torch.matmul(q, kk.transpose(2, 3)) * attn.scaling + mask.to(q.dtype)
            w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(vv.dtype)
            o = torch.matmul(w, vv).transpose(1, 2).reshape(B, -1, self.nh * self.D)
            h = res + attn.o_proj(o)
            res = h
            h = res + layer.mlp(layer.post_attention_layernorm(h))
        h = m.norm(h)
        return self.codec_head(h), h, torch.stack(present_k, 0), torch.stack(present_v, 0)


def _pos(S, start=0):
    return torch.arange(start, start + S, dtype=torch.long).view(1, 1, S).expand(3, 1, S).contiguous()


def _empty_past(L, kv, D):
    return torch.zeros(L, 1, kv, 0, D), torch.zeros(L, 1, kv, 0, D)


class TalkerStepStatic(nn.Module):
    """Static-KV talker decode step (CUDA-graph friendly: fully static shapes).

      in : inputs_embeds[1,1,H], cache_position(i64 scalar), position_ids[3,1,1],
           past_k[L,1,kv,MAXLEN,D], past_v[L,1,kv,MAXLEN,D]   (fixed-length buffers)
      out: logits[1,1,Vc], hidden[1,1,H], present_k[L,1,kv,MAXLEN,D], present_v[...]

    The new K/V are SCATTERED into the fixed buffer at cache_position (in-place via scatter), so
    present has the SAME shape as past -> the C# side binds present->past device-resident with a
    stable address (CUDA-graph replay). The attention mask is full-length [1,1,1,MAXLEN]: key j is
    allowed iff j <= cache_position (causal + masks the unfilled tail). Single-token decode only
    (S=1); prefill uses the dynamic talker.onnx. Math identical to TalkerStep -> byte-exact."""

    def __init__(self, talker, max_len):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head
        self.rope, self.repeat_kv = _talker_ops()
        c = talker.config
        self.nh, self.nkv, self.D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.groups = self.nh // self.nkv
        self.max_len = int(max_len)

    def forward(self, inputs_embeds, cache_position, position_ids, past_k, past_v):
        m = self.model
        cos, sin = m.rotary_emb(inputs_embeds, position_ids)   # [3,B,1,D]
        cos, sin = cos[0], sin[0]                              # rows equal -> standard rope
        B = inputs_embeds.shape[0]
        dev = inputs_embeds.device

        # full-length additive mask [1,1,1,MAXLEN] (fp32): key j allowed iff j <= cache_position.
        neg = torch.finfo(torch.float32).min
        key_idx = torch.arange(self.max_len, device=dev)                      # [MAXLEN]
        allowed = key_idx <= cache_position                                   # [MAXLEN] bool
        mask = torch.where(allowed, torch.zeros((), dtype=torch.float32, device=dev),
                           torch.full((), neg, dtype=torch.float32, device=dev)).view(1, 1, 1, self.max_len)

        # scatter index for writing the new K/V at cache_position along the seq dim (dim 2 of [1,kv,MAXLEN,D]).
        scatter_idx = cache_position.view(1, 1, 1, 1).expand(1, self.nkv, 1, self.D)  # [1,kv,1,D]

        h = inputs_embeds
        present_k, present_v = [], []
        for i, layer in enumerate(m.layers):
            attn = layer.self_attn
            res = h
            x = layer.input_layernorm(h)
            q = attn.q_norm(attn.q_proj(x).reshape(B, -1, self.nh, self.D)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(x).reshape(B, -1, self.nkv, self.D)).transpose(1, 2)
            v = attn.v_proj(x).reshape(B, -1, self.nkv, self.D).transpose(1, 2)
            q, k = self.rope(q, k, cos, sin)                  # k,v: [1,kv,1,D]
            # write new k/v into the fixed buffer at cache_position -> present (same [1,kv,MAXLEN,D]).
            pk = past_k[i].scatter(2, scatter_idx, k)
            pv = past_v[i].scatter(2, scatter_idx, v)
            present_k.append(pk)
            present_v.append(pv)
            kk = self.repeat_kv(pk, self.groups)
            vv = self.repeat_kv(pv, self.groups)
            if q.dtype == torch.float16:
                scores = torch.matmul(q.float(), kk.float().transpose(2, 3)) * attn.scaling + mask.float()
            else:
                scores = torch.matmul(q, kk.transpose(2, 3)) * attn.scaling + mask.to(q.dtype)
            w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(vv.dtype)
            o = torch.matmul(w, vv).transpose(1, 2).reshape(B, -1, self.nh * self.D)
            h = res + attn.o_proj(o)
            res = h
            h = res + layer.mlp(layer.post_attention_layernorm(h))
        h = m.norm(h)
        return self.codec_head(h), h, torch.stack(present_k, 0), torch.stack(present_v, 0)


def _export_static(cfg, H, L, kv, D, g):
    """Export the static-KV decode graph (talker_static.onnx) + validate byte-exact vs dynamic step.
    Loads its OWN fp32 model copy for validation (run() may have mutated the shared talker to bf16/CUDA)
    and a dtype copy for the actual export, so it's independent of run()'s in-place .to(dtype)."""
    max_len = C.STATIC_MAX_LEN
    print(f"== static-KV talker (max_len={max_len}) ==")
    ref_tts = C.load_tts(device="cpu", dtype="float32")
    ref_talker = ref_tts.model.talker.eval()
    step = TalkerStep(ref_talker).eval()
    sstep = TalkerStepStatic(ref_talker, max_len).eval()

    # Build a few-token history with the DYNAMIC step, then check the STATIC step reproduces the
    # next-token decode byte-exact (the static path is decode-only; prefill uses talker.onnx).
    T = 6
    emb = torch.randn(1, T, H, generator=g) * 0.1
    am = torch.ones(1, T, dtype=torch.long)
    pk0, pv0 = _empty_past(L, kv, D)
    with torch.inference_mode():
        _, _, pk, pv = step(emb, am, _pos(T), pk0, pv0)        # dynamic prefill -> [L,1,kv,T,D]
        emb1 = torch.randn(1, 1, H, generator=g) * 0.1
        dlog, dh, _, _ = step(emb1, torch.ones(1, T + 1, dtype=torch.long), _pos(1, start=T), pk, pv)

        # static buffers: zero [L,1,kv,MAXLEN,D], copy the T-length history into [:T].
        spk = torch.zeros(L, 1, kv, max_len, D)
        spv = torch.zeros(L, 1, kv, max_len, D)
        spk[:, :, :, :T, :] = pk
        spv[:, :, :, :T, :] = pv
        cache_pos = torch.tensor(T, dtype=torch.long)
        spos = _pos(1, start=T)
        slog, sh, npk, npv = sstep(emb1, cache_pos, spos, spk, spv)
    C.compare("talker_static_decode_vs_dynamic_logits", dlog, slog, atol_warn=1e-4)
    C.compare("talker_static_decode_vs_dynamic_hidden", dh, sh, atol_warn=1e-4)

    path = os.path.join(C.PKG_DIR, "talker_static.onnx")
    print("== exporting", path)
    ex = (emb1, cache_pos, spos, spk, spv)
    mod = sstep
    if C.DTYPE in ("fp16", "bf16"):
        dt = torch.float16 if C.DTYPE == "fp16" else torch.bfloat16
        mod = TalkerStepStatic(ref_talker.to(dt).to(C.DEVICE), max_len).eval()
        ex = (emb1.to(dt).to(C.DEVICE), cache_pos.to(C.DEVICE), spos.to(C.DEVICE),
              spk.to(dt).to(C.DEVICE), spv.to(dt).to(C.DEVICE))
    C.export_onnx(
        mod, ex, path,
        input_names=["inputs_embeds", "cache_position", "position_ids", "past_k", "past_v"],
        output_names=["logits", "hidden", "present_k", "present_v"],
        dynamic_axes={},   # fully static -> CUDA-graph friendly
    )
    if C.DTYPE in ("int8", "int4"):
        C.postprocess_dtype(path, C.DTYPE)
    print(f"   wrote {path} ({os.path.getsize(path)/1e6:.0f} MB, dtype={C.DTYPE})")

    if C.DTYPE == "fp32":
        s = C.ort_session(path)
        so = s.run(None, {"inputs_embeds": emb1.numpy().astype(np.float32),
                          "cache_position": np.array(T, dtype=np.int64),
                          "position_ids": spos.numpy().astype(np.int64),
                          "past_k": spk.numpy().astype(np.float32),
                          "past_v": spv.numpy().astype(np.float32)})
        C.compare("talker_static_onnx_vs_torch", slog.numpy(), so[0])


def run(tts=None):
    if tts is None:
        print("== loading model (fp32) for talker ...")
        tts = C.load_tts(device="cpu", dtype="float32")
    talker = tts.model.talker.eval()
    step = TalkerStep(talker).eval()
    cfg = C.load_config()["talker_config"]
    H, L, kv, D = (int(cfg["hidden_size"]), int(cfg["num_hidden_layers"]),
                   int(cfg["num_key_value_heads"]), int(cfg["head_dim"]))
    g = torch.Generator().manual_seed(0)
    T = 12
    emb = torch.randn(1, T, H, generator=g) * 0.1
    am = torch.ones(1, T, dtype=torch.long)
    pos = _pos(T)
    pk0, pv0 = _empty_past(L, kv, D)

    # validate manual prefill (P=0) == original talker.model + codec_head
    with torch.inference_mode():
        out = talker.model(inputs_embeds=emb, attention_mask=am, position_ids=pos, use_cache=True)
        ref_h = out.last_hidden_state
        ref_logits = talker.codec_head(ref_h)
        my_logits, my_h, pk, pv = step(emb, am, pos, pk0, pv0)
    C.compare("talker_prefill_hidden_vs_orig", ref_h, my_h, atol_warn=1e-3)
    C.compare("talker_prefill_logits_vs_orig", ref_logits, my_logits, atol_warn=1e-3)
    print(f"   present_k={tuple(pk.shape)} (expect [{L},1,{kv},{T},{D}])")

    # validate KV cache: prefill(T) + decode(1) == full prefill(T+1)[last]
    emb2 = torch.randn(1, 1, H, generator=g) * 0.1
    am2 = torch.ones(1, T + 1, dtype=torch.long)
    with torch.inference_mode():
        dlog, dh, _, _ = step(emb2, am2, _pos(1, start=T), pk, pv)
        fout = talker.model(inputs_embeds=torch.cat([emb, emb2], 1), attention_mask=am2,
                            position_ids=_pos(T + 1), use_cache=True)
        flog = talker.codec_head(fout.last_hidden_state)[:, -1:]
    C.compare("talker_decode_kvcache_vs_full", flog, dlog, atol_warn=1e-3)

    # export single graph; trace with S=T and a NON-empty past so the concat traces cleanly,
    # dynamic axes allow P=0 (prefill) and S=1 (decode) at runtime.
    Ptrace = 4
    pkt = torch.randn(L, 1, kv, Ptrace, D, generator=g) * 0.1
    pvt = torch.randn(L, 1, kv, Ptrace, D, generator=g) * 0.1
    am_t = torch.ones(1, T + Ptrace, dtype=torch.long)
    path = os.path.join(C.PKG_DIR, "talker.onnx")
    print("== exporting", path)
    mod, ex = step, (emb, am_t, pos, pkt, pvt)
    if C.DTYPE in ("fp16", "bf16"):   # native reduced-precision trace on GPU
        dt = torch.float16 if C.DTYPE == "fp16" else torch.bfloat16
        mod = TalkerStep(talker.to(dt).to(C.DEVICE)).eval()
        ex = (emb.to(dt).to(C.DEVICE), am_t.to(C.DEVICE), pos.to(C.DEVICE),
              pkt.to(dt).to(C.DEVICE), pvt.to(dt).to(C.DEVICE))
    C.export_onnx(
        mod, ex, path,
        input_names=["inputs_embeds", "attention_mask", "position_ids", "past_k", "past_v"],
        output_names=["logits", "hidden", "present_k", "present_v"],
        dynamic_axes={"inputs_embeds": {1: "S"}, "attention_mask": {1: "TP"}, "position_ids": {2: "S"},
                      "past_k": {3: "P"}, "past_v": {3: "P"}, "logits": {1: "S"}, "hidden": {1: "S"},
                      "present_k": {3: "TP"}, "present_v": {3: "TP"}},
    )
    if C.DTYPE in ("int8", "int4"):
        C.postprocess_dtype(path, C.DTYPE)
    print(f"   wrote {path} ({os.path.getsize(path)/1e6:.0f} MB, dtype={C.DTYPE})")

    # ORT parity at fp32: prefill (P=0) and decode (P=T)
    if C.DTYPE == "fp32":
        s = C.ort_session(path)
        lo, ho, pko, pvo = s.run(None, {"inputs_embeds": emb.numpy().astype(np.float32),
                                        "attention_mask": am.numpy().astype(np.int64),
                                        "position_ids": pos.numpy().astype(np.int64),
                                        "past_k": pk0.numpy().astype(np.float32),
                                        "past_v": pv0.numpy().astype(np.float32)})
        C.compare("talker_prefill_onnx_vs_torch", my_logits.numpy(), lo)
        ld, hd, _, _ = s.run(None, {"inputs_embeds": emb2.numpy().astype(np.float32),
                                    "attention_mask": am2.numpy().astype(np.int64),
                                    "position_ids": _pos(1, start=T).numpy().astype(np.int64),
                                    "past_k": pko, "past_v": pvo})
        C.compare("talker_decode_onnx_vs_torch", dlog.numpy(), ld)
    print("talker export done.")

    # static-KV decode graph (CUDA-graph candidate). Validated byte-exact, but NOT wired into the C#
    # runtime: ORT C# can't host-seed / drive an in-place device KV cache under cuda-graph (see the
    # qwen3-tts-csharp-inference memory). Opt-in via EXPORT_STATIC_TALKER=1 so default exports stay lean.
    if os.environ.get("EXPORT_STATIC_TALKER") == "1":
        _export_static(cfg, H, L, kv, D, g)


def main():
    ap = argparse.ArgumentParser(description="Export Qwen3-TTS talker (unified prefill/decode) to ONNX.")
    C.add_io_args(ap)
    args = ap.parse_args()
    C.apply_io_args(args)
    run()
    C.gate_and_exit()


if __name__ == "__main__":
    main()
