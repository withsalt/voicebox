# Copyright (c) 2026
# Export the Qwen3-TTS code-predictor (sub-talker) as ONE FUSED ONNX graph that produces a
# whole frame's 15 fine codes in a single Run -- the 15 autoregressive steps, the per-step
# input-embedding lookup, and the sampling are all unrolled INSIDE the graph. This collapses
# the old 15 Runs/frame (+ 15 C#<->GPU KV round-trips + 15 C# embedding gathers) down to 1.
#
#   code_predictor_sample.onnx / code_predictor_greedy.onnx
#                         (fused, static shapes; runtime chooses by subtalker_do_sample)
#     in : past_hidden[1,1,Hin]   talker hidden that produced group0 (native dtype)
#          emb0[1,1,Hin]          = codec_embedding[group0], gathered in C# (native dtype)
#          do_sample(i64 scalar)  0=greedy argmax, >0=sample
#          temperature(f32 scalar)
#          randoms[G-1] (f32)     one uniform per step for inverse-CDF multinomial sampling
#     out: fine_codes[G-1] (i64)  the 15 fine codec groups
#          next_sum[1,1,Hin]      = emb0 + Σ cp_codec_embedding[gs][code_gs]  (talker next-input
#                                   embedding sum, native dtype; C# only adds tts_pad)
#
# Sampling matches the C# Sampler / HF warper order: temperature -> top-k -> softmax ->
# multinomial (implemented as inverse-CDF over the cumulative probs with the supplied uniform).
# top_k is baked from generation_config (subtalker_top_k); temperature is a runtime input.
# Greedy (do_sample=0) is argmax over the raw logits -> byte-exact vs the PyTorch greedy path.
#
# cp_codec_embedding[G-1,Vcp,Hin] and the 15 lm_head weights are baked as initializers (the
# small_to_mtp_projection and 5 transformer layers run inside the graph as before).
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

import _common as C


def _ops():
    from qwen_tts.core.models import modeling_qwen3_tts as M
    return M.apply_rotary_pos_emb, M.repeat_kv


class CodePredictorFused(nn.Module):
    """Fused whole-frame sub-talker: unrolled G-1 step greedy/sample decode in one graph."""

    def __init__(self, cp, top_k, mode="mixed"):
        super().__init__()
        if mode not in ("mixed", "sample", "greedy"):
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.mtp = cp.small_to_mtp_projection
        self.model = cp.model
        self.rope, self.repeat_kv = _ops()
        c = cp.config
        self.nh, self.nkv, self.D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.kv_groups = self.nh // self.nkv
        self.L = c.num_hidden_layers
        self.n_steps = len(cp.lm_head)                  # G-1 = 15
        self.top_k = int(top_k)
        # stacked output heads [G-1, Vcp, Hcp] and input embeddings [G-1, Vcp, Hin]
        self.register_buffer("head_w", torch.stack([h.weight.detach() for h in cp.lm_head], 0), persistent=False)
        self.register_buffer("cp_emb", torch.stack([e.weight.detach() for e in cp.model.codec_embedding], 0), persistent=False)
        # baked 2-token prefill causal mask [[0,-inf],[0,0]] (fp32). Decode steps (S=1) need no mask
        # (the single new token attends to every key), so only gs=0 uses this constant.
        neg = torch.finfo(torch.float32).min
        pm = torch.zeros(1, 1, 2, 2, dtype=torch.float32)
        pm[0, 0, 0, 1] = neg
        self.register_buffer("prefill_mask", pm, persistent=False)

    def _decode(self, cur, position_ids, pk_list, pv_list, mask):
        """5-layer decode (mtp -> rope -> layers -> norm). Identical math to the old per-step graph
        (greedy stays byte-exact). KV is carried as per-layer Python lists across the unrolled steps
        (no stack/Slice churn). <paramref name="mask"/> is the fp32 prefill mask for gs=0 or None for
        decode steps (no masking). Returns (hidden, new_pk_list, new_pv_list)."""
        x = self.mtp(cur)
        m = self.model
        cos, sin = m.rotary_emb(x, position_ids)
        B = x.shape[0]
        h = x
        npk, npv = [], []
        for i, layer in enumerate(m.layers):
            attn = layer.self_attn
            res = h
            xx = layer.input_layernorm(h)
            q = attn.q_norm(attn.q_proj(xx).reshape(B, -1, self.nh, self.D)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(xx).reshape(B, -1, self.nkv, self.D)).transpose(1, 2)
            v = attn.v_proj(xx).reshape(B, -1, self.nkv, self.D).transpose(1, 2)
            q, k = self.rope(q, k, cos, sin)
            if pk_list is not None:
                k = torch.cat([pk_list[i], k], dim=2)
                v = torch.cat([pv_list[i], v], dim=2)
            npk.append(k)
            npv.append(v)
            kk = self.repeat_kv(k, self.kv_groups)
            vv = self.repeat_kv(v, self.kv_groups)
            qf = q.float() if q.dtype == torch.float16 else q
            kf = kk.float() if q.dtype == torch.float16 else kk
            scores = torch.matmul(qf, kf.transpose(2, 3)) * attn.scaling
            if mask is not None:
                scores = scores + mask.to(device=scores.device, dtype=scores.dtype)
            w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(vv.dtype)
            o = torch.matmul(w, vv).transpose(1, 2).reshape(B, -1, self.nh * self.D)
            h = res + attn.o_proj(o)
            res = h
            h = res + layer.mlp(layer.post_attention_layernorm(h))
        return m.norm(h), npk, npv

    def _onehot(self, logits, do_sample_bool, temperature, u):
        """Select one token as a ONE-HOT row [1,1,Vcp] (greedy argmax when do_sample=0, else
        temp->top-k->softmax->inverse-CDF). Returning a one-hot lets the per-step embedding lookup
        be a matmul (one_hot @ cp_emb) -- pure-GPU, no int index. The max/kth thresholds are read
        from the TopK output (NOT a separate ReduceMax, which ORT pins to CPU -> per-step memcpy).
        Mathematically equals argmax/sample + Gather, and is hardened to ALWAYS be a true one-hot:
          * greedy ties -> FIRST index (matches torch/np argmax and fine_codes=argmax), not multi-hot;
          * sample u > cdf[-1] (fp32 cdf can fall a few ULP short of 1.0) -> last in-support token,
            not an all-zero row that would emit token 0 + feed a zero embedding forward;
          * temperature clamped > 0 so a 0 temperature can't produce inf/nan."""
        lf = logits.reshape(-1).float()                          # [Vcp] fp32
        if self.mode == "greedy":
            topv, _ = torch.topk(lf, 1)                           # [1] max threshold, CUDA-friendly
            return (lf >= topv[0:1]).float().view(1, 1, -1)

        t = torch.clamp(temperature, min=1e-6)                   # guard temp=0 -> no inf/nan
        ls = lf / t
        topv, _ = torch.topk(ls, self.top_k)                     # [top_k] sorted desc; topv[0]=max
        kth = topv[self.top_k - 1:self.top_k]                    # [1] k-th largest (top-k threshold)
        neg = torch.finfo(torch.float32).min
        # sample: clamp u to cdf[-1] (slice, stays on GPU; .max() would ReduceMax->CPU memcpy) so
        # u > cdf[-1] still selects the last supported token instead of an all-zero row.
        masked = torch.where(ls < kth, torch.full_like(ls, neg), ls)
        probs = torch.softmax(masked, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        u_eff = torch.minimum(u.reshape(1), cdf[-1:])            # [1]
        ge = (cdf >= u_eff).float()                              # 0..0,1..1 step at the selection
        ge_prev = torch.cat([ge.new_zeros(1), ge[:-1]], dim=0)
        oh_sample = ge - ge_prev                                 # one-hot at the first cdf>=u index
        if self.mode == "sample":
            return oh_sample.view(1, 1, -1)

        # mixed legacy graph: keep both branches in one ONNX for old runtimes/packages.
        mx = topv[0:1]                                            # [1] max  (argmax threshold)
        oh_greedy = (ls >= mx).float()
        oh = torch.where(do_sample_bool, oh_sample, oh_greedy)
        return oh.view(1, 1, -1)                                  # [1,1,Vcp]

    def forward(self, past_hidden, emb0, do_sample, temperature, randoms):
        do_sample_bool = do_sample > 0
        Hin = emb0.shape[-1]
        dev = emb0.device
        if self.mode == "sample":
            keep_inputs = do_sample.float() * 0.0
        elif self.mode == "greedy":
            keep_inputs = do_sample.float() * 0.0 + temperature.reshape(()) * 0.0 + randoms.sum() * 0.0
        else:
            keep_inputs = None
        cur = torch.cat([past_hidden, emb0], dim=1)              # [1,2,Hin] prefill
        pk_list = pv_list = None                                # KV carried as per-layer lists
        pos_start, S = 0, 2
        onehots = []
        sum_emb = emb0                                          # next_sum = emb0 + Σ cp_emb
        for gs in range(self.n_steps):
            pos = torch.arange(pos_start, pos_start + S, dtype=torch.long, device=dev).view(1, S)
            mask = self.prefill_mask if gs == 0 else None
            h, pk_list, pv_list = self._decode(cur, pos, pk_list, pv_list, mask)
            logits = torch.matmul(h[:, -1:], self.head_w[gs].transpose(0, 1))   # [1,1,Vcp]
            if keep_inputs is not None:
                logits = logits + keep_inputs.to(dtype=logits.dtype)
            oh = self._onehot(logits, do_sample_bool, temperature, randoms[gs])  # [1,1,Vcp]
            onehots.append(oh)
            emb_gs = torch.matmul(oh.to(self.cp_emb.dtype), self.cp_emb[gs])      # [1,1,Hin]
            sum_emb = sum_emb + emb_gs
            if gs < self.n_steps - 1:
                cur = emb_gs
                pos_start += S
                S = 1
        # token ids only materialized once, at the end (one small device->host copy of [G-1]).
        fine_codes = torch.argmax(torch.cat(onehots, 0), dim=-1).reshape(-1).to(torch.int64)   # [G-1]
        return fine_codes, sum_emb


def _ref_greedy(cp, codec_emb, past_hidden, group0, n_steps, Hin):
    """Reference whole-frame greedy via the ORIGINAL HF cp.model + lm_heads (autoregressive),
    mirroring _engine._cp_frame. Returns (codes[list], next_sum[1,1,Hin])."""
    mtp = cp.small_to_mtp_projection
    emb0 = codec_emb[group0].view(1, 1, Hin)
    cur = torch.cat([past_hidden, emb0], dim=1)
    past = None
    codes = []
    sum_emb = emb0.clone()
    pos_start, S = 0, 2
    with torch.inference_mode():
        for gs in range(n_steps):
            pos = torch.arange(pos_start, pos_start + S, dtype=torch.long).view(1, S)
            out = cp.model(inputs_embeds=mtp(cur), past_key_values=past, use_cache=True, position_ids=pos)
            past = out.past_key_values
            h = out.last_hidden_state[:, -1:]
            tok = int(cp.lm_head[gs](h)[0, -1].argmax())
            codes.append(tok)
            emb_gs = cp.model.codec_embedding[gs].weight[tok].view(1, 1, Hin)
            sum_emb = sum_emb + emb_gs
            if gs < n_steps - 1:
                cur = emb_gs
                pos_start += S
                S = 1
    return codes, sum_emb


def run(tts=None):
    if tts is None:
        print("== loading model (fp32) for code_predictor ...")
        tts = C.load_tts(device="cpu", dtype="float32")
    cp = tts.model.talker.code_predictor.eval()
    codec_emb = tts.model.talker.model.codec_embedding.weight.detach()    # [Vc, Hin] for emb0
    cfg = C.load_config()["talker_config"]
    Hin = int(cfg["hidden_size"])

    gen_path = os.path.join(C.MODEL, "generation_config.json")
    gen = json.load(open(gen_path, encoding="utf-8")) if os.path.isfile(gen_path) else {}
    top_k = int(gen.get("subtalker_top_k", 50))
    temp_default = float(gen.get("subtalker_temperature", 0.9))
    print(f"   baked top_k={top_k}  (runtime temperature input, default {temp_default})")

    step = CodePredictorFused(cp, top_k, mode="mixed").eval()
    greedy_step = CodePredictorFused(cp, top_k, mode="greedy").eval()
    sample_step = CodePredictorFused(cp, top_k, mode="sample").eval()
    n_steps = step.n_steps
    g = torch.Generator().manual_seed(0)

    # ---- validate fused GREEDY codes + next_sum vs the original HF cp (autoregressive) ----
    past_hidden = torch.randn(1, 1, Hin, generator=g) * 0.1
    group0 = 123
    emb0 = codec_emb[group0].view(1, 1, Hin)
    do_sample0 = torch.tensor(0, dtype=torch.long)
    temp = torch.tensor(temp_default, dtype=torch.float32)
    rnd = torch.rand(n_steps, generator=g)
    with torch.inference_mode():
        fcodes, nsum = step(past_hidden, emb0, do_sample0, temp, rnd)
    ref_codes, ref_sum = _ref_greedy(cp, codec_emb, past_hidden, group0, n_steps, Hin)
    same = [int(a) for a in fcodes.tolist()] == ref_codes
    print(f"   fused greedy codes == HF greedy: {same}  (fused={fcodes.tolist()[:6]}... ref={ref_codes[:6]}...)")
    if not same:
        raise SystemExit("FUSED GREEDY MISMATCH vs HF cp -- aborting export.")
    C.compare("cp_fused_next_sum_vs_ref", ref_sum, nsum, atol_warn=1e-4)

    with torch.inference_mode():
        gcodes, gsum = greedy_step(past_hidden, emb0, do_sample0, temp, rnd)
    greedy_same = [int(a) for a in gcodes.tolist()] == ref_codes
    print(f"   greedy-only codes == HF greedy: {greedy_same}  (greedy={gcodes.tolist()[:6]}...)")
    if not greedy_same:
        raise SystemExit("GREEDY-ONLY MISMATCH vs HF cp -- aborting export.")
    C.compare("cp_greedy_next_sum_vs_ref", ref_sum, gsum, atol_warn=1e-4)

    do_sample1 = torch.tensor(1, dtype=torch.long)
    with torch.inference_mode():
        mixed_sample_codes, mixed_sample_sum = step(past_hidden, emb0, do_sample1, temp, rnd)
        sample_codes, sample_sum = sample_step(past_hidden, emb0, do_sample1, temp, rnd)
    sample_same = mixed_sample_codes.tolist() == sample_codes.tolist()
    print(f"   sample-only codes == mixed sample: {sample_same}  (sample={sample_codes.tolist()[:6]}...)")
    if not sample_same:
        raise SystemExit("SAMPLE-ONLY MISMATCH vs mixed sample -- aborting export.")
    C.compare("cp_sample_next_sum_vs_mixed", mixed_sample_sum, sample_sum, atol_warn=1e-4)

    # ---- export ----
    ex = (past_hidden, emb0, do_sample0, temp, rnd)
    export_modules = [
        ("code_predictor.onnx", step),
        ("code_predictor_sample.onnx", sample_step),
        ("code_predictor_greedy.onnx", greedy_step),
    ]
    if C.DTYPE in ("fp16", "bf16"):   # native reduced-precision trace on GPU
        dt = torch.float16 if C.DTYPE == "fp16" else torch.bfloat16
        cp_typed = cp.to(dt).to(C.DEVICE)
        export_modules = [
            ("code_predictor.onnx", CodePredictorFused(cp_typed, top_k, mode="mixed").eval()),
            ("code_predictor_sample.onnx", CodePredictorFused(cp_typed, top_k, mode="sample").eval()),
            ("code_predictor_greedy.onnx", CodePredictorFused(cp_typed, top_k, mode="greedy").eval()),
        ]
        ex = (past_hidden.to(dt).to(C.DEVICE), emb0.to(dt).to(C.DEVICE),
              do_sample0.to(C.DEVICE), temp.to(C.DEVICE), rnd.to(C.DEVICE))

    written = []
    for filename, mod in export_modules:
        path = os.path.join(C.PKG_DIR, filename)
        print("== exporting", path)
        C.export_onnx(
            mod, ex, path,
            input_names=["past_hidden", "emb0", "do_sample", "temperature", "randoms"],
            output_names=["fine_codes", "next_sum"],
            dynamic_axes={},   # fully static -> CUDA-graph friendly
        )
        if C.DTYPE in ("int8", "int4"):
            C.postprocess_dtype(path, C.DTYPE)
        written.append(path)
        print(f"   wrote {path} ({os.path.getsize(path)/1e6:.0f} MB, dtype={C.DTYPE})")

    # ---- ORT greedy parity (fp32): exported graph must match torch fused greedy exactly ----
    if C.DTYPE == "fp32":
        path = os.path.join(C.PKG_DIR, "code_predictor_greedy.onnx")
        s = C.ort_session(path)
        fo, so_ = s.run(["fine_codes", "next_sum"], {
            "past_hidden": past_hidden.numpy().astype(np.float32),
            "emb0": emb0.numpy().astype(np.float32),
            "do_sample": np.array(0, dtype=np.int64),
            "temperature": np.array(temp_default, dtype=np.float32),
            "randoms": rnd.numpy().astype(np.float32)})
        ort_same = fo.tolist() == ref_codes
        print(f"   ORT fused greedy codes == HF greedy: {ort_same}  (ort={fo.tolist()[:6]}...)")
        if not ort_same:
            raise SystemExit("ORT GREEDY-ONLY MISMATCH -- aborting export.")
        C.compare("cp_greedy_onnx_next_sum_vs_torch", gsum.float().numpy(), so_)

        path = os.path.join(C.PKG_DIR, "code_predictor_sample.onnx")
        s = C.ort_session(path)
        fo, so_ = s.run(["fine_codes", "next_sum"], {
            "past_hidden": past_hidden.numpy().astype(np.float32),
            "emb0": emb0.numpy().astype(np.float32),
            "do_sample": np.array(1, dtype=np.int64),
            "temperature": np.array(temp_default, dtype=np.float32),
            "randoms": rnd.numpy().astype(np.float32)})
        ort_sample_same = fo.tolist() == sample_codes.tolist()
        print(f"   ORT sample-only codes == torch sample-only: {ort_sample_same}  (ort={fo.tolist()[:6]}...)")
        if not ort_sample_same:
            raise SystemExit("ORT SAMPLE-ONLY MISMATCH -- aborting export.")
        C.compare("cp_sample_onnx_next_sum_vs_torch", sample_sum.float().numpy(), so_)
    print("code_predictor (sample/greedy fused) export done.")


def main():
    ap = argparse.ArgumentParser(description="Export Qwen3-TTS fused code_predictor to ONNX.")
    C.add_io_args(ap)
    args = ap.parse_args()
    C.apply_io_args(args)
    run()
    C.gate_and_exit()


if __name__ == "__main__":
    main()
