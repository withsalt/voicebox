# Copyright (c) 2026
# Export the VoxCPM LM stack (base_lm 24L -> FSQ mix -> residual_lm 6L -> dit/stop heads)
# to ONE ONNX graph serving both prefill (past P=0) and decode (S=1, P>0) -- the unified
# single-graph KV pattern of the qwen3-tts talker (halves package size vs separate graphs).
#
#   lm.onnx
#     in : inputs_embeds[1,S,H], feat_embed_masked[1,S,H], fsq_mask[1,S],
#          attention_mask[1,P+S], position_ids[1,S],
#          base_past_k[24,1,2,P,64], base_past_v[...], res_past_k[6,1,2,P,64], res_past_v[...]
#     out: dit_hidden[1,H], stop_logits[1,2], lm_hidden[1,H], residual_hidden[1,H],
#          base_present_k[24,1,2,P+S,64], base_present_v, res_present_k, res_present_v
#
# Text embedding lookup stays OUTSIDE the graph (C# EmbeddingStore over
# embeddings/text_embedding.bin), so int8 quantization touches only MatMul weights.
import os
import argparse

import numpy as np
import torch

import _common as C
from _modules import LmGqaModule, LmModule, FeatEncoderModule


def _empty_past(L, kv, D):
    return torch.zeros(L, 1, kv, 0, D), torch.zeros(L, 1, kv, 0, D)


def _pos(S, start=0):
    return torch.arange(start, start + S, dtype=torch.long).view(1, S)


def _ref_prefill(vox, text_ids, text_mask, audio_mask, feat):
    """Original-code prefill (SDPA enable_gqa path) -> reference outputs."""
    with torch.inference_mode():
        feat_embed = vox.enc_to_lm_proj(vox.feat_encoder(feat))
        text_embed = vox.base_lm.embed_tokens(text_ids)
        tm, am = text_mask.unsqueeze(-1), audio_mask.unsqueeze(-1)
        combined = tm * text_embed + am * feat_embed
        enc, _ = vox.base_lm(inputs_embeds=combined, is_causal=True)
        enc_mix = vox.fsq_layer(enc) * am + enc * tm
        rin = enc_mix + am * feat_embed
        rh, _ = vox.residual_lm(inputs_embeds=rin, is_causal=True)
        lm_hidden = enc_mix[:, -1, :]
        residual_hidden = rh[:, -1, :]
        dit_hidden = vox.lm_to_dit_proj(lm_hidden) + vox.res_to_dit_proj(residual_hidden)
        stop_logits = vox.stop_head(vox.stop_actn(vox.stop_proj(lm_hidden)))
    return combined, am * feat_embed, dit_hidden, stop_logits, lm_hidden, residual_hidden


def run(vox=None):
    if vox is None:
        print("== loading model (fp32) for lm ...")
        vox = C.load_vox()
    cfg = vox.config.lm_config
    H = cfg.hidden_size
    Lb, Lr = cfg.num_hidden_layers, vox.config.residual_lm_num_layers
    kv, D = cfg.num_key_value_heads, H // cfg.num_attention_heads

    lm = LmModule(vox).eval()
    g = torch.Generator().manual_seed(0)
    T = 12
    Ta = 4  # trailing "audio" positions in the synthetic prefill
    text_ids = torch.randint(3, 1000, (1, T), generator=g)
    text_ids[0, -Ta:] = 0
    text_mask = torch.ones(1, T)
    text_mask[0, -Ta:] = 0.0
    audio_mask = 1.0 - text_mask
    feat = torch.randn(1, T, vox.patch_size, vox.feat_dim, generator=g) * 0.5
    feat = feat * audio_mask.view(1, T, 1, 1)

    combined, fem, r_dit, r_stop, r_lm, r_res = _ref_prefill(vox, text_ids, text_mask, audio_mask, feat)

    bpk0, bpv0 = _empty_past(Lb, kv, D)
    rpk0, rpv0 = _empty_past(Lr, kv, D)
    am_pre = torch.ones(1, T, dtype=torch.long)
    with torch.inference_mode():
        m_dit, m_stop, m_lm, m_res, bk, bv, rk, rv = lm(
            combined, fem, audio_mask, am_pre, _pos(T), bpk0, bpv0, rpk0, rpv0
        )
    C.compare("lm_prefill_dit_vs_orig", C.to_numpy(r_dit), C.to_numpy(m_dit), atol_warn=1e-4)
    C.compare("lm_prefill_stop_vs_orig", C.to_numpy(r_stop), C.to_numpy(m_stop), atol_warn=1e-4)
    C.compare("lm_prefill_lmh_vs_orig", C.to_numpy(r_lm), C.to_numpy(m_lm), atol_warn=1e-4)
    C.compare("lm_prefill_resh_vs_orig", C.to_numpy(r_res), C.to_numpy(m_res), atol_warn=1e-4)
    print(f"   base_present_k={tuple(bk.shape)} (expect [{Lb},1,{kv},{T},{D}])")

    # KV-cache decode: prefill(T) + step(1) must equal full prefill(T+1)[last].
    # A decode step is exactly a length-1 "audio" suffix (fsq_mask=1, embeds=curr_embed).
    curr = torch.randn(1, 1, H, generator=g) * 0.5
    one = torch.ones(1, 1)
    with torch.inference_mode():
        d_dit, d_stop, d_lm, d_res, *_ = lm(
            curr, curr, one, torch.ones(1, T + 1, dtype=torch.long), _pos(1, T), bk, bv, rk, rv
        )
        f_dit, f_stop, f_lm, f_res, *_ = lm(
            torch.cat([combined, curr], 1),
            torch.cat([fem, curr], 1),
            torch.cat([audio_mask, one], 1),
            torch.ones(1, T + 1, dtype=torch.long),
            _pos(T + 1),
            bpk0, bpv0, rpk0, rpv0,
        )
    C.compare("lm_decode_kvcache_vs_full_dit", C.to_numpy(f_dit), C.to_numpy(d_dit), atol_warn=1e-4)
    C.compare("lm_decode_kvcache_vs_full_stop", C.to_numpy(f_stop), C.to_numpy(d_stop), atol_warn=1e-4)

    # export single graph; trace with S=T and a NON-empty past so the concat traces
    # cleanly; dynamic axes allow P=0 (prefill) and S=1 (decode) at runtime.
    Ptr = 4
    bpkt = torch.randn(Lb, 1, kv, Ptr, D, generator=g) * 0.1
    bpvt = torch.randn(Lb, 1, kv, Ptr, D, generator=g) * 0.1
    rpkt = torch.randn(Lr, 1, kv, Ptr, D, generator=g) * 0.1
    rpvt = torch.randn(Lr, 1, kv, Ptr, D, generator=g) * 0.1
    am_t = torch.ones(1, T + Ptr, dtype=torch.long)
    path = os.path.join(C.PKG_DIR, "lm.onnx")
    print("== exporting", path)
    mod = lm
    ex = (combined, fem, audio_mask, am_t, _pos(T, Ptr), bpkt, bpvt, rpkt, rpvt)
    if C.DTYPE == "fp16":  # native half trace (mutates the shared modules in place)
        mod = lm.half().eval()
        ex = tuple(t.half() if t.is_floating_point() else t for t in ex)
    C.export_onnx(
        mod,
        ex,
        path,
        input_names=[
            "inputs_embeds", "feat_embed_masked", "fsq_mask", "attention_mask", "position_ids",
            "base_past_k", "base_past_v", "res_past_k", "res_past_v",
        ],
        output_names=[
            "dit_hidden", "stop_logits", "lm_hidden", "residual_hidden",
            "base_present_k", "base_present_v", "res_present_k", "res_present_v",
        ],
        dynamic_axes={
            "inputs_embeds": {1: "S"}, "feat_embed_masked": {1: "S"}, "fsq_mask": {1: "S"},
            "attention_mask": {1: "TP"}, "position_ids": {1: "S"},
            "base_past_k": {3: "P"}, "base_past_v": {3: "P"},
            "res_past_k": {3: "P"}, "res_past_v": {3: "P"},
            "base_present_k": {3: "TP"}, "base_present_v": {3: "TP"},
            "res_present_k": {3: "TP"}, "res_present_v": {3: "TP"},
        },
    )
    if C.DTYPE == "int8":  # fp16 was traced natively above; fp32 is a no-op
        C.postprocess_dtype(path, "int8")
    size = os.path.getsize(path) + (
        os.path.getsize(path + ".data") if os.path.isfile(path + ".data") else 0
    )
    print(f"   wrote {path} ({size/1e6:.0f} MB, dtype={C.DTYPE})")
    if C.DTYPE == "int8":
        # 量化质量门禁: int8-acc4 实测 dit ~31dB / stop ~63dB (int4 曾为 18.7/36dB → 弃用)
        s8 = C.ort_session(path)
        f32 = lambda t: C.to_numpy(t).astype(np.float32)
        po8 = s8.run(None, {
            "inputs_embeds": f32(combined), "feat_embed_masked": f32(fem),
            "fsq_mask": f32(audio_mask), "attention_mask": am_pre.numpy(),
            "position_ids": _pos(T).numpy(),
            "base_past_k": f32(bpk0), "base_past_v": f32(bpv0),
            "res_past_k": f32(rpk0), "res_past_v": f32(rpv0),
        })
        C.compare("lm_int8_prefill_dit_vs_fp32", C.to_numpy(m_dit), po8[0],
                  atol_warn=1e-1, min_snr=22.0, min_cos=0.99)
        do8 = s8.run(None, {
            "inputs_embeds": f32(curr), "feat_embed_masked": f32(curr),
            "fsq_mask": one.numpy(), "attention_mask": torch.ones(1, T + 1, dtype=torch.long).numpy(),
            "position_ids": _pos(1, T).numpy(),
            "base_past_k": po8[4], "base_past_v": po8[5], "res_past_k": po8[6], "res_past_v": po8[7],
        })
        C.compare("lm_int8_decode_dit_vs_fp32", C.to_numpy(d_dit), do8[0],
                  atol_warn=1e-1, min_snr=22.0, min_cos=0.99)
        C.compare("lm_int8_decode_stop_vs_fp32", C.to_numpy(d_stop), do8[1],
                  atol_warn=1e-1, min_snr=30.0, min_cos=0.99)
        del s8

    # ORT parity at fp32: prefill (P=0) and decode (P=T)
    if C.DTYPE == "fp32":
        s = C.ort_session(path)
        f32 = lambda t: C.to_numpy(t).astype(np.float32)
        po = s.run(None, {
            "inputs_embeds": f32(combined), "feat_embed_masked": f32(fem),
            "fsq_mask": f32(audio_mask), "attention_mask": am_pre.numpy(),
            "position_ids": _pos(T).numpy(),
            "base_past_k": f32(bpk0), "base_past_v": f32(bpv0),
            "res_past_k": f32(rpk0), "res_past_v": f32(rpv0),
        })
        C.compare("lm_prefill_onnx_vs_torch_dit", C.to_numpy(m_dit), po[0])
        C.compare("lm_prefill_onnx_vs_torch_stop", C.to_numpy(m_stop), po[1])
        do = s.run(None, {
            "inputs_embeds": f32(curr), "feat_embed_masked": f32(curr),
            "fsq_mask": one.numpy(), "attention_mask": torch.ones(1, T + 1, dtype=torch.long).numpy(),
            "position_ids": _pos(1, T).numpy(),
            "base_past_k": po[4], "base_past_v": po[5], "res_past_k": po[6], "res_past_v": po[7],
        })
        C.compare("lm_decode_onnx_vs_torch_dit", C.to_numpy(d_dit), do[0])
        C.compare("lm_decode_onnx_vs_torch_stop", C.to_numpy(d_stop), do[1])
    print("lm export done.")


def run_gqa(vox=None):
    """lm_gqa.onnx: GroupQueryAttention contrib-op 的 CPU 加速变体。

    与 lm.onnx 同权重同数学 (非比特一致: 融合 kernel 归约顺序不同); KV 逐层 IO,
    消除堆叠 KV 的 Split/Concat 每步全量拷贝与注意力 Transpose 物化
    (profiled: P=100 时 ~5.5ms/步 → GQA ~2ms; P=550 时 ~7.5ms → ~2ms)。
    导出后与 lm.onnx 做 initializer 去重 (共享 lm.onnx.data), 包体积仅 +几MB。
    fp16 包跳过 (保标准图); int8 包先量化再去重。"""
    if C.DTYPE == "fp16":
        print("fp16 包保留标准图, 跳过 lm_gqa。")
        return
    if vox is None:
        print("== loading model (fp32) for lm_gqa ...")
        vox = C.load_vox()
    cfg = vox.config.lm_config
    H = cfg.hidden_size
    Lb, Lr = cfg.num_hidden_layers, vox.config.residual_lm_num_layers
    kv, D = cfg.num_key_value_heads, H // cfg.num_attention_heads

    std = LmModule(vox).eval()
    mod = LmGqaModule(vox).eval()

    # ---- 与标准模块同一合成前缀对拍 (prefill P=0) ----
    g = torch.Generator().manual_seed(0)
    T = 12
    Ta = 4
    text_ids = torch.randint(3, 1000, (1, T), generator=g)
    text_ids[0, -Ta:] = 0
    text_mask = torch.ones(1, T)
    text_mask[0, -Ta:] = 0.0
    audio_mask = 1.0 - text_mask
    feat = torch.randn(1, T, vox.patch_size, vox.feat_dim, generator=g) * 0.5
    feat = feat * audio_mask.view(1, T, 1, 1)
    combined, fem, _, _, _, _ = _ref_prefill(vox, text_ids, text_mask, audio_mask, feat)

    def empties(L):
        return tuple(torch.zeros(1, kv, 0, D) for _ in range(L))

    def seql(total):
        return torch.tensor([total - 1], dtype=torch.int32), torch.tensor(total, dtype=torch.int32)

    bpk0, bpv0 = _empty_past(Lb, kv, D)
    rpk0, rpv0 = _empty_past(Lr, kv, D)
    am_pre = torch.ones(1, T, dtype=torch.long)
    sk, tot = seql(T)
    with torch.inference_mode():
        s_out = std(combined, fem, audio_mask, am_pre, _pos(T), bpk0, bpv0, rpk0, rpv0)
        g_out = mod(combined, fem, audio_mask, sk, tot, empties(Lb), empties(Lb), empties(Lr), empties(Lr))
    C.compare("lm_gqa_prefill_dit_vs_std", C.to_numpy(s_out[0]), C.to_numpy(g_out[0]), atol_warn=1e-4)
    C.compare("lm_gqa_prefill_stop_vs_std", C.to_numpy(s_out[1]), C.to_numpy(g_out[1]), atol_warn=1e-4)
    C.compare("lm_gqa_prefill_lmh_vs_std", C.to_numpy(s_out[2]), C.to_numpy(g_out[2]), atol_warn=1e-4)
    # present 对拍: std 堆叠 [L,1,kv,T,D] vs gqa 逐层
    C.compare("lm_gqa_prefill_bk_vs_std", C.to_numpy(s_out[4]),
              C.to_numpy(torch.stack(list(g_out[4:4 + Lb]), 0)), atol_warn=1e-4)

    # ---- decode 一步 (KV cache) 对拍 ----
    curr = torch.randn(1, 1, H, generator=g) * 0.5
    one = torch.ones(1, 1)
    sk1, tot1 = seql(T + 1)
    with torch.inference_mode():
        s_dec = std(curr, curr, one, torch.ones(1, T + 1, dtype=torch.long), _pos(1, T),
                    s_out[4], s_out[5], s_out[6], s_out[7])
        g_dec = mod(curr, curr, one, sk1, tot1,
                    tuple(g_out[4:4 + Lb]), tuple(g_out[4 + Lb:4 + 2 * Lb]),
                    tuple(g_out[4 + 2 * Lb:4 + 2 * Lb + Lr]), tuple(g_out[4 + 2 * Lb + Lr:]))
    C.compare("lm_gqa_decode_dit_vs_std", C.to_numpy(s_dec[0]), C.to_numpy(g_dec[0]), atol_warn=1e-4)
    C.compare("lm_gqa_decode_stop_vs_std", C.to_numpy(s_dec[1]), C.to_numpy(g_dec[1]), atol_warn=1e-4)

    # ---- 导出 (非空 past trace; 动态轴逐层生成) ----
    Ptr = 4
    bkt = tuple(torch.randn(1, kv, Ptr, D, generator=g) * 0.1 for _ in range(Lb))
    bvt = tuple(torch.randn(1, kv, Ptr, D, generator=g) * 0.1 for _ in range(Lb))
    rkt = tuple(torch.randn(1, kv, Ptr, D, generator=g) * 0.1 for _ in range(Lr))
    rvt = tuple(torch.randn(1, kv, Ptr, D, generator=g) * 0.1 for _ in range(Lr))
    skt, tott = seql(T + Ptr)

    past_in = [f"base_past_k_{i}" for i in range(Lb)] + [f"base_past_v_{i}" for i in range(Lb)] + \
              [f"res_past_k_{i}" for i in range(Lr)] + [f"res_past_v_{i}" for i in range(Lr)]
    pres_out = [n.replace("past", "present") for n in past_in]
    input_names = ["inputs_embeds", "feat_embed_masked", "fsq_mask",
                   "seqlens_k", "total_sequence_length", *past_in]
    output_names = ["dit_hidden", "stop_logits", "lm_hidden", "residual_hidden", *pres_out]
    dyn = {"inputs_embeds": {1: "S"}, "feat_embed_masked": {1: "S"}, "fsq_mask": {1: "S"}}
    for n in past_in:
        dyn[n] = {2: "P"}
    for n in pres_out:
        dyn[n] = {2: "T"}

    path = os.path.join(C.PKG_DIR, "lm_gqa.onnx")
    print("== exporting", path)
    C.export_onnx(
        mod,
        (combined, fem, audio_mask, skt, tott, bkt, bvt, rkt, rvt),
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dyn,
    )
    if C.DTYPE == "int8":
        C.postprocess_dtype(path, "int8")

    # ---- ORT 对拍 (prefill P=0 + decode P=T), 仅 fp32 严格门控 ----
    if C.DTYPE == "fp32":
        s = C.ort_session(path)
        f32 = lambda t: C.to_numpy(t).astype(np.float32)
        feed = {
            "inputs_embeds": f32(combined), "feat_embed_masked": f32(fem),
            "fsq_mask": f32(audio_mask),
            "seqlens_k": np.array([T - 1], dtype=np.int32),
            "total_sequence_length": np.array(T, dtype=np.int32),
        }
        for i in range(Lb):
            feed[f"base_past_k_{i}"] = np.zeros((1, kv, 0, D), np.float32)
            feed[f"base_past_v_{i}"] = np.zeros((1, kv, 0, D), np.float32)
        for i in range(Lr):
            feed[f"res_past_k_{i}"] = np.zeros((1, kv, 0, D), np.float32)
            feed[f"res_past_v_{i}"] = np.zeros((1, kv, 0, D), np.float32)
        po = s.run(None, feed)
        C.compare("lm_gqa_prefill_onnx_vs_torch_dit", C.to_numpy(g_out[0]), po[0])
        C.compare("lm_gqa_prefill_onnx_vs_torch_stop", C.to_numpy(g_out[1]), po[1])

        feed2 = {
            "inputs_embeds": f32(curr), "feat_embed_masked": f32(curr),
            "fsq_mask": one.numpy().astype(np.float32),
            "seqlens_k": np.array([T], dtype=np.int32),
            "total_sequence_length": np.array(T + 1, dtype=np.int32),
        }
        for j, n in enumerate(past_in):
            feed2[n] = po[4 + j]
        do = s.run(None, feed2)
        C.compare("lm_gqa_decode_onnx_vs_torch_dit", C.to_numpy(g_dec[0]), do[0])
        C.compare("lm_gqa_decode_onnx_vs_torch_stop", C.to_numpy(g_dec[1]), do[1])
        del s

    # ---- 与 lm.onnx 共享权重 (包体积仅 +几MB) ----
    C.dedup_initializers(os.path.join(C.PKG_DIR, "lm.onnx"), path)
    size = os.path.getsize(path)
    if size > 50e6:   # 去重失配 (如量化字节不一致) 会静默留下整份权重
        raise RuntimeError(f"lm_gqa 去重后仍有 {size / 1e6:.0f}MB, 与 lm.onnx 权重未对上。")
    print(f"   wrote {path} ({size / 1e6:.1f} MB after dedup, dtype={C.DTYPE})")
    print("lm_gqa export done.")


def run_feat_encoder(vox=None):
    """feat_encoder.onnx: feat[1,T,Pz,Dz] -> feat_embed[1,T,H] (dynamic T)."""
    if vox is None:
        vox = C.load_vox()
    fe = FeatEncoderModule(vox).eval()
    g = torch.Generator().manual_seed(1)
    T = 5
    feat = torch.randn(1, T, vox.patch_size, vox.feat_dim, generator=g) * 0.5
    with torch.inference_mode():
        ref = vox.enc_to_lm_proj(vox.feat_encoder(feat))
        mine = fe(feat)
    C.compare("feat_encoder_vs_orig", C.to_numpy(ref), C.to_numpy(mine), atol_warn=1e-4)

    path = os.path.join(C.PKG_DIR, "feat_encoder.onnx")
    print("== exporting", path)
    mod, ex = fe, feat
    if C.DTYPE == "fp16":
        mod, ex = fe.half().eval(), feat.half()
    C.export_onnx(
        mod, (ex,), path,
        input_names=["feat"], output_names=["feat_embed"],
        dynamic_axes={"feat": {1: "T"}, "feat_embed": {1: "T"}},
    )
    if C.DTYPE == "int8":
        # int8 包: feat_encoder 权重 int8 (每步全量读 244MB → ~71MB, 4.3→1.2ms/步)
        C.postprocess_dtype(path, "int8")
        s8 = C.ort_session(path)
        o8 = s8.run(None, {"feat": C.to_numpy(feat).astype(np.float32)})[0]
        C.compare("feat_encoder_int8_vs_torch_fp32", C.to_numpy(mine), o8,
                  atol_warn=1e-1, min_snr=25.0, min_cos=0.99)   # acc4 实测 ~33dB
        del s8
    print(f"   wrote {path} ({os.path.getsize(path)/1e6:.0f} MB)")

    if C.DTYPE == "fp32":
        s = C.ort_session(path)
        o = s.run(None, {"feat": C.to_numpy(feat).astype(np.float32)})[0]
        C.compare("feat_encoder_onnx_vs_torch", C.to_numpy(mine), o)
        o1 = s.run(None, {"feat": C.to_numpy(feat[:, :1]).astype(np.float32)})[0]
        C.compare("feat_encoder_onnx_T1", C.to_numpy(mine[:, :1]), o1)
    print("feat_encoder export done.")


def main():
    ap = argparse.ArgumentParser(description="Export VoxCPM lm.onnx (+lm_gqa.onnx) + feat_encoder.onnx.")
    C.add_io_args(ap)
    args = ap.parse_args()
    C.apply_io_args(args)
    run()
    run_gqa()
    run_feat_encoder()
    C.gate_and_exit()


if __name__ == "__main__":
    main()
