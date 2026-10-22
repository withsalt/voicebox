#!/usr/bin/env python3
"""Validate the streaming ONNX against the offline PyTorch model.

Two checks:
  (1) Export correctness: per-frame streaming ONNX outputs (m, lsnr, coefs) vs the
      offline DfNet whole-sequence outputs, on identical libdf features. Proves GRU
      state propagation + conv causality are reproduced exactly.
  (2) End-to-end: reconstruct enhanced audio from the *streaming* m/coefs (model
      mask + deep filter + libdf synthesis) and compare to the shipped denoised wav.

The streaming driver mirrors the C# runner: a fixed encWindow ring buffer
(zero-initialised), feeding T=encWindow feature frames per call.
"""
from __future__ import annotations

import os
import wave

import numpy as np
import onnxruntime as ort
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = r"E:/Temp/hush/model_best.ckpt"
RAW = r"E:/Temp/hush/assets/audio/sample_00006_raw.wav"
DENOISED = r"E:/Temp/hush/assets/audio/sample_00006_denoised.wav"
ONNX = os.path.join(HERE, "outputs", "combined_streaming.onnx")

from dfnet_se import DfNetSE, as_complex, as_real, get_config, get_norm_alpha
from libdf import DF, erb, erb_norm, unit_norm
from export_streaming import load_model


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0 if sw == 2 \
        else np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), sr


def cmp(a, b, label):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    max_err = float(np.max(np.abs(a - b))) if n else float("nan")
    corr = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))
    print(f"  {label:24s} n={n:7d} max_err={max_err:.3e} corr={corr:.7f}")
    return max_err, corr


@torch.no_grad()
def main():
    model = load_model()
    cfg = model.config
    raw, _ = read_wav(RAW)
    den, _ = read_wav(DENOISED)

    df_state = DF(sr=cfg.sr, fft_size=cfg.fft_size, hop_size=cfg.hop_size,
                  nb_bands=cfg.nb_erb, min_nb_erb_freqs=cfg.min_nb_freqs)
    alpha = get_norm_alpha(df_state.sr(), df_state.hop_size(), cfg.norm_tau)
    n_fft, hop = df_state.fft_size(), df_state.hop_size()

    audio = torch.from_numpy(raw).unsqueeze(0)
    audio = torch.nn.functional.pad(audio, (0, n_fft))
    spec_np = df_state.analysis(audio.numpy(), reset=True)
    erb_w = df_state.erb_widths()
    spec = torch.as_tensor(spec_np)
    erb_feat = torch.as_tensor(erb_norm(erb(spec_np, erb_w), alpha))           # [1,T,nb_erb]
    spec_feat = as_real(torch.as_tensor(unit_norm(spec_np[..., :cfg.nb_df], alpha)))  # [1,T,nb_df,2]

    spec_t = as_real(spec).unsqueeze(1)               # [1,1,T,F,2]
    erb_t = erb_feat.unsqueeze(1)                     # [1,1,T,nb_erb]
    spec_ft = spec_feat.unsqueeze(1)                  # [1,1,T,nb_df,2]

    # ---- offline reference ----
    spec_e_off, m_off, lsnr_off, dfc_off, _ = model.model(spec_t.clone(), erb_t, spec_ft)
    T = erb_feat.shape[1]
    print(f"frames T={T}  fft={n_fft} hop={hop} nb_erb={cfg.nb_erb} nb_df={cfg.nb_df} O={cfg.df_order}")

    m_off_np = m_off[0, 0].numpy()                    # [T,nb_erb]
    lsnr_off_np = lsnr_off[0, :, 0].numpy()           # [T]
    # dfc_off [1,O,T,nb_df,2] -> per-frame [T,nb_df,O*2] (re/im interleaved per order)
    dfc = dfc_off[0].permute(1, 2, 0, 3).reshape(T, cfg.nb_df, cfg.df_order * 2).numpy()

    # ---- streaming ONNX ----
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    H = model.model.enc.emb_gru.gru.hidden_size
    Le = model.model.erb_dec.emb_gru.gru.num_layers
    Ld = model.model.df_dec.df_gru.gru.num_layers
    W = cfg.conv_kernel_inp[0]                        # encWindow

    h0 = np.zeros((1, 1, H), np.float32)
    erb_h0 = np.zeros((Le, 1, H), np.float32)
    df_h0 = np.zeros((Ld, 1, H), np.float32)
    ring_e = np.zeros((W, cfg.nb_erb), np.float32)
    ring_s = np.zeros((W, 2, cfg.nb_df), np.float32)

    erb_np = erb_feat[0].numpy()                      # [T,nb_erb]
    spec_fp = spec_feat[0].permute(0, 2, 1).numpy()   # [T,2,nb_df]

    m_s = np.zeros((T, cfg.nb_erb), np.float32)
    lsnr_s = np.zeros(T, np.float32)
    coefs_s = np.zeros((T, cfg.nb_df, cfg.df_order * 2), np.float32)

    for t in range(T):
        ring_e[:-1] = ring_e[1:]; ring_e[-1] = erb_np[t]
        ring_s[:-1] = ring_s[1:]; ring_s[-1] = spec_fp[t]
        fe = ring_e.reshape(1, 1, W, cfg.nb_erb)
        fs = ring_s.transpose(1, 0, 2).reshape(1, 2, W, cfg.nb_df)
        outs = sess.run(["lsnr", "m", "coefs", "h1", "erb_h1", "df_h1"],
                        {"feat_erb": fe, "feat_spec": fs, "h0": h0, "erb_h0": erb_h0, "df_h0": df_h0})
        lsnr_s[t] = np.asarray(outs[0]).ravel()[0]
        m_s[t] = np.asarray(outs[1]).ravel()[:cfg.nb_erb]
        coefs_s[t] = np.asarray(outs[2]).reshape(cfg.nb_df, cfg.df_order * 2)
        h0 = np.asarray(outs[3], np.float32).reshape(1, 1, H)
        erb_h0 = np.asarray(outs[4], np.float32).reshape(Le, 1, H)
        df_h0 = np.asarray(outs[5], np.float32).reshape(Ld, 1, H)

    print("export correctness (offline vs streaming ONNX):")
    cmp(m_off_np, m_s, "erb_mask m")
    cmp(lsnr_off_np, lsnr_s, "lsnr")
    cmp(dfc, coefs_s, "df coefs")

    # ---- end-to-end reconstruction from STREAMING outputs ----
    from dfnet_se import DfOutputReshapeMF
    m_full = torch.from_numpy(m_s).view(1, 1, T, cfg.nb_erb)
    coefs_full = torch.from_numpy(coefs_s).view(1, T, cfg.nb_df, cfg.df_order * 2)
    dfc_full = DfOutputReshapeMF(cfg.df_order, cfg.nb_df)(coefs_full)
    spec_m = model.model.mask(spec_t.clone(), m_full)
    spec_e = model.model.apply_df(spec_t.clone(), dfc_full)
    spec_e[..., cfg.nb_df:, :] = spec_m[..., cfg.nb_df:, :]
    spec_e_c = as_complex(spec_e.squeeze(1))
    enh = df_state.synthesis(spec_e_c.numpy(), reset=True)
    enh = np.asarray(enh, np.float32).ravel()
    delay = n_fft - hop
    enh = enh[delay:delay + len(raw)]
    print("end-to-end (streaming reconstruction vs shipped denoised.wav):")
    cmp(enh, den, "enhanced audio")


if __name__ == "__main__":
    main()
