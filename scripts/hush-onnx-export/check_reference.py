#!/usr/bin/env python3
"""Ground-truth check: reproduce Hush enhancement with libdf and compare to the
shipped reference denoised sample.

This mirrors hush_src/scripts/infer_single.py exactly (libdf feature extraction +
synthesis), which the upstream docstring claims is bit-for-bit identical to the
released denoised samples. We use it to (a) confirm our understanding of the
authoritative DSP pipeline and (b) produce a gold reference for validating the
streaming ONNX export.

Usage:
    python check_reference.py
"""
from __future__ import annotations

import os
import wave

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = r"E:/Temp/hush/model_best.ckpt"
RAW = r"E:/Temp/hush/assets/audio/sample_00006_raw.wav"
DENOISED = r"E:/Temp/hush/assets/audio/sample_00006_denoised.wav"

from dfnet_se import DfNetSE, as_complex, as_real, get_config, get_norm_alpha
from libdf import DF, erb, erb_norm, unit_norm


def read_wav_mono_f32(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), sr


@torch.no_grad()
def enhance_libdf(model, audio, pad_delay=True):
    """Exact port of infer_single.enhance (atten_lim_db=None)."""
    config = model.config
    df_state = DF(sr=config.sr, fft_size=config.fft_size, hop_size=config.hop_size,
                  nb_bands=config.nb_erb, min_nb_erb_freqs=config.min_nb_freqs)
    orig_len = int(audio.shape[-1])
    n_fft = df_state.fft_size()
    hop = df_state.hop_size()
    if pad_delay:
        audio = torch.nn.functional.pad(audio, (0, n_fft))
    alpha = get_norm_alpha(df_state.sr(), df_state.hop_size(), config.norm_tau)
    spec_np = df_state.analysis(audio.numpy(), reset=True)
    erb_fb = df_state.erb_widths()
    spec = torch.as_tensor(spec_np)
    erb_feat_np = erb_norm(erb(spec_np, erb_fb), alpha)
    spec_feat_np = unit_norm(spec_np[..., : config.nb_df], alpha)
    spec_t = as_real(spec).unsqueeze(1)
    erb_feat_t = torch.as_tensor(erb_feat_np).unsqueeze(1)
    spec_feat_t = as_real(torch.as_tensor(spec_feat_np)).unsqueeze(1)
    spec_enh = model.model(spec_t.clone(), erb_feat_t, spec_feat_t)[0]
    spec_enh_c = as_complex(spec_enh.squeeze(1)).cpu()
    enh_np = df_state.synthesis(spec_enh_c.numpy(), reset=True)
    enh = torch.from_numpy(np.asarray(enh_np, dtype=np.float32))
    if pad_delay:
        delay = n_fft - hop
        enh = enh[:, delay : orig_len + delay]
    return enh


def load_model():
    config = get_config()
    model = DfNetSE(config)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    try:
        model.model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def compare(a, b, label):
    n = min(len(a), len(b))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    max_err = float(np.max(np.abs(a - b))) if n else float("nan")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    corr = float(np.dot(a, b) / denom) if n else float("nan")
    rmse = float(np.sqrt(np.mean((a - b) ** 2))) if n else float("nan")
    print(f"  {label}: n={n} max_err={max_err:.6e} rmse={rmse:.6e} corr={corr:.6f}")
    return max_err, corr


def main():
    print("loading model + checkpoint ...")
    model = load_model()
    cfg = model.config
    print(f"  config: sr={cfg.sr} fft={cfg.fft_size} hop={cfg.hop_size} "
          f"nb_erb={cfg.nb_erb} nb_df={cfg.nb_df} df_order={cfg.df_order}")

    raw, sr_raw = read_wav_mono_f32(RAW)
    den, sr_den = read_wav_mono_f32(DENOISED)
    print(f"  raw: {len(raw)} samp @ {sr_raw} Hz; denoised: {len(den)} samp @ {sr_den} Hz")

    audio = torch.from_numpy(raw).unsqueeze(0)
    enh = enhance_libdf(model, audio, pad_delay=True).squeeze(0).numpy()
    print("comparison (our libdf enhance vs shipped denoised.wav):")
    compare(enh, den, "enhanced_vs_reference")

    # Save reference for later parity use.
    out = os.path.join(HERE, "outputs")
    os.makedirs(out, exist_ok=True)
    raw.astype("<f4").tofile(os.path.join(out, "raw16k.bin"))
    enh.astype("<f4").tofile(os.path.join(out, "enhanced_libdf.bin"))
    den.astype("<f4").tofile(os.path.join(out, "denoised_ref.bin"))
    print(f"saved reference bins -> {out}")


if __name__ == "__main__":
    main()
