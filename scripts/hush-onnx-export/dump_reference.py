#!/usr/bin/env python3
"""Hush (DeepFilterNet3-SE, 16 kHz) streaming reference + parity-vector dump.

Pure numpy port of the libDF DSP + deepfilter-rt combined_streaming / StatefulH0
per-frame orchestration, driving the re-exported `combined_streaming.onnx` frame by
frame. This is the *exact* arithmetic the C# VoiceBox.DNR.Hush mirrors, so its output
can be compared to the C# implementation at tight tolerance.

It also (a) loads the shipped reference denoised wav and reports correlation, closing
the loop numpy-port == libDF == reference; and (b) dumps per-frame intermediate tensors
for staged C# parity tests.

Usage:
    python dump_reference.py                      # uses the shipped 16k raw sample
    python dump_reference.py path/to/16k_mono.wav # custom input

Outputs (outputs/vectors/, little-endian float32):
    meta.json, input.bin, enhanced.bin, feat_erb.bin, feat_spec.bin,
    mask.bin, coefs.bin, lsnr.bin, reference_denoised.bin (if available)
"""
import json
import math
import os
import sys
import wave

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
# Point at the C# project's bundled model so vectors match what C# loads byte-for-byte.
MODEL_DIR = os.path.join(
    HERE, "..", "..", "src", "VoiceBox.DNR.Hush",
    "runtimes", "models", "dnr", "hush", "hush16k",
)
OUT_DIR = os.path.join(HERE, "outputs", "vectors")
RAW = r"E:/Temp/hush/assets/audio/sample_00006_raw.wav"
DENOISED = r"E:/Temp/hush/assets/audio/sample_00006_denoised.wav"

MIN_DB_THRESH = -15.0
MAX_DB_ERB_THRESH = 35.0
MAX_DB_DF_THRESH = 35.0


def parse_ini(path):
    cfg = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line[0] in "[#;":
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


# ----------------------- DSP (port of libDF) -----------------------
def vorbis_window(n):
    half = n / 2.0
    w = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = math.sin(0.5 * math.pi * (i + 0.5) / half)
        w[i] = math.sin(0.5 * math.pi * s * s)
    return w.astype(np.float32)


def freq2erb(f):
    return 9.265 * math.log1p(f / (24.7 * 9.265))


def erb2freq(e):
    return 24.7 * 9.265 * (math.exp(e / 9.265) - 1.0)


def erb_fb(sr, fft, nb_bands, min_nb_freqs):
    nyq = sr // 2
    freq_width = sr / fft
    erb_low = freq2erb(0.0)
    erb_high = freq2erb(nyq)
    step = (erb_high - erb_low) / nb_bands
    erb = [0] * nb_bands
    prev_freq = 0
    freq_over = 0
    for i in range(1, nb_bands + 1):
        f = erb2freq(erb_low + i * step)
        fb = int(math.floor(f / freq_width + 0.5))
        nb_freqs = fb - prev_freq - freq_over
        if nb_freqs < min_nb_freqs:
            freq_over = min_nb_freqs - nb_freqs
            nb_freqs = min_nb_freqs
        else:
            freq_over = 0
        erb[i - 1] = nb_freqs
        prev_freq = fb
    erb[nb_bands - 1] += 1
    too_large = sum(erb) - (fft // 2 + 1)
    if too_large > 0:
        erb[nb_bands - 1] -= too_large
    assert sum(erb) == fft // 2 + 1, (sum(erb), fft // 2 + 1)
    return erb


def norm_alpha(sr, hop, tau):
    dt = hop / sr
    a = math.exp(-dt / tau)
    precision = 3
    rounded = 1.0
    while rounded >= 1.0:
        scale = 10 ** precision
        rounded = math.floor(a * scale + 0.5) / scale
        precision += 1
    return np.float32(rounded)


class DfState:
    def __init__(self, sr, fft, hop, nb_erb, nb_df, min_nb_erb_freqs):
        self.fft = fft
        self.hop = hop
        self.freq = fft // 2 + 1
        self.nb_erb = nb_erb
        self.nb_df = nb_df
        self.window = vorbis_window(fft)
        self.wnorm = np.float32(1.0 / (fft * fft / (2.0 * hop)))
        self.erb = erb_fb(sr, fft, nb_erb, min_nb_erb_freqs)
        self.amem = np.zeros(fft - hop, dtype=np.float32)
        self.smem = np.zeros(fft - hop, dtype=np.float32)
        self.init_norm_states()

    def init_norm_states(self):
        self.mean_state = np.linspace(-60.0, -90.0, self.nb_erb, dtype=np.float32)
        self.unit_state = np.linspace(0.001, 0.0001, self.nb_df, dtype=np.float32)

    def analysis(self, frame):
        split = self.fft - self.hop
        buf = np.empty(self.fft, dtype=np.float32)
        buf[:split] = self.amem * self.window[:split]
        buf[split:] = frame * self.window[split:]
        asplit = len(self.amem) - self.hop
        if asplit > 0:
            self.amem[:asplit] = self.amem[self.hop:]
        self.amem[asplit:] = frame
        spec = np.fft.rfft(buf.astype(np.float64))
        spec = spec * float(self.wnorm)
        return spec.astype(np.complex64)

    def synthesis(self, spec):
        x = np.fft.irfft(spec.astype(np.complex128), n=self.fft) * self.fft
        x = (x.astype(np.float32)) * self.window
        out = np.empty(self.hop, dtype=np.float32)
        out[:] = x[:self.hop] + self.smem[:self.hop]
        split = len(self.smem) - self.hop
        if split > 0:
            self.smem[:split] = self.smem[self.hop:]
        for i in range(split):
            self.smem[i] += x[self.hop + i]
        for i in range(split, len(self.smem)):
            self.smem[i] = x[self.hop + i]
        return out

    def feat_erb(self, spec, alpha):
        out = np.zeros(self.nb_erb, dtype=np.float32)
        bcsum = 0
        for b, bw in enumerate(self.erb):
            k = np.float32(1.0 / bw)
            acc = np.float32(0.0)
            for j in range(bw):
                c = spec[bcsum + j]
                acc += (np.float32(c.real) * np.float32(c.real) + np.float32(c.imag) * np.float32(c.imag)) * k
            out[b] = acc
            bcsum += bw
        for b in range(self.nb_erb):
            out[b] = np.float32(math.log10(out[b] + 1e-10) * 10.0)
        for b in range(self.nb_erb):
            self.mean_state[b] = out[b] * (1.0 - alpha) + self.mean_state[b] * alpha
            out[b] = (out[b] - self.mean_state[b]) / np.float32(40.0)
        return out

    def feat_cplx(self, spec, alpha):
        re = np.zeros(self.nb_df, dtype=np.float32)
        im = np.zeros(self.nb_df, dtype=np.float32)
        for f in range(self.nb_df):
            c = spec[f]
            norm = np.float32(math.sqrt(np.float32(c.real) ** 2 + np.float32(c.imag) ** 2))
            self.unit_state[f] = norm * (1.0 - alpha) + self.unit_state[f] * alpha
            denom = np.float32(math.sqrt(self.unit_state[f]))
            re[f] = np.float32(c.real) / denom
            im[f] = np.float32(c.imag) / denom
        return re, im

    def apply_mask(self, spec, gains):
        bcsum = 0
        for b, bw in enumerate(self.erb):
            g = gains[b]
            for j in range(bw):
                spec[bcsum + j] *= g
            bcsum += bw


def apply_deep_filter(out_spec, x_frames, coefs, nb_df, df_order):
    for freq in range(nb_df):
        re = np.float32(0.0)
        im = np.float32(0.0)
        for o in range(df_order):
            s = x_frames[o][freq]
            cr = np.float32(coefs[freq, o * 2])
            ci = np.float32(coefs[freq, o * 2 + 1])
            sre = np.float32(s.real)
            sim = np.float32(s.imag)
            re += sre * cr - sim * ci
            im += sre * ci + sim * cr
        out_spec[freq] = np.complex64(complex(float(re), float(im)))


def read_wav_mono(path):
    with wave.open(path, "rb") as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
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


def main():
    cfg = parse_ini(os.path.join(MODEL_DIR, "config.ini"))
    sr = int(cfg.get("sr", 16000))
    fft = int(cfg.get("fft_size", 320))
    hop = int(cfg.get("hop_size", 160))
    nb_erb = int(cfg.get("nb_erb", 32))
    nb_df = int(cfg.get("nb_df", 64))
    df_order = int(cfg.get("df_order", 5))
    df_lookahead = int(cfg.get("df_lookahead", 0))
    conv_lookahead = int(cfg.get("conv_lookahead", 0))
    min_nb_erb_freqs = int(cfg.get("min_nb_erb_freqs", 2))
    tau = float(cfg.get("norm_tau", 1))
    enc_kernel_t = int(cfg.get("conv_kernel_inp", "3,3").split(",")[0])
    lookahead = max(df_lookahead, conv_lookahead)
    enc_window = enc_kernel_t
    alpha = norm_alpha(sr, hop, tau)

    src = sys.argv[1] if len(sys.argv) > 1 else RAW
    x, in_sr = read_wav_mono(src)
    if in_sr != sr:
        raise ValueError(f"expected {sr} Hz input, got {in_sr} Hz")
    nframes = len(x) // hop
    x = x[:nframes * hop]

    model_path = os.path.join(MODEL_DIR, "combined_streaming.onnx")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    out_names = ["lsnr", "m", "coefs", "h1", "erb_h1", "df_h1"]
    # GRU state shapes from model metadata.
    shp = {i.name: i.shape for i in sess.get_inputs()}
    H = int(shp["h0"][2]); Le = int(shp["erb_h0"][0]); Ld = int(shp["df_h0"][0])

    df = DfState(sr, fft, hop, nb_erb, nb_df, min_nb_erb_freqs)
    freq = df.freq
    h0 = np.zeros((1, 1, H), dtype=np.float32)
    erb_h0 = np.zeros((Le, 1, H), dtype=np.float32)
    df_h0 = np.zeros((Ld, 1, H), dtype=np.float32)

    enc_erb = np.zeros((enc_window, nb_erb), dtype=np.float32)
    enc_spec = np.zeros((enc_window, nb_df, 2), dtype=np.float32)
    enc_pos = enc_window - 1
    enc_count = enc_window - 1

    bufx = max(df_order, max(lookahead, 1))
    bufy = df_order + lookahead
    xbuf = [np.zeros(freq, dtype=np.complex64) for _ in range(bufx)]
    ybuf = [np.zeros(freq, dtype=np.complex64) for _ in range(bufy)]

    enhanced = np.zeros(nframes * hop, dtype=np.float32)
    feat_erb_all = np.zeros((nframes, nb_erb), dtype=np.float32)
    feat_spec_all = np.zeros((nframes, 2, nb_df), dtype=np.float32)
    mask_all = np.zeros((nframes, nb_erb), dtype=np.float32)
    coefs_all = np.zeros((nframes, nb_df, df_order * 2), dtype=np.float32)
    lsnr_all = np.zeros(nframes, dtype=np.float32)
    zeros_mask = np.zeros(nb_erb, dtype=np.float32)

    for fi in range(nframes):
        frame = x[fi * hop:(fi + 1) * hop]
        spec = df.analysis(frame)
        xbuf.pop(0); xbuf.append(spec.copy())
        ybuf.pop(0); ybuf.append(spec.copy())

        fe = df.feat_erb(spec, alpha)
        fre, fim = df.feat_cplx(spec, alpha)
        feat_erb_all[fi] = fe
        feat_spec_all[fi, 0] = fre
        feat_spec_all[fi, 1] = fim

        slot = enc_pos % enc_window
        enc_erb[slot] = fe
        enc_spec[slot, :, 0] = fre
        enc_spec[slot, :, 1] = fim
        enc_pos += 1
        enc_count = min(enc_count + 1, enc_window)

        t = enc_count
        oldest = (enc_pos - t) % enc_window
        ferb = np.zeros((1, 1, t, nb_erb), dtype=np.float32)
        fspec = np.zeros((1, 2, t, nb_df), dtype=np.float32)
        for ti in range(t):
            s = (oldest + ti) % enc_window
            ferb[0, 0, ti] = enc_erb[s]
            fspec[0, 0, ti] = enc_spec[s, :, 0]
            fspec[0, 1, ti] = enc_spec[s, :, 1]

        outs = sess.run(out_names, {
            "feat_erb": ferb, "feat_spec": fspec,
            "h0": h0, "erb_h0": erb_h0, "df_h0": df_h0,
        })
        lsnr = float(np.asarray(outs[0]).reshape(-1)[0])
        m = np.asarray(outs[1]).reshape(-1)[:nb_erb].astype(np.float32)
        coefs = np.asarray(outs[2]).reshape(-1)[:nb_df * df_order * 2].astype(np.float32).reshape(nb_df, df_order * 2)
        h0 = np.asarray(outs[3]).astype(np.float32).reshape(1, 1, H)
        erb_h0 = np.asarray(outs[4]).astype(np.float32).reshape(Le, 1, H)
        df_h0 = np.asarray(outs[5]).astype(np.float32).reshape(Ld, 1, H)

        lsnr_all[fi] = lsnr
        mask_all[fi] = m
        coefs_all[fi] = coefs

        mask_idx = bufy - 1 - lookahead
        apply_mask = lsnr <= MAX_DB_ERB_THRESH
        apply_zero = lsnr < MIN_DB_THRESH
        if apply_mask:
            df.apply_mask(ybuf[mask_idx], zeros_mask if apply_zero else m)

        out_spec = ybuf[mask_idx].copy()
        if (lsnr <= MAX_DB_DF_THRESH) and (not apply_zero):
            apply_deep_filter(out_spec, xbuf[:df_order], coefs, nb_df, df_order)

        enhanced[fi * hop:(fi + 1) * hop] = df.synthesis(out_spec)

    os.makedirs(OUT_DIR, exist_ok=True)
    x.astype("<f4").tofile(os.path.join(OUT_DIR, "input.bin"))
    enhanced.astype("<f4").tofile(os.path.join(OUT_DIR, "enhanced.bin"))
    feat_erb_all.astype("<f4").tofile(os.path.join(OUT_DIR, "feat_erb.bin"))
    feat_spec_all.astype("<f4").tofile(os.path.join(OUT_DIR, "feat_spec.bin"))
    mask_all.astype("<f4").tofile(os.path.join(OUT_DIR, "mask.bin"))
    coefs_all.astype("<f4").tofile(os.path.join(OUT_DIR, "coefs.bin"))
    lsnr_all.astype("<f4").tofile(os.path.join(OUT_DIR, "lsnr.bin"))

    # Closing-the-loop check vs the shipped reference (no delay compensation here,
    # so compare the time-aligned overlapping region).
    ref_corr = None
    if os.path.exists(DENOISED):
        den, _ = read_wav_mono(DENOISED)
        den.astype("<f4").tofile(os.path.join(OUT_DIR, "reference_denoised.bin"))
        delay = fft - hop
        a = enhanced[delay:].astype(np.float64)
        b = den[:len(a)].astype(np.float64)
        n = min(len(a), len(b)); a, b = a[:n], b[:n]
        ref_corr = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

    meta = {
        "sr": sr, "fft_size": fft, "hop_size": hop, "freq_bins": freq,
        "nb_erb": nb_erb, "nb_df": nb_df, "df_order": df_order,
        "lookahead": lookahead, "enc_window": enc_window,
        "alpha": float(alpha), "frames": nframes, "input_len": int(nframes * hop),
        "erb_widths": df.erb, "model": "hush16k/combined_streaming.onnx",
        "feat_spec_layout": "frame-major [frames,2(re/im),nb_df]",
        "coefs_layout": "frame-major [frames,nb_df,df_order*2]",
        "delay_samples": fft - hop,
        "reference_corr_delay_aligned": ref_corr,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    rms = float(np.sqrt(np.mean(enhanced ** 2)))
    print(f"frames={nframes} input_len={len(x)} enhanced_rms={rms:.5f}")
    print(f"lsnr range=[{lsnr_all.min():.2f}, {lsnr_all.max():.2f}] dB")
    if ref_corr is not None:
        print(f"reference corr (delay-aligned) = {ref_corr:.6f}")
    print(f"vectors -> {OUT_DIR}")


if __name__ == "__main__":
    main()
