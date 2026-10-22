"""
DeepFilterNet (DF3-LL) 流式参考实现 + 对拍向量导出。

仅依赖 numpy + onnxruntime（conda voicebox 已装），把 libDF 的 DSP 与 deepfilter-rt
`lib.rs` 的 combined_streaming / StatefulH0 逐帧编排在 numpy 中精确复刻，驱动
`combined_streaming.onnx`，逐帧输出增强音频，并 dump 中间张量供 C# 单元测试逐级对拍。

用法：
    python dump_reference.py                 # 用确定性合成信号
    python dump_reference.py path/to/48k.wav  # 用指定 wav（自动转 48k 单声道 f32）

输出（outputs/vectors/）：
    meta.json, input.bin, enhanced.bin, feat_erb.bin, feat_spec.bin, mask.bin, coefs.bin, lsnr.bin
所有 .bin 均为 little-endian float32。
"""
import json
import math
import os
import sys
import wave

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(
    HERE, "..", "..", "src", "VoiceBox.DNR.DeepFilterNet",
    "runtimes", "models", "dnr", "deepfilternet", "dfn3_ll",
)
OUT_DIR = os.path.join(HERE, "outputs", "vectors")

MIN_DB_THRESH = -15.0
MAX_DB_ERB_THRESH = 35.0
MAX_DB_DF_THRESH = 35.0


# ----------------------- config -----------------------
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
        fb = int(math.floor(f / freq_width + 0.5))  # round half away from zero (positive)
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
    assert sum(erb) == fft // 2 + 1
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
        spec = np.fft.rfft(buf.astype(np.float64))  # complex128
        spec = spec * float(self.wnorm)
        return spec.astype(np.complex64)

    def synthesis(self, spec):
        # 未归一化逆变换 = numpy irfft × fft
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
    # x_frames: list of df_order complex64 arrays (oldest-first); coefs: [nb_df, df_order*2]
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


# ----------------------- input -----------------------
def load_wav_48k_mono(path, sr_target=48000):
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
    if sr != sr_target:
        idx = np.arange(0, len(x), sr / sr_target)
        i0 = np.floor(idx).astype(int)
        i0 = np.clip(i0, 0, len(x) - 2)
        frac = (idx - i0).astype(np.float32)
        x = (1 - frac) * x[i0] + frac * x[i0 + 1]
    return x.astype(np.float32)


def synth_input(sr, seconds):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    rng = np.random.default_rng(12345)
    # 类语音的几个谐波 + 调幅，叠加有色噪声，得到时变 lsnr。
    speech = np.zeros(n, dtype=np.float64)
    for f0, amp in [(150, 0.5), (300, 0.3), (450, 0.2), (900, 0.12)]:
        speech += amp * np.sin(2 * np.pi * f0 * t)
    env = 0.5 * (1 + np.sin(2 * np.pi * 2.0 * t))  # 2 Hz 调幅
    speech *= env
    noise = rng.standard_normal(n)
    # 简单低通有色噪声
    for _ in range(4):
        noise = np.convolve(noise, np.ones(8) / 8, mode="same")
    x = 0.6 * speech + 0.25 * noise
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.7
    return x.astype(np.float32)


def main():
    cfg = parse_ini(os.path.join(MODEL_DIR, "config.ini"))
    sr = int(cfg.get("sr", 48000))
    fft = int(cfg.get("fft_size", 960))
    hop = int(cfg.get("hop_size", 480))
    nb_erb = int(cfg.get("nb_erb", 32))
    nb_df = int(cfg.get("nb_df", 96))
    df_order = int(cfg.get("df_order", 5))
    df_lookahead = int(cfg.get("df_lookahead", 0))
    conv_lookahead = int(cfg.get("conv_lookahead", 0))
    min_nb_erb_freqs = int(cfg.get("min_nb_erb_freqs", 2))
    tau = float(cfg.get("norm_tau", 1))
    enc_kernel_t = int(cfg.get("conv_kernel_inp", "3,3").split(",")[0])
    lookahead = max(df_lookahead, conv_lookahead)
    enc_window = enc_kernel_t

    alpha = norm_alpha(sr, hop, tau)

    if len(sys.argv) > 1:
        x = load_wav_48k_mono(sys.argv[1], sr)
        nframes = len(x) // hop
        x = x[:nframes * hop]
    else:
        x = synth_input(sr, 3.0)
        nframes = len(x) // hop
        x = x[:nframes * hop]

    model_path = os.path.join(MODEL_DIR, "combined_streaming.onnx")
    if not os.path.exists(model_path):
        # 模型缺失：自动下载离线 ONNX 并构建有状态流式模型。
        print(f"combined_streaming.onnx not found at {model_path}; building it...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(HERE, "build_streaming_onnx.py")], check=True)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    out_names = ["lsnr", "m", "coefs", "h1", "erb_h1", "df_h1"]

    df = DfState(sr, fft, hop, nb_erb, nb_df, min_nb_erb_freqs)
    freq = df.freq

    # GRU 状态
    h0 = np.zeros((1, 1, 512), dtype=np.float32)
    erb_h0 = np.zeros((2, 1, 512), dtype=np.float32)
    df_h0 = np.zeros((3, 1, 512), dtype=np.float32)

    # enc 环形缓冲
    enc_erb = np.zeros((enc_window, nb_erb), dtype=np.float32)
    enc_spec = np.zeros((enc_window, nb_df, 2), dtype=np.float32)
    enc_pos = enc_window - 1
    enc_count = enc_window - 1

    # 滚动谱缓冲
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
        spec = df.analysis(frame)  # complex64 [freq]

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
        h0 = np.asarray(outs[3]).astype(np.float32).reshape(1, 1, 512)
        erb_h0 = np.asarray(outs[4]).astype(np.float32).reshape(2, 1, 512)
        df_h0 = np.asarray(outs[5]).astype(np.float32).reshape(3, 1, 512)

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

    meta = {
        "sr": sr, "fft_size": fft, "hop_size": hop, "freq_bins": freq,
        "nb_erb": nb_erb, "nb_df": nb_df, "df_order": df_order,
        "lookahead": lookahead, "enc_window": enc_window,
        "alpha": float(alpha), "frames": nframes, "input_len": int(nframes * hop),
        "erb_widths": df.erb, "model": "dfn3_ll/combined_streaming.onnx",
        "feat_spec_layout": "frame-major [frames,2(re/im),nb_df]",
        "coefs_layout": "frame-major [frames,nb_df,df_order*2]",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    rms = float(np.sqrt(np.mean(enhanced ** 2)))
    print(f"frames={nframes} input_len={len(x)} enhanced_rms={rms:.5f}")
    print(f"lsnr range=[{lsnr_all.min():.1f}, {lsnr_all.max():.1f}] dB")
    print(f"vectors -> {OUT_DIR}")


if __name__ == "__main__":
    main()
