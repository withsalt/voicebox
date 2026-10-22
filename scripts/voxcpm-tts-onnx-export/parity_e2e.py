# Copyright (c) 2026
# End-to-end parity: PyTorch VoxCPMModel vs the exported ONNX package (via _engine.py),
# with IDENTICAL per-step CFM noise on both sides (host-injected numpy RNG). Also captures
# the golden files used by the regression validation flow:
#
#   out/golden/
#     text_ids_{case}.npy       int64  [T]              tokenized ids incl. audio_start
#     latents_{case}.npy        float32 [Dz,steps*patch] ONNX-engine latents (fp32 pkg)
#     wav_{case}.npy            float32 [S]              ONNX-engine waveform
#     wav_stream_{case}.npy     float32 [S]              concat of streaming 80ms chunks
#     noise_{case}.npy          float32 [steps,Dz,patch] the exact CFM noises used
#     prompt_16k.wav / prompt_feat.npy                   voice-clone prompt (pre-resampled)
#     golden_meta.json                                   texts, seeds, steps, tolerances
#
# Cases: en (plain + streaming), zh (plain), clone (prompt continuation, en prompt).
#
# strict=True (fp32): SNR/cos gate. fp16/int4: informational smoke only (finite,
# non-silent, plausible length) -- diffusion trajectories diverge under reduced
# precision, so sample-level parity is not a meaningful gate there.
import os
import json
import argparse

import numpy as np
import torch

import _common as C

EN_TEXT = "Hello, this is a test of the VoxCPM text to speech system."
ZH_TEXT = "今天天气真不错，我们一起去公园散步吧。"
PROMPT_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VoxCPM", "examples", "example.wav")
PROMPT_TEXT = "This is an example audio transcript for training."
CLONE_TEXT = "The quick brown fox jumps over the lazy dog."


class _NoiseTap:
    """Feeds a shared noise sequence to BOTH torch and onnx paths (and records it)."""

    def __init__(self, seed, feat_dim, patch):
        self.rng = np.random.default_rng(seed)
        self.feat_dim = feat_dim
        self.patch = patch
        self.trace = []

    def next(self):
        z = self.rng.standard_normal((1, self.feat_dim, self.patch), dtype=np.float32)
        self.trace.append(z.copy())
        return z


class _CfmNoisePatch:
    """Context manager: UnifiedCFM.forward draws z from the tap instead of torch.randn."""

    def __init__(self, tap):
        self.tap = tap

    def __enter__(self):
        from voxcpm.modules.locdit.unified_cfm import UnifiedCFM

        self._cls = UnifiedCFM
        self._orig = UnifiedCFM.forward
        tap = self.tap

        def patched(mod, mu, n_timesteps, patch_size, cond, temperature=1.0, cfg_value=1.0,
                    sway_sampling_coef=1.0, use_cfg_zero_star=True):
            z = torch.from_numpy(tap.next()).to(mu.device, mu.dtype)
            t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
            t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
            return mod.solve_euler(x=z, t_span=t_span, mu=mu, cond=cond,
                                   cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)

        UnifiedCFM.forward = patched
        return self

    def __exit__(self, *exc):
        self._cls.forward = self._orig
        return False


class _ReplayRng:
    """np.random.default_rng stand-in replaying a recorded noise trace in the engine."""

    def __init__(self, trace):
        self._it = iter([z.copy() for z in trace])

    def standard_normal(self, shape, dtype=np.float32):
        return next(self._it).reshape(shape).astype(dtype)


def _engine_with_replay(eng, trace, **kwargs):
    orig = np.random.default_rng
    np.random.default_rng = lambda seed=None: _ReplayRng(trace)
    try:
        return eng.generate(**kwargs)
    finally:
        np.random.default_rng = orig


def _load_prompt_16k(vox):
    """example.wav -> mono 16k float32 via torchaudio.functional.resample (pure ops;
    torchaudio.load itself needs torchcodec, so decode with soundfile)."""
    import soundfile as sf
    import torchaudio

    data, sr = sf.read(PROMPT_WAV, dtype="float32", always_2d=True)
    w = torch.from_numpy(data.T)  # [ch,S]
    if w.size(0) > 1:
        w = w.mean(dim=0, keepdim=True)
    if sr != vox.sample_rate:
        w = torchaudio.functional.resample(w, sr, vox.sample_rate)
    return w[0].numpy().copy()


def _torch_prompt_feat(vox, wav16k):
    """build_prompt_cache's feature extraction without torchaudio.load (wav already mono 16k)."""
    audio = torch.from_numpy(wav16k).unsqueeze(0)
    patch_len = vox.patch_size * vox.chunk_size
    if audio.size(1) % patch_len != 0:
        audio = torch.nn.functional.pad(audio, (patch_len - audio.size(1) % patch_len, 0))
    with torch.inference_mode():
        feat = vox.audio_vae.encode(audio.unsqueeze(0).float(), vox.sample_rate).cpu()
    return feat.view(vox.audio_vae.latent_dim, -1, vox.patch_size).permute(1, 2, 0)


def _smoke(name, eng, wav):
    ok = bool(np.isfinite(wav).all()) and float(np.abs(wav).max()) > 1e-3 and len(wav) >= eng.patch_len * 3
    print(f"   smoke[{name}]: finite={bool(np.isfinite(wav).all())} peak={float(np.abs(wav).max()):.3f} "
          f"len={len(wav)} ok={ok}")
    if not ok:
        C._RESULTS.append(dict(name=f"smoke_{name}", n=len(wav), max_abs=1.0,
                               snr_db=0.0, cos=0.0, len_mismatch=True))


def _save_golden(case, ids, out, trace, meta, eng, extra=None):
    np.save(os.path.join(C.GOLD_DIR, f"text_ids_{case}.npy"), np.asarray(ids, dtype=np.int64))
    np.save(os.path.join(C.GOLD_DIR, f"latents_{case}.npy"), out["latents"].astype(np.float32))
    np.save(os.path.join(C.GOLD_DIR, f"wav_{case}.npy"), out["wav"].astype(np.float32))
    np.save(os.path.join(C.GOLD_DIR, f"noise_{case}.npy"), np.concatenate(trace, axis=0).astype(np.float32))
    entry = dict(name=case, steps=out["steps"], wav_len=int(len(out["wav"])),
                 sample_rate=eng.cfg["sample_rate"])
    entry.update(extra or {})
    meta["cases"].append(entry)


def run(strict=True):
    from _engine import VoxCpmOnnxEngine

    eng = VoxCpmOnnxEngine(C.PKG_DIR)
    vox = C.load_vox()
    meta = {"cases": [], "package": os.path.basename(C.PKG_DIR), "dtype": C.DTYPE,
            "prompt_text": PROMPT_TEXT, "clone_text": CLONE_TEXT}

    # ---------------- tokenizer parity (cheap, always) ----------------
    for tag, text in [("en", EN_TEXT), ("zh", ZH_TEXT), ("clone", PROMPT_TEXT + CLONE_TEXT)]:
        eng_ids = eng.tokenize(text)
        ref_ids = list(vox.text_tokenizer(text))
        if eng_ids != ref_ids:
            print(f"   tokenizer[{tag}]: MISMATCH\n     eng: {eng_ids}\n     ref: {ref_ids}")
            C._RESULTS.append(dict(name=f"tokenizer_{tag}", n=len(ref_ids), max_abs=1.0,
                                   snr_db=0.0, cos=0.0, len_mismatch=True))
        else:
            print(f"   tokenizer[{tag}]: MATCH ({len(eng_ids)} ids)")

    # ---------------- case en: plain + streaming ----------------
    print(f"\n== case en: {EN_TEXT!r} (seed=1234)")
    limit = min(int(len(eng.tokenize(EN_TEXT)) * eng.ratio + eng.bias), eng.max_length)
    tap = _NoiseTap(1234, eng.feat_dim, eng.patch)
    if strict:
        with _CfmNoisePatch(tap):
            with torch.inference_mode():
                ref_wav = vox.generate(target_text=EN_TEXT, max_len=limit, seed=0)
        ref_wav = ref_wav.squeeze(0).float().cpu().numpy()
        out = _engine_with_replay(eng, tap.trace, text=EN_TEXT, collect_latents=True)
        print(f"   torch steps={len(tap.trace)}, onnx steps={out['steps']}, "
              f"wav {len(ref_wav)} vs {len(out['wav'])}")
        C.compare("e2e_wav_en", ref_wav, out["wav"], atol_warn=5e-3, min_snr=20.0, min_cos=0.995)

        # streaming: torch chunks vs engine chunks on the same trajectory (fresh tap, same seed)
        tap_s = _NoiseTap(1234, eng.feat_dim, eng.patch)
        with _CfmNoisePatch(tap_s):
            with torch.inference_mode():
                ref_chunks = [c.squeeze(0).float().cpu().numpy()
                              for c in vox.generate_streaming(target_text=EN_TEXT, max_len=limit, seed=0)]
        out_s = _engine_with_replay(eng, tap_s.trace, text=EN_TEXT, streaming=True, collect_latents=True)
        ref_cat = np.concatenate(ref_chunks)
        eng_cat = np.concatenate(out_s["chunks"])
        print(f"   streaming: torch {len(ref_chunks)} chunks vs onnx {len(out_s['chunks'])}")
        C.compare("e2e_stream_en", ref_cat, eng_cat, atol_warn=5e-3, min_snr=20.0, min_cos=0.995)
        np.save(os.path.join(C.GOLD_DIR, "wav_stream_en.npy"), eng_cat.astype(np.float32))
    else:
        out = eng.generate(text=EN_TEXT, seed=1234, collect_latents=True)
        out_s = eng.generate(text=EN_TEXT, seed=1234, streaming=True)
        _smoke("en", eng, out["wav"])
        _smoke("en_stream", eng, np.concatenate(out_s["chunks"]))
    ids_en = eng.tokenize(EN_TEXT) + [eng.audio_start]
    if strict:
        _save_golden("en", ids_en, out, tap.trace, meta, eng, {"text": EN_TEXT, "seed": 1234})

    # ---------------- case zh: plain ----------------
    print(f"\n== case zh: {ZH_TEXT!r} (seed=5678)")
    limit = min(int(len(eng.tokenize(ZH_TEXT)) * eng.ratio + eng.bias), eng.max_length)
    tap = _NoiseTap(5678, eng.feat_dim, eng.patch)
    if strict:
        with _CfmNoisePatch(tap):
            with torch.inference_mode():
                ref_wav = vox.generate(target_text=ZH_TEXT, max_len=limit, seed=0)
        ref_wav = ref_wav.squeeze(0).float().cpu().numpy()
        out = _engine_with_replay(eng, tap.trace, text=ZH_TEXT, collect_latents=True)
        print(f"   torch steps={len(tap.trace)}, onnx steps={out['steps']}, "
              f"wav {len(ref_wav)} vs {len(out['wav'])}")
        C.compare("e2e_wav_zh", ref_wav, out["wav"], atol_warn=5e-3, min_snr=20.0, min_cos=0.995)
        _save_golden("zh", eng.tokenize(ZH_TEXT) + [eng.audio_start], out, tap.trace, meta, eng,
                     {"text": ZH_TEXT, "seed": 5678})
    else:
        out = eng.generate(text=ZH_TEXT, seed=5678)
        _smoke("zh", eng, out["wav"])

    # ---------------- case clone: prompt continuation ----------------
    print(f"\n== case clone: {CLONE_TEXT!r} (seed=97531, prompt={os.path.basename(PROMPT_WAV)})")
    wav16k = _load_prompt_16k(vox)
    prompt_feat = eng.encode_prompt(wav16k)
    if strict:
        import soundfile as sf

        # FLOAT subtype: C# 测试读此文件后过同一 VAE 图, 必须拿到与 prompt_feat.npy
        # 完全相同的输入 (默认 PCM_16 会引入量化差, AR 环路混沌会放大它)
        sf.write(os.path.join(C.GOLD_DIR, "prompt_16k.wav"), wav16k, vox.sample_rate, subtype="FLOAT")
        np.save(os.path.join(C.GOLD_DIR, "prompt_feat.npy"), prompt_feat.astype(np.float32))
    limit = min(int(len(eng.tokenize(CLONE_TEXT)) * eng.ratio + eng.bias), eng.max_length)
    tap = _NoiseTap(97531, eng.feat_dim, eng.patch)
    if strict:
        # VAE-encode parity is gated here on its own; the AR loop itself is chaotic
        # (FSQ rounding) so BOTH references must start from the SAME prompt latents --
        # feed torch the ONNX-encoded features, exactly like the C# runtime will use.
        ref_feat = _torch_prompt_feat(vox, wav16k)
        C.compare("clone_prompt_feat_onnx_vs_torch", ref_feat.numpy(), prompt_feat)
        cache = {"prompt_text": PROMPT_TEXT, "audio_feat": torch.from_numpy(prompt_feat.copy())}
        with _CfmNoisePatch(tap):
            with torch.inference_mode():
                ref_wav, _, _ = vox.generate_with_prompt_cache(
                    target_text=CLONE_TEXT, prompt_cache=cache, max_len=limit, seed=0)
        ref_wav = ref_wav.squeeze(0).float().cpu().numpy()
        out = _engine_with_replay(eng, tap.trace, text=CLONE_TEXT, prompt_text=PROMPT_TEXT,
                                  prompt_wav=wav16k, collect_latents=True)
        print(f"   torch steps={len(tap.trace)}, onnx steps={out['steps']}, "
              f"wav {len(ref_wav)} vs {len(out['wav'])}")
        C.compare("e2e_wav_clone", ref_wav, out["wav"], atol_warn=5e-3, min_snr=20.0, min_cos=0.995)
        _save_golden("clone", eng.tokenize(PROMPT_TEXT + CLONE_TEXT) + [eng.audio_start], out,
                     tap.trace, meta, eng,
                     {"text": CLONE_TEXT, "prompt_text": PROMPT_TEXT, "seed": 97531,
                      "prompt_wav": "prompt_16k.wav", "prompt_patches": int(prompt_feat.shape[0])})
    else:
        out = eng.generate(text=CLONE_TEXT, prompt_text=PROMPT_TEXT, prompt_wav=wav16k, seed=97531)
        _smoke("clone", eng, out["wav"])

    if strict:  # golden files only come from the fp32 package
        with open(os.path.join(C.GOLD_DIR, "golden_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\n   golden -> {C.GOLD_DIR}")


def main():
    ap = argparse.ArgumentParser(description="VoxCPM e2e parity (torch vs ONNX package) + golden capture.")
    C.add_io_args(ap)
    args = ap.parse_args()
    C.apply_io_args(args)
    run(strict=(C.DTYPE == "fp32"))
    C.gate_and_exit()


if __name__ == "__main__":
    main()
