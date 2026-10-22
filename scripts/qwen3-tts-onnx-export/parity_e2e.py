# Copyright (c) 2026
# End-to-end parity: the exported ONNX package (driven by the numpy reference engine, which
# mirrors the C# pipeline) must reproduce the original PyTorch model's greedy output.
#
#   * golden: tts.generate_custom_voice(..., do_sample=False, subtalker_dosample=False)
#             with the prompt inputs_embeds / codes captured via monkeypatch.
#   * engine fed the CAPTURED prompt -> codes must match the golden codes EXACTLY (greedy),
#             and the decoded wav must match within SNR>=40dB / cos>=0.999.
#   * golden artifacts are saved to GOLD_DIR for the C# parity tests.
import os
import json
import argparse
import numpy as np
import torch

import _common as C
from _engine import Qwen3OnnxEngine, build_prompt

TEXT = "这是一个测试。"
SPEAKER = "Vivian"
LANGUAGE = "Chinese"


def capture_golden(tts=None):
    print(f"== golden: text={TEXT!r} speaker={SPEAKER} language={LANGUAGE} (greedy)")
    if tts is None:
        tts = C.load_tts(device="cpu", dtype="float32")
    cap = {}
    orig_tg = tts.model.talker.generate

    def cap_tg(*a, **k):
        cap["inputs_embeds"] = C.to_numpy(k["inputs_embeds"])
        cap["attention_mask"] = C.to_numpy(k["attention_mask"])
        cap["tts_pad_embed"] = C.to_numpy(k["tts_pad_embed"])
        return orig_tg(*a, **k)

    tts.model.talker.generate = cap_tg
    orig_mg = tts.model.generate

    def cap_mg(*a, **k):
        res = orig_mg(*a, **k)
        cap["codes"] = C.to_numpy(res[0][0]).astype(np.int64)   # [T,16]
        return res

    tts.model.generate = cap_mg

    at = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    cap["text_ids"] = C.to_numpy(tts.processor(text=at, return_tensors="pt")["input_ids"]).reshape(-1).astype(np.int64)

    with torch.inference_mode():
        wavs, sr = tts.generate_custom_voice(text=TEXT, speaker=SPEAKER, language=LANGUAGE,
                                             do_sample=False, subtalker_dosample=False, max_new_tokens=512)
    cap["wav"] = np.asarray(wavs[0], dtype=np.float32)
    cap["sr"] = int(sr)
    print(f"   golden codes shape={cap['codes'].shape} wav={cap['wav'].shape} sr={sr}")
    return cap


def run(tts=None, strict=True):
    cap = capture_golden(tts)
    # save golden for C# tests
    np.save(os.path.join(C.GOLD_DIR, "prompt_inputs_embeds.npy"), cap["inputs_embeds"])
    np.save(os.path.join(C.GOLD_DIR, "tts_pad_embed.npy"), cap["tts_pad_embed"])
    np.save(os.path.join(C.GOLD_DIR, "golden_codes.npy"), cap["codes"])
    np.save(os.path.join(C.GOLD_DIR, "golden_wav.npy"), cap["wav"])
    with open(os.path.join(C.GOLD_DIR, "golden_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"text": TEXT, "speaker": SPEAKER, "language": LANGUAGE, "sr": cap["sr"],
                   "promptLen": int(cap["inputs_embeds"].shape[1]),
                   "numFrames": int(cap["codes"].shape[0])}, f, ensure_ascii=False, indent=2)

    print("== running ONNX engine on captured prompt ...")
    eng = Qwen3OnnxEngine(C.PKG_DIR)

    # prompt-construction parity: build_prompt (the C# spec) vs the model's own prompt
    np.save(os.path.join(C.GOLD_DIR, "text_ids.npy"), cap["text_ids"])
    my_prompt, my_pad = build_prompt(eng.store, cap["text_ids"], SPEAKER, LANGUAGE)
    C.compare("prompt_construction_vs_model", cap["inputs_embeds"], my_prompt, atol_warn=1e-4)
    C.compare("prompt_pad_vs_model", cap["tts_pad_embed"].reshape(-1), my_pad.reshape(-1), atol_warn=1e-4)

    prompt = cap["inputs_embeds"]
    attn_len = int(cap["attention_mask"].shape[1])
    pad = cap["tts_pad_embed"]
    codes = eng.generate(prompt, attn_len, pad, max_frames=cap["codes"].shape[0] + 8)
    print(f"   engine codes shape={codes.shape}  golden={cap['codes'].shape}")

    # codes: byte-exact for greedy fp32; fp16/int8 will argmax-flip (expected) -> only report.
    g = cap["codes"]
    n = min(len(codes), len(g))
    same = codes.shape == g.shape
    match_frames = int((codes[:n] == g[:n]).all(axis=1).sum())
    print(f"   codes shape engine={codes.shape} golden={g.shape}  frame-exact={match_frames}/{len(g)}"
          f"  elem-mismatch={int((codes[:n] != g[:n]).sum())}/{g[:n].size}")
    if strict:   # fp32 must reproduce greedy byte-for-byte
        if same:
            C.compare("e2e_codes_match", g.astype(np.float64), codes.astype(np.float64), atol_warn=0)
        else:
            C.compare("e2e_codes_match", g[:n].astype(np.float64), codes[:n].astype(np.float64), atol_warn=0)

    # codec parity (decode golden codes) is dtype-independent (codec is always fp32) -> always gate.
    C.compare("codec_e2e_vs_golden_wav", cap["wav"], eng.decode(g), atol_warn=1e-2)
    # full-pipeline wav (engine's own codes) only meaningful when codes match (fp32).
    if strict and same:
        C.compare("e2e_wav_vs_golden", cap["wav"], eng.decode(codes), atol_warn=1e-2)
    print("e2e parity done.")


def main():
    ap = argparse.ArgumentParser(description="Qwen3-TTS end-to-end parity (ONNX vs PyTorch).")
    C.add_io_args(ap)
    args = ap.parse_args()
    C.apply_io_args(args)
    run()
    C.gate_and_exit(min_snr=40.0, min_cos=0.999)


if __name__ == "__main__":
    main()
