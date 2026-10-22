# -*- coding: utf-8 -*-
"""Dump golden vectors for CAM++ ONNX regression validation.

输出 outputs/vectors/:
  <name>.wav            测试音频副本 (C# 测试只依赖本目录)
  <name>.fbank.bin      Kaldi fbank, CMN 前 [T,80] f32LE (输入按 kaldi 约定 ×32768)
  <name>.cmn.bin        CMN 后特征 [T,80] f32LE (ONNX 模型的直接输入)
  <name>.embedding.bin  ONNX embedding [192] f32LE
  <name>.windows.bin    滑窗 embedding [K,192] f32LE (window/hop 见 meta, 校验 C# 流式窗口逻辑)
  meta.json             形状/参数/成对余弦得分

C# 侧对拍链路: wav → fbank(对 .fbank.bin) → CMN(对 .cmn.bin) → onnx(对 .embedding.bin)
             → 滑窗(对 .windows.bin) → 余弦(对 meta.scores)

注意: 滑窗黄金向量对每个 hop 无条件产出, 不建模 C# 的静音门;
C# 侧逐窗对拍必须设 MinWindowRms=0, 否则静音窗被跳过后按下标对齐会错位。
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "speech_campplus_sv_zh-cn_16k-common")
DEFAULT_OUT_DIR = os.path.normpath(os.path.join(
    SCRIPT_DIR, "..", "..", "src", "VoiceBox.Speaker.CampPlus", "runtimes", "models", "speaker", "campplus"))
VECTORS_DIR = os.path.join(SCRIPT_DIR, "outputs", "vectors")

EXAMPLES = ["speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav", "speaker2_a_cn_16k.wav"]
WAVEFORM_SCALE = 32768.0   # C# 内部 kaldi int16 幅度约定
WINDOW_FRAMES = 198        # 滑窗黄金向量: 2.0s 窗
HOP_FRAMES = 25            # 250ms 跳


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="含 campplus.onnx 的部署目录")
    return p.parse_args()


def load_wav(path, expect_sr=16000):
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == expect_sr, f"{path}: sample rate {sr} != {expect_sr}"
    return data[:, 0]


def kaldi_fbank(samples, scale=WAVEFORM_SCALE):
    import torchaudio.compliance.kaldi as Kaldi
    wav = torch.from_numpy(samples * scale).unsqueeze(0)
    return Kaldi.fbank(wav, num_mel_bins=80).numpy()  # [T, 80], CMN 前


def cmn(feat):
    return feat - feat.mean(axis=0, keepdims=True)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def save_bin(path, arr):
    np.ascontiguousarray(arr, dtype=np.float32).tofile(path)


def read_threshold(model_dir):
    """与 01_export_onnx.py 同源: configuration.json 的 yesOrno_thr。"""
    with open(os.path.join(model_dir, "configuration.json"), encoding="utf-8") as f:
        return float(json.load(f)["model"].get("yesOrno_thr", 0.31))


def main():
    args = parse_args()
    onnx_path = os.path.join(args.out_dir, "campplus.onnx")
    assert os.path.exists(onnx_path), f"run 01_export_onnx.py first: {onnx_path} missing"
    os.makedirs(VECTORS_DIR, exist_ok=True)

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def embed(feats_2d):
        return sess.run(None, {"feats": feats_2d[None, ...].astype(np.float32)})[0][0]

    meta = {
        "sample_rate": 16000,
        "waveform_scale": WAVEFORM_SCALE,
        "feature_dim": 80,
        "embedding_size": 192,
        "window_frames": WINDOW_FRAMES,
        "hop_frames": HOP_FRAMES,
        "threshold": read_threshold(args.model_dir),
        "files": {},
        "scores": {},
    }

    embeddings = {}
    for name in EXAMPLES:
        stem = os.path.splitext(name)[0]
        wav_path = os.path.join(args.model_dir, "examples", name)
        samples = load_wav(wav_path)
        shutil.copy(wav_path, os.path.join(VECTORS_DIR, name))

        fbank = kaldi_fbank(samples)              # [T, 80]
        feats = cmn(fbank)                        # [T, 80]
        emb = embed(feats)                        # [192]
        embeddings[stem] = emb

        # 滑窗黄金向量: 每个窗独立 CMN (与 C# 流式路径一致)
        t = fbank.shape[0]
        windows = []
        for start in range(0, t - WINDOW_FRAMES + 1, HOP_FRAMES):
            win = cmn(fbank[start:start + WINDOW_FRAMES])
            windows.append(embed(win))
        windows = np.stack(windows) if windows else np.zeros((0, 192), np.float32)

        save_bin(os.path.join(VECTORS_DIR, f"{stem}.fbank.bin"), fbank)
        save_bin(os.path.join(VECTORS_DIR, f"{stem}.cmn.bin"), feats)
        save_bin(os.path.join(VECTORS_DIR, f"{stem}.embedding.bin"), emb)
        save_bin(os.path.join(VECTORS_DIR, f"{stem}.windows.bin"), windows)
        meta["files"][stem] = {
            "wav": name,
            "num_samples": int(samples.shape[0]),
            "num_frames": int(t),
            "num_windows": int(windows.shape[0]),
        }
        print(f"{stem}: samples={samples.shape[0]} frames={t} windows={windows.shape[0]}")

    stems = [os.path.splitext(n)[0] for n in EXAMPLES]
    for i in range(len(stems)):
        for j in range(i + 1, len(stems)):
            key = f"{stems[i]}|{stems[j]}"
            meta["scores"][key] = round(cosine(embeddings[stems[i]], embeddings[stems[j]]), 6)

    with open(os.path.join(VECTORS_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"scores: {meta['scores']}")
    print(f"vectors written to {VECTORS_DIR}")


if __name__ == "__main__":
    main()
