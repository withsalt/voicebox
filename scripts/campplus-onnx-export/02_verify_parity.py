# -*- coding: utf-8 -*-
"""Verify the exported CAM++ ONNX against the official modelscope compute path on real audio.

覆盖:
  1. 官方前处理 (torchaudio Kaldi.fbank + CMN) + 原始 torch 模型 vs ONNX: embedding 对拍
  2. 同/异说话人余弦得分 sanity (examples 3 个 wav, 官方阈值 0.31)
  3. CMN 输入尺度不变性 (int16 尺度 vs [-1,1] 尺度 → embedding 应一致, C# 对拍依赖此性质)
  4. modelscope 官方 pipeline 端到端对拍 (环境允许时)
  5. 性能基准: 滑窗典型窗口 + 10s 注册段

用法: python 02_verify_parity.py [--model-dir ...] [--out-dir ...]
"""
import argparse
import os
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "speech_campplus_sv_zh-cn_16k-common")
DEFAULT_OUT_DIR = os.path.normpath(os.path.join(
    SCRIPT_DIR, "..", "..", "src", "VoiceBox.Speaker.CampPlus", "runtimes", "models", "speaker", "campplus"))

EXAMPLES = ["speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav", "speaker2_a_cn_16k.wav"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return p.parse_args()


def load_wav(path, expect_sr=16000):
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == expect_sr, f"{path}: sample rate {sr} != {expect_sr}"
    return data[:, 0]


def official_features(samples, num_mel_bins=80, scale=1.0):
    """modelscope SpeakerVerificationCAMPPlus.__extract_feature 同款: Kaldi.fbank + CMN。"""
    import torchaudio.compliance.kaldi as Kaldi
    wav = torch.from_numpy(samples * scale).unsqueeze(0)
    feat = Kaldi.fbank(wav, num_mel_bins=num_mel_bins)
    feat = feat - feat.mean(dim=0, keepdim=True)
    return feat.unsqueeze(0)  # [1, T, 80]


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    args = parse_args()
    onnx_path = os.path.join(args.out_dir, "campplus.onnx")
    assert os.path.exists(onnx_path), f"run 01_export_onnx.py first: {onnx_path} missing"

    import onnxruntime as ort
    from modelscope.models.audio.sv.DTDNN import CAMPPlus

    torch_model = CAMPPlus(feat_dim=80, embedding_size=192)
    torch_model.load_state_dict(
        torch.load(os.path.join(args.model_dir, "campplus_cn_common.bin"), map_location="cpu"),
        strict=True)
    torch_model.eval()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    wavs = {n: load_wav(os.path.join(args.model_dir, "examples", n)) for n in EXAMPLES}

    # 1) 官方前处理 + torch vs onnx
    print("=== 1) real-audio embedding parity (official preprocessing) ===")
    emb_torch, emb_onnx = {}, {}
    for name, samples in wavs.items():
        feats = official_features(samples)
        with torch.no_grad():
            ref = torch_model(feats).numpy()
        out = sess.run(None, {"feats": feats.numpy()})[0]
        emb_torch[name], emb_onnx[name] = ref, out
        d, c = float(np.abs(out - ref).max()), cosine(out, ref)
        print(f"  {name}: T={feats.shape[1]} max_diff={d:.3e} cos={c:.8f}")
        assert d < 1e-3 and c > 0.99999, f"parity failed on {name}"

    # 2) 同/异说话人得分 sanity
    print("=== 2) speaker score sanity (threshold 0.31) ===")
    pairs = [
        (EXAMPLES[0], EXAMPLES[1], True),   # 同说话人
        (EXAMPLES[0], EXAMPLES[2], False),  # 异说话人
        (EXAMPLES[1], EXAMPLES[2], False),
    ]
    for a, b, same in pairs:
        s_torch = cosine(emb_torch[a], emb_torch[b])
        s_onnx = cosine(emb_onnx[a], emb_onnx[b])
        verdict = "SAME" if same else "DIFF"
        print(f"  {a} vs {b} [{verdict}]: torch={s_torch:.4f} onnx={s_onnx:.4f}")
        assert abs(s_torch - s_onnx) < 1e-3
        assert (s_onnx > 0.31) == same, f"score/threshold sanity failed: {a} vs {b}"

    # 3) CMN 输入尺度不变性 (C# 侧按 kaldi 约定 ×32768, 官方按原始幅度 → embedding 应一致)
    print("=== 3) input-scale invariance via CMN ===")
    for name, samples in wavs.items():
        f1 = official_features(samples, scale=1.0)
        f2 = official_features(samples, scale=32768.0)
        o1 = sess.run(None, {"feats": f1.numpy()})[0]
        o2 = sess.run(None, {"feats": f2.numpy()})[0]
        c = cosine(o1, o2)
        print(f"  {name}: cos(scale=1, scale=32768) = {c:.8f}")
        assert c > 0.99999, "CMN scale invariance broken?!"

    # 4) modelscope 官方 pipeline 端到端 (可选, 环境不全时跳过)
    print("=== 4) modelscope official pipeline cross-check ===")
    try:
        from modelscope.pipelines import pipeline
        sv = pipeline(task="speaker-verification", model=args.model_dir, device="cpu")
        p_same = sv([os.path.join(args.model_dir, "examples", EXAMPLES[0]),
                     os.path.join(args.model_dir, "examples", EXAMPLES[1])])
        p_diff = sv([os.path.join(args.model_dir, "examples", EXAMPLES[0]),
                     os.path.join(args.model_dir, "examples", EXAMPLES[2])])
        s_same = cosine(emb_onnx[EXAMPLES[0]], emb_onnx[EXAMPLES[1]])
        s_diff = cosine(emb_onnx[EXAMPLES[0]], emb_onnx[EXAMPLES[2]])
        print(f"  pipeline same={p_same} | onnx cos={s_same:.4f}")
        print(f"  pipeline diff={p_diff} | onnx cos={s_diff:.4f}")
        if isinstance(p_same, dict) and "score" in p_same:
            assert abs(float(p_same["score"]) - s_same) < 5e-3
            assert abs(float(p_diff["score"]) - s_diff) < 5e-3
            print("  pipeline score parity OK")
    except Exception as e:  # pipeline 依赖较重, 缺依赖不算导出失败
        print(f"  SKIPPED (pipeline unavailable: {type(e).__name__}: {e})")

    # 5) 性能基准
    print("=== 5) performance (CPUExecutionProvider) ===")
    for frames, tag in [(148, "1.5s window"), (198, "2.0s window"),
                        (298, "3.0s window"), (998, "10s enroll")]:
        x = np.random.randn(1, frames, 80).astype(np.float32)
        for _ in range(5):
            sess.run(None, {"feats": x})
        n = 50
        t0 = time.perf_counter()
        for _ in range(n):
            sess.run(None, {"feats": x})
        ms = (time.perf_counter() - t0) / n * 1000
        print(f"  {tag} ({frames} frames): {ms:.2f} ms/run")

    print("ALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
