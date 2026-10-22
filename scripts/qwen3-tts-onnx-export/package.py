# Copyright (c) 2026
# Copy the Qwen BBPE tokenizer files and write metadata.json into the C# deployment package.
import os
import json
import shutil
import argparse

import _common as C

TOKENIZER_FILES = ["vocab.json", "merges.txt", "tokenizer_config.json"]


def run():
    tok_dir = os.path.join(C.PKG_DIR, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    copied = []
    for fn in TOKENIZER_FILES:
        src = os.path.join(C.MODEL, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(tok_dir, fn))
            copied.append(fn)
    cfg = C.load_config()
    graphs = {
        "talker": "talker.onnx",
        "codePredictorSample": "code_predictor_sample.onnx",
        "codePredictorGreedy": "code_predictor_greedy.onnx",
        "codecDecoder": "codec_decoder.onnx",
    }
    legacy_cp = os.path.join(C.PKG_DIR, "code_predictor.onnx")
    if os.path.isfile(legacy_cp):
        graphs["codePredictor"] = "code_predictor.onnx"
    meta = {
        "ttsModelSize": cfg.get("tts_model_size"),
        "ttsModelType": cfg.get("tts_model_type"),
        "tokenizerType": cfg.get("tokenizer_type"),
        "dtype": C.DTYPE,
        "opset": C.OPSET,
        "sampleRate": 24000,
        "graphs": graphs,
        "embeddingsConfig": "embeddings/config.json",
        "tokenizerDir": "tokenizer",
    }
    with open(os.path.join(C.PKG_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"   tokenizer files: {copied}")
    print(f"   metadata.json -> {C.PKG_DIR}")


def main():
    ap = argparse.ArgumentParser(description="Package tokenizer + metadata for C#.")
    C.add_io_args(ap)
    args = ap.parse_args()
    C.apply_io_args(args)
    run()


if __name__ == "__main__":
    main()
