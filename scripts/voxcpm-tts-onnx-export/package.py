# Copyright (c) 2026
# Final packaging: tokenizer files + metadata.json (graph dtype map, versions).
import os
import json
import shutil
import argparse
import datetime

import _common as C

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def run():
    for name in TOKENIZER_FILES:
        src = os.path.join(C.MODEL, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(C.PKG_DIR, "tokenizer", name))
            print(f"   tokenizer/{name}")
        else:
            print(f"   (skip missing {name})")

    import onnx
    import onnxruntime
    import torch

    # int8 包: lm/cfm/fe 权重 int8 (MatMulNBits acc4), vae 恒 fp32
    small = {"fp16": "fp16", "int8": "int8"}.get(C.DTYPE, "fp32")
    graph_dtypes = {
        "lm.onnx": C.DTYPE,
        "feat_encoder.onnx": small,
        "cfm_decode.onnx": small,
        "audio_vae_decoder.onnx": "fp32",
        "audio_vae_encoder.onnx": "fp32",
    }
    if C.DTYPE != "fp16":
        graph_dtypes["lm_gqa.onnx"] = C.DTYPE   # 与 lm.onnx 同权重的 GQA 加速变体
    meta = {
        "package": os.path.basename(C.PKG_DIR),
        "dtype": C.DTYPE,
        "opset": C.OPSET,
        "graph_dtypes": graph_dtypes,
        "quant_block_size": C.QUANT_BLOCK_SIZE if C.DTYPE == "int8" else None,
        "exported_with": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_model": os.path.basename(C.MODEL),
    }
    mp = os.path.join(C.PKG_DIR, "metadata.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"   wrote {mp}")

    total = 0
    for root, _, files in os.walk(C.PKG_DIR):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    print(f"   package total: {total/1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser(description="Package tokenizer + metadata for the C# deployment dir.")
    C.add_io_args(ap)
    args = ap.parse_args()
    C.apply_io_args(args)
    run()


if __name__ == "__main__":
    main()
