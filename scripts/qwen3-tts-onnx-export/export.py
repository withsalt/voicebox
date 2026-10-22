# Copyright (c) 2026
# One-shot Qwen3-TTS -> ONNX export driver. Builds a self-contained C# deployment package:
#
#   <out-dir>/qwen3-tts-<size>-onnx-<dtype>/
#     talker.onnx                       unified prefill/decode transformer (KV cache)
#     code_predictor_sample/greedy.onnx sub-talker whole-frame fused graphs
#     codec_decoder.onnx                codes -> 24kHz waveform (always fp32)
#     embeddings/*.bin + config.json    projected text/codec embeddings + full model config + gen defaults
#     tokenizer/{vocab.json,merges.txt,tokenizer_config.json}
#     metadata.json
#
# Usage:
#   python export.py --model-dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice --out-dir ./out --dtype fp16
#   python export.py --dtype fp32      # strict e2e parity (byte-exact greedy codes)
import argparse

import _common as C


def main():
    ap = argparse.ArgumentParser(description="Export Qwen3-TTS (12Hz CustomVoice) to a C# ONNX package.")
    C.add_io_args(ap)
    ap.add_argument("--skip-parity", action="store_true", help="skip the end-to-end PyTorch parity check.")
    args = ap.parse_args()
    C.apply_io_args(args)
    print(f"== model-dir : {C.MODEL}")
    print(f"== package   : {C.PKG_DIR}")
    print(f"== dtype     : {C.DTYPE}   opset: {C.OPSET}   device: {C.DEVICE}\n")

    import export_codec_decoder, export_embeddings, export_talker, export_code_predictor, package

    # Note: talker/cp fp16 export mutates the model in place (.half()), so each step loads its
    # own copy rather than sharing one — robust over the small extra load cost.
    print("\n========== [1/5] codec_decoder (fp32) ==========")
    export_codec_decoder.run()
    print("\n========== [2/5] embeddings + config ==========")
    export_embeddings.run()
    print(f"\n========== [3/5] talker ({C.DTYPE}) ==========")
    export_talker.run()
    print(f"\n========== [4/5] code_predictor sample/greedy ({C.DTYPE}) ==========")
    export_code_predictor.run()
    print("\n========== [5/5] package (tokenizer + metadata) ==========")
    package.run()

    if not args.skip_parity:
        print("\n========== parity (e2e vs PyTorch) ==========")
        import parity_e2e
        parity_e2e.run(strict=(C.DTYPE == "fp32"))

    print("\n== export complete ==")
    print(f"   package: {C.PKG_DIR}")
    print("   C# Qwen3TTSOptions.ModelPath -> this directory.")
    C.gate_and_exit()


if __name__ == "__main__":
    main()
