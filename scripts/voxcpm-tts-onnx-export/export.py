# Copyright (c) 2026
# One-shot VoxCPM-0.5B -> ONNX export driver. Builds a self-contained C# deployment package:
#
#   <out-dir>/voxcpm-0.5b-onnx-<dtype>/
#     lm.onnx                         unified prefill/decode LM (base+residual, KV cache)
#     feat_encoder.onnx               latent patches -> LM embeds (prompt + per-step)
#     cfm_decode.onnx                 fused 10-step Euler flow-matching per patch
#     audio_vae_decoder.onnx          latents -> 16kHz waveform (always fp32)
#     audio_vae_encoder.onnx          waveform -> latents, voice-clone prompt (always fp32)
#     embeddings/text_embedding.bin   [V,H] fp32 host lookup table
#     embeddings/multichar_zh.json    zh multi-char token split table (tokenizer parity)
#     config.json / metadata.json     C# manifest + package info
#     tokenizer/                      tokenizer.json + configs
#
# Usage:
#   python export.py --model-dir ./VoxCPM-0.5B --out-dir ./out --dtype fp32
#   python export.py --dtype int8        # smallest, weight-only int8 (MatMulNBits acc4)
import argparse

import _common as C


def main():
    ap = argparse.ArgumentParser(description="Export VoxCPM-0.5B to a C# ONNX package.")
    C.add_io_args(ap)
    ap.add_argument("--skip-parity", action="store_true", help="skip the end-to-end PyTorch parity check.")
    args = ap.parse_args()
    C.apply_io_args(args)
    print(f"== model-dir : {C.MODEL}")
    print(f"== package   : {C.PKG_DIR}")
    print(f"== dtype     : {C.DTYPE}   opset: {C.OPSET}   device: {C.DEVICE}\n")

    import export_lm, export_cfm, export_audio_vae, export_embeddings, package

    vox = C.load_vox()
    print("\n========== [1/6] embeddings + manifest ==========")
    export_embeddings.run(vox)
    print(f"\n========== [2/6] lm + lm_gqa + feat_encoder ({C.DTYPE}) ==========")
    export_lm.run(vox)
    export_lm.run_gqa(vox)
    export_lm.run_feat_encoder(vox)
    print(f"\n========== [3/6] cfm_decode ({'fp16' if C.DTYPE == 'fp16' else 'fp32'}) ==========")
    export_cfm.run(vox)
    print("\n========== [4/6] audio_vae (fp32) ==========")
    export_audio_vae.run(vox)
    print("\n========== [5/6] package (tokenizer + metadata) ==========")
    package.run()

    if not args.skip_parity:
        print("\n========== [6/6] parity (e2e vs PyTorch) + golden ==========")
        import parity_e2e

        parity_e2e.run(strict=(C.DTYPE == "fp32"))

    print("\n== export complete ==")
    print(f"   package: {C.PKG_DIR}")
    print("   C# VoxCPMTTSOptions.ModelPath -> this directory.")
    C.gate_and_exit()


if __name__ == "__main__":
    main()
