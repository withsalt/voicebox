# Copyright (c) 2026
# Export the VoxCPM AudioVAE (always fp32; weight_norm folded, scripted snake swapped):
#
#   audio_vae_decoder.onnx : latents[1,64,T] -> wav[1,1,T*640]   (streaming + final decode)
#   audio_vae_encoder.onnx : wav[1,1,S]      -> latents[1,64,S/640]  (voice-clone prompt only)
#
# The decoder is the per-step hot path in streaming mode (3-patch window = T=6);
# the encoder runs once per voice. Both keep T/S dynamic.
import os
import argparse

import numpy as np
import torch

import _common as C
from _modules import VaeDecoderModule, VaeEncoderModule, prepare_vae_for_export


def run(vox=None):
    if vox is None:
        print("== loading model (fp32) for audio_vae ...")
        vox = C.load_vox()

    g = torch.Generator().manual_seed(3)
    Dz = vox.audio_vae.latent_dim
    chunk = vox.audio_vae.chunk_size

    # references BEFORE folding weight_norm (original parametrisation)
    T = 6
    lat = torch.randn(1, Dz, T, generator=g) * 0.5
    S = chunk * 10
    wav_in = (torch.randn(1, 1, S, generator=g) * 0.1).clamp(-1, 1)
    with torch.inference_mode():
        ref_wav = vox.audio_vae.decode(lat)
        ref_lat = vox.audio_vae.encode(wav_in, vox.sample_rate)

    folded = prepare_vae_for_export(vox)
    print(f"   weight_norm folded on {folded} modules")
    dec = VaeDecoderModule(vox).eval()
    enc = VaeEncoderModule(vox).eval()
    with torch.inference_mode():
        my_wav = dec(lat)
        my_lat = enc(wav_in)
    C.compare("vae_decode_folded_vs_orig", C.to_numpy(ref_wav), C.to_numpy(my_wav), atol_warn=1e-4)
    C.compare("vae_encode_folded_vs_orig", C.to_numpy(ref_lat), C.to_numpy(my_lat), atol_warn=1e-4)

    dpath = os.path.join(C.PKG_DIR, "audio_vae_decoder.onnx")
    print("== exporting", dpath)
    C.export_onnx(
        dec, (lat,), dpath,
        input_names=["latents"], output_names=["wav"],
        dynamic_axes={"latents": {2: "T"}, "wav": {2: "N"}},
    )
    print(f"   wrote {dpath} ({os.path.getsize(dpath)/1e6:.0f} MB, fp32)")

    epath = os.path.join(C.PKG_DIR, "audio_vae_encoder.onnx")
    print("== exporting", epath)
    C.export_onnx(
        enc, (wav_in,), epath,
        input_names=["wav"], output_names=["latents"],
        dynamic_axes={"wav": {2: "S"}, "latents": {2: "T"}},
    )
    print(f"   wrote {epath} ({os.path.getsize(epath)/1e6:.0f} MB, fp32)")

    s = C.ort_session(dpath)
    o = s.run(None, {"latents": C.to_numpy(lat).astype(np.float32)})[0]
    C.compare("vae_decode_onnx_vs_torch", C.to_numpy(my_wav), o)
    o2 = s.run(None, {"latents": C.to_numpy(lat[:, :, :2]).astype(np.float32)})[0]
    C.compare("vae_decode_onnx_T2", C.to_numpy(dec(lat[:, :, :2])), o2)

    se = C.ort_session(epath)
    oe = se.run(None, {"wav": C.to_numpy(wav_in).astype(np.float32)})[0]
    C.compare("vae_encode_onnx_vs_torch", C.to_numpy(my_lat), oe)
    print("audio_vae export done.")


def main():
    ap = argparse.ArgumentParser(description="Export VoxCPM AudioVAE encoder/decoder to ONNX (fp32).")
    C.add_io_args(ap)
    args = ap.parse_args()
    C.apply_io_args(args)
    run()
    C.gate_and_exit()


if __name__ == "__main__":
    main()
