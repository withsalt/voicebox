# Copyright (c) 2026
# Export the VoxCPM local-DiT flow-matching sampler as ONE fused static graph:
# the whole 10-step Euler solve (CFG batch=2, cfg-zero-star, sway t-span) per patch.
#
#   cfm_decode.onnx
#     in : dit_hidden[1,H], cond[1,Dz,Pz], noise[1,Dz,Pz], cfg_value[1]
#     out: pred_feat[1,Dz,Pz]
#
# Noise is a graph INPUT (host-controlled RNG -> reproducible parity across
# torch / python-ORT / C#). All shapes static -> ORT-friendly hot path
# (one Run per 80 ms audio patch instead of 9 estimator Runs).
import os
import argparse

import numpy as np
import torch

import _common as C
from _modules import CfmDecodeModule


def run(vox=None):
    if vox is None:
        print("== loading model (fp32) for cfm ...")
        vox = C.load_vox()
    n_steps = int(C.GENERATION_DEFAULTS["inference_timesteps"])  # baked in; manifest 同源
    mod = CfmDecodeModule(vox, n_timesteps=n_steps).eval()

    g = torch.Generator().manual_seed(2)
    H, Dz, Pz = vox.config.lm_config.hidden_size, vox.feat_dim, vox.patch_size
    mu = torch.randn(1, H, generator=g) * 0.5
    cond = torch.randn(1, Dz, Pz, generator=g) * 0.5
    noise = torch.randn(1, Dz, Pz, generator=g)
    cfg = torch.tensor([2.0], dtype=torch.float32)

    # reference: original solve_euler with the exact same z / t_span
    with torch.inference_mode():
        t_span = torch.linspace(1, 0, n_steps + 1, dtype=torch.float32)
        t_span = t_span + 1.0 * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        ref = vox.feat_decoder.solve_euler(
            x=noise.clone(), t_span=t_span, mu=mu, cond=cond, cfg_value=2.0, use_cfg_zero_star=True
        )
        mine = mod(mu, cond, noise, cfg)
    C.compare("cfm_vs_orig_solve_euler", C.to_numpy(ref), C.to_numpy(mine), atol_warn=1e-4)

    path = os.path.join(C.PKG_DIR, "cfm_decode.onnx")
    print("== exporting", path)
    exp_mod, ex = mod, (mu, cond, noise, cfg)
    if C.DTYPE == "fp16":
        exp_mod = mod.half().eval()
        ex = (mu.half(), cond.half(), noise.half(), cfg.half())
    C.export_onnx(
        exp_mod,
        ex,
        path,
        input_names=["dit_hidden", "cond", "noise", "cfg_value"],
        output_names=["pred_feat"],
        dynamic_axes={},  # fully static
    )
    if C.DTYPE == "int8":
        # 扩散采样器权重 int8: cfm 是量化包每步带宽大头 (258MB × 9 evals ≈ 2.3GB/步),
        # int8 后带宽降 4x (37→20ms/步)。8bit 而非 4bit: 10 步 Euler 误差逐步复合,
        # 扩散对量化最敏感 (int4 复合 SNR 仅 9dB, 不可用)。
        C.postprocess_dtype(path, "int8")
        s8 = C.ort_session(path)
        o8 = s8.run(None, {
            "dit_hidden": C.to_numpy(mu).astype(np.float32),
            "cond": C.to_numpy(cond).astype(np.float32),
            "noise": C.to_numpy(noise).astype(np.float32),
            "cfg_value": np.array([2.0], dtype=np.float32),
        })[0]
        # 量化质量门禁 (单 patch, 10 步复合后): int8 权重 + acc4 激活量化实测 ~28dB;
        # 低于 22dB 视为量化失败。轨迹级差异由 parity smoke + 听感把关。
        C.compare("cfm_int8_vs_torch_fp32", C.to_numpy(mine), o8,
                  atol_warn=1e-1, min_snr=22.0, min_cos=0.99)
        del s8
    print(f"   wrote {path} ({os.path.getsize(path)/1e6:.0f} MB)")

    if C.DTYPE == "fp32":
        s = C.ort_session(path)
        o = s.run(None, {
            "dit_hidden": C.to_numpy(mu).astype(np.float32),
            "cond": C.to_numpy(cond).astype(np.float32),
            "noise": C.to_numpy(noise).astype(np.float32),
            "cfg_value": np.array([2.0], dtype=np.float32),
        })[0]
        C.compare("cfm_onnx_vs_torch", C.to_numpy(mine), o)
        # cfg sensitivity smoke: different cfg must change the output
        o15 = s.run(None, {
            "dit_hidden": C.to_numpy(mu).astype(np.float32),
            "cond": C.to_numpy(cond).astype(np.float32),
            "noise": C.to_numpy(noise).astype(np.float32),
            "cfg_value": np.array([1.5], dtype=np.float32),
        })[0]
        assert float(np.abs(o - o15).max()) > 1e-4, "cfg_value input has no effect"
    print("cfm export done.")


def main():
    ap = argparse.ArgumentParser(description="Export VoxCPM cfm_decode.onnx (fused Euler solve).")
    C.add_io_args(ap)
    args = ap.parse_args()
    C.apply_io_args(args)
    run()
    C.gate_and_exit()


if __name__ == "__main__":
    main()
