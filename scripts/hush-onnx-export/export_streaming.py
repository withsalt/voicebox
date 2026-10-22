#!/usr/bin/env python3
"""Export Hush (DeepFilterNet3-SE, 16 kHz) to a fused streaming ONNX with
externalized GRU states, matching the I/O contract of
VoiceBox.DNR.DeepFilterNet's DeepFilterNetOnnxRunner (combined_streaming.onnx).

Streaming contract (one 10 ms frame per call):
  inputs : feat_erb [1,1,T,nb_erb]      ERB features, T = encoder conv context frames
           feat_spec[1,2,T,nb_df]       complex spec features (planar re/im channels)
           h0       [1,1,H]             encoder GRU state
           erb_h0   [Le,1,H]            ERB decoder GRU state
           df_h0    [Ld,1,H]            DF decoder GRU state
  outputs: lsnr     [1,1,1]
           m        [1,1,1,nb_erb]      ERB gain mask (sigmoid)
           coefs    [1,1,nb_df,O*2]     deep-filter coefficients (re/im interleaved per order)
           h1, erb_h1, df_h1            updated GRU states

The convs are causal (ConstantPad2d top=k_t-1); feeding T>=conv_kernel_inp[0]
frames and taking only the last time step yields output bit-identical to the
offline whole-sequence model, provided the GRU state is carried across frames.

Usage:
    python export_streaming.py
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor, nn

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = r"E:/Temp/hush/model_best.ckpt"
OUT_ONNX = os.path.join(HERE, "outputs", "combined_streaming.onnx")
OPSET = 17

from dfnet_se import DfNetSE, get_config


def run_sgru(sgru, x: Tensor, h: Tensor):
    """SqueezedGRU_S.forward but threading an explicit hidden state in/out."""
    y = sgru.linear_in(x)
    y, h = sgru.gru(y, h)
    y = sgru.linear_out(y)
    if sgru.gru_skip is not None:
        y = y + sgru.gru_skip(x)
    return y, h


class StreamingDfNet(nn.Module):
    """Single-frame streaming wrapper over a loaded DfNet (model.model)."""

    def __init__(self, dfnet: nn.Module):
        super().__init__()
        self.enc = dfnet.enc
        self.erb_dec = dfnet.erb_dec
        self.df_dec = dfnet.df_dec
        self.lsnr_scale = dfnet.enc.lsnr_scale
        self.lsnr_offset = dfnet.enc.lsnr_offset

    def forward(self, feat_erb: Tensor, feat_spec: Tensor,
                h0: Tensor, erb_h0: Tensor, df_h0: Tensor):
        enc = self.enc
        # --- encoder conv stack over T frames (causal pad inside conv0) ---
        e0 = enc.erb_conv0(feat_erb)
        e1 = enc.erb_conv1(e0)
        e2 = enc.erb_conv2(e1)
        e3 = enc.erb_conv3(e2)
        c0 = enc.df_conv0(feat_spec)
        c1 = enc.df_conv1(c0)

        cemb = c1.permute(0, 2, 3, 1).flatten(2)        # [B,T,conv_ch*nb_df//2]
        cemb = enc.df_fc_emb(cemb)                       # [B,T,emb_in_dim]
        emb = e3.permute(0, 2, 3, 1).flatten(2)          # [B,T,emb_in_dim]
        emb = enc.combine(emb, cemb)

        # --- keep only the last (causal) frame from here on ---
        emb = emb[:, -1:, :]
        e0 = e0[:, :, -1:, :]
        e1 = e1[:, :, -1:, :]
        e2 = e2[:, :, -1:, :]
        e3 = e3[:, :, -1:, :]
        c0 = c0[:, :, -1:, :]

        # --- encoder GRU (external state) + lsnr ---
        emb, h1 = run_sgru(enc.emb_gru, emb, h0)         # [B,1,emb_out_dim]
        lsnr = enc.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset

        # --- ERB decoder (external state) -> mask ---
        dec = self.erb_dec
        b, _, t, f8 = e3.shape
        edec, erb_h1 = run_sgru(dec.emb_gru, emb, erb_h0)
        edec = edec.view(b, t, f8, -1).permute(0, 3, 1, 2)
        x3 = dec.convt3(dec.conv3p(e3) + edec)
        x2 = dec.convt2(dec.conv2p(e2) + x3)
        x1 = dec.convt1(dec.conv1p(e1) + x2)
        m = dec.conv0_out(dec.conv0p(e0) + x1)           # [B,1,1,nb_erb]

        # --- DF decoder (external state) -> coefs ---
        dfd = self.df_dec
        c, df_h1 = run_sgru(dfd.df_gru, emb, df_h0)
        if dfd.df_skip is not None:
            c = c + dfd.df_skip(emb)
        cp = dfd.df_convp(c0).permute(0, 2, 3, 1)         # [B,1,nb_df,O*2]
        c = dfd.df_out(c)
        c = c.view(b, t, dfd.df_bins, dfd.df_out_ch) + cp  # [B,1,nb_df,O*2]

        return lsnr, m, c, h1, erb_h1, df_h1


def load_model():
    cfg = get_config()
    model = DfNetSE(cfg)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    try:
        model.model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def gru_dims(dfnet):
    H = dfnet.enc.emb_gru.gru.hidden_size
    Le = dfnet.erb_dec.emb_gru.gru.num_layers
    Ld = dfnet.df_dec.df_gru.gru.num_layers
    return H, Le, Ld


def export(model):
    cfg = model.config
    wrapper = StreamingDfNet(model.model).eval()
    H, Le, Ld = gru_dims(model.model)
    T = cfg.conv_kernel_inp[0]  # encoder temporal receptive field (3)

    # StreamingDfNet keeps only the last time frame after the encoder. This is exact ONLY
    # when every decoder path is temporally pointwise (kernel_t == 1) and df_lookahead == 0
    # (encoder convs are causal). Guard against a re-export from a config that breaks this,
    # which would otherwise silently drop temporal context and produce wrong masks/coefs.
    kt_conv = tuple(cfg.conv_kernel)[0]
    kt_convt = tuple(cfg.convt_kernel)[0]
    assert kt_conv == 1 and kt_convt == 1 and cfg.df_pathway_kernel_size_t == 1, (
        "single-frame streaming export requires temporally-pointwise decoders "
        f"(conv_kernel_t={kt_conv}, convt_kernel_t={kt_convt}, "
        f"df_pathway_kernel_size_t={cfg.df_pathway_kernel_size_t}); got a config needing >1 frame of context."
    )
    assert cfg.df_lookahead == 0 and cfg.conv_lookahead == 0, (
        f"streaming export assumes zero lookahead (df_lookahead={cfg.df_lookahead}, "
        f"conv_lookahead={cfg.conv_lookahead})."
    )

    feat_erb = torch.zeros(1, 1, T, cfg.nb_erb)
    feat_spec = torch.zeros(1, 2, T, cfg.nb_df)
    h0 = torch.zeros(1, 1, H)
    erb_h0 = torch.zeros(Le, 1, H)
    df_h0 = torch.zeros(Ld, 1, H)

    os.makedirs(os.path.dirname(OUT_ONNX), exist_ok=True)
    torch.onnx.export(
        wrapper,
        (feat_erb, feat_spec, h0, erb_h0, df_h0),
        OUT_ONNX,
        input_names=["feat_erb", "feat_spec", "h0", "erb_h0", "df_h0"],
        output_names=["lsnr", "m", "coefs", "h1", "erb_h1", "df_h1"],
        dynamic_axes={
            "feat_erb": {2: "T"},
            "feat_spec": {2: "T"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported -> {OUT_ONNX}")
    print(f"  dims: H={H} Le={Le} Ld={Ld} T={T} nb_erb={cfg.nb_erb} nb_df={cfg.nb_df} O={cfg.df_order}")
    return H, Le, Ld, T


if __name__ == "__main__":
    m = load_model()
    export(m)
