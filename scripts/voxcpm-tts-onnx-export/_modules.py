# Copyright (c) 2026
# ONNX-friendly wrapper modules for VoxCPM-0.5B.
#
# Why rewritten forwards instead of tracing the originals:
#   * MiniCPMAttention uses SDPA with enable_gqa=True (fragile ONNX lowering) and an
#     in-place StaticKVCache (scatter writes, fixed max_len). Here: manual attention
#     (GQA via grouped broadcast matmul, no repeated-KV materialisation) + standard
#     past/present KV concat with a dynamic sequence axis, one weight copy serving
#     prefill AND decode (qwen3-tts pattern).
#   * The CFM Euler loop (10 steps, CFG batch=2, cfg-zero-star) is data-independent ->
#     unrolled into a single static graph; the initial noise is a graph INPUT so the
#     host fully controls randomness (parity + reproducibility).
#   * AudioVAE carries weight_norm (folded before export) and a @torch.jit.script snake
#     (swapped for a plain equivalent: inputs are always 3D so the reshape is a no-op).
#
# Shape legend: B=1, S=new tokens, P=past tokens, H=1024, Dh=64 (head), L=layers,
# KV=2 (kv heads), Pz=2 (patch), Dz=64 (latent feat).
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = torch.finfo(torch.float32).min


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    """Computed in fp32 exactly like modules/minicpm4/model.py (cos/sin buffers may be
    fp16 after a native half trace -> re-float explicitly)."""
    q32, k32 = q.float(), k.float()
    cos32, sin32 = cos.float(), sin.float()
    qe = q32 * cos32 + _rotate_half(q32) * sin32
    ke = k32 * cos32 + _rotate_half(k32) * sin32
    return qe.to(q.dtype), ke.to(k.dtype)


# NOTE: GQA no longer materialises repeated KV heads (the old _repeat_kv
# expand->flatten copied [B,KV*n_rep,T,Dh] per layer per step, a memcpy that grows
# linearly with past length). Attention now uses grouped BROADCAST MatMul:
# q viewed as [B,KV,n_rep,S,Dh] against k/v [B,KV,1,T,Dh] -- identical dot products
# in identical order (bit-exact vs the expand version), zero KV duplication.


def causal_mask_with_past(inputs_embeds, attention_mask):
    """Additive fp32 mask [B,1,S,P+S]: causal over the new S block, past fully visible.

    Built with the cumsum trick (no .size() -> stays dynamic under the legacy tracer);
    attention_mask is an all-ones [B,P+S] length carrier. Same construction as the
    qwen3-tts talker export."""
    ref = inputs_embeds[:, :, 0].float()  # [B,S]
    s_idx = torch.cumsum(torch.ones_like(ref), dim=1) - 1  # 0..S-1
    k_ones = torch.ones_like(attention_mask, dtype=torch.float32)  # [B,P+S]
    k_idx = torch.cumsum(k_ones, dim=1) - 1  # 0..P+S-1
    s_len = s_idx[:, -1:] + 1
    t_len = k_idx[:, -1:] + 1
    q_abs = (t_len - s_len) + s_idx  # absolute query index
    allowed = k_idx[:, None, :] <= q_abs[:, :, None]  # [B,S,P+S]
    zero = torch.zeros((), dtype=torch.float32, device=inputs_embeds.device)
    neg = torch.full((), NEG_INF, dtype=torch.float32, device=inputs_embeds.device)
    return torch.where(allowed, zero, neg)[:, None]


def minicpm_layers(model, h, cos, sin, mask=None, past_k=None, past_v=None, static_seq=None):
    """Manual MiniCPMModel forward (use_mup=False residual adds, fp32-safe softmax).

    static_seq: sequence length known at trace time (locenc=3 / dit=5, dynamic batch);
                None -> batch is static (B=1), sequence dynamic (the LM case).
    Returns (hidden, [k]*L, [v]*L) with k/v [B,KV,P+S,Dh]."""
    cfg = model.config
    nh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    dh = cfg.hidden_size // nh if cfg.kv_channels is None else cfg.kv_channels
    groups = nh // nkv
    scale = dh**-0.5
    B = h.size(0)  # static by construction in every graph we export

    pks, pvs = [], []
    for i, layer in enumerate(model.layers):
        attn = layer.self_attn
        res = h
        x = layer.input_layernorm(h)
        q, k, v = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
        if static_seq is not None:
            q = q.reshape(-1, static_seq, nh, dh).transpose(1, 2)
            k = k.reshape(-1, static_seq, nkv, dh).transpose(1, 2)
            v = v.reshape(-1, static_seq, nkv, dh).transpose(1, 2)
        else:
            q = q.reshape(B, -1, nh, dh).transpose(1, 2)
            k = k.reshape(B, -1, nkv, dh).transpose(1, 2)
            v = v.reshape(B, -1, nkv, dh).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        if past_k is not None:
            k = torch.cat([past_k[i], k], dim=2)
            v = torch.cat([past_v[i], v], dim=2)
        pks.append(k)
        pvs.append(v)
        if groups == 1:
            kk, vv, m = k, v, mask                        # full MHA: plain 4D path
        else:
            # grouped broadcast: q [B,KV,g,S,Dh], k/v [B,KV,1,T,Dh], mask [B,1,1,S,T]
            q = (q.reshape(B, nkv, groups, -1, dh) if static_seq is None
                 else q.reshape(-1, nkv, groups, static_seq, dh))
            kk = k.unsqueeze(2)
            vv = v.unsqueeze(2)
            m = None if mask is None else mask.unsqueeze(1)
        # scores in fp32 when the model runs fp16 (overflow safety, qwen3-tts learning);
        # identity for the fp32 master graphs.
        if q.dtype == torch.float16:
            scores = torch.matmul(q.float(), kk.float().transpose(-1, -2)) * scale
        else:
            scores = torch.matmul(q, kk.transpose(-1, -2)) * scale
        if m is not None:
            scores = scores + m
        w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(vv.dtype)
        o = torch.matmul(w, vv)
        if groups != 1:                                   # [B,KV,g,S,Dh] -> [B,NH,S,Dh]
            o = (o.reshape(B, nh, -1, dh) if static_seq is None
                 else o.reshape(-1, nh, static_seq, dh))
        o = o.transpose(1, 2)
        if static_seq is not None:
            o = o.reshape(-1, static_seq, nh * dh)
        else:
            o = o.reshape(B, -1, nh * dh)
        h = res + attn.o_proj(o)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return model.norm(h), pks, pvs


def rope_table(model, n):
    """First n rows of a MiniCPMModel's LongRoPE cos/sin cache (fp32 constants).

    Sequences here are <= 4096 << original_max_position_embeddings=32768, so the
    short_factor branch (already materialised in cos_cached) always applies."""
    cos = model.rope_emb.cos_cached[:n].detach().float().clone()
    sin = model.rope_emb.sin_cached[:n].detach().float().clone()
    return cos, sin


# ----------------------------------------------------------------------------
# GQA contrib-op attention (CPU 加速变体 lm_gqa.onnx 专用)
# ----------------------------------------------------------------------------
class _GqaAttention(torch.autograd.Function):
    """Attention block that exports as ONE com.microsoft::GroupQueryAttention node.

    Why: the exported primitive attention (transpose + concat past + broadcast matmul
    + softmax) costs ~5.5ms/step at P=100 growing to ~7.5ms at P=550 on CPU (profiled:
    Transpose/Concat/Split dominate). The fused GQA CPU kernel does rope + kv-append +
    causal attention in ~0.07ms/layer at P=550 -- and takes per-layer past/present
    directly, so the stacked-KV Split/Concat copies disappear too.

    Eager forward = reference math (numerically identical to minicpm_layers, used by
    export-time parity gates); symbolic = the contrib node (do_rotary=1, non-interleaved,
    half-table cos/sin caches). fp32 only -- the fp16 package keeps the standard graph.
    """

    @staticmethod
    def forward(ctx, q, k, v, past_k, past_v, seqlens_k, total_seq, cos_cache, sin_cache,
                num_heads, kv_heads, scale):
        B, S, _ = q.shape
        dh = q.shape[2] // num_heads
        p = past_k.shape[2]
        pos = torch.arange(p, p + S, device=q.device)
        qh = q.reshape(B, S, num_heads, dh).transpose(1, 2)
        kh = k.reshape(B, S, kv_heads, dh).transpose(1, 2)
        vh = v.reshape(B, S, kv_heads, dh).transpose(1, 2)
        c, s = cos_cache[pos], sin_cache[pos]                      # [S, dh/2] 半表

        def rot(x):
            x1, x2 = x[..., : dh // 2], x[..., dh // 2:]
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

        qh, kh = rot(qh), rot(kh)
        kk = torch.cat([past_k, kh], dim=2)
        vv = torch.cat([past_v, vh], dim=2)
        t = kk.shape[2]
        g = num_heads // kv_heads
        qg = qh.reshape(B, kv_heads, g, S, dh)
        scores = torch.matmul(qg, kk.unsqueeze(2).transpose(-1, -2)) * scale
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=q.device))[t - S: t]
        scores = scores.masked_fill(~causal[None, None, None], NEG_INF)
        w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(vv.dtype)
        o = torch.matmul(w, vv.unsqueeze(2))                      # [B,KV,g,S,Dh]
        o = o.reshape(B, num_heads, S, dh).transpose(1, 2)
        return o.reshape(B, S, num_heads * dh), kk, vv

    @staticmethod
    def symbolic(g, q, k, v, past_k, past_v, seqlens_k, total_seq, cos_cache, sin_cache,
                 num_heads, kv_heads, scale):
        return g.op(
            "com.microsoft::GroupQueryAttention",
            q, k, v, past_k, past_v, seqlens_k, total_seq, cos_cache, sin_cache,
            num_heads_i=num_heads, kv_num_heads_i=kv_heads,
            do_rotary_i=1, rotary_interleaved_i=0, scale_f=scale,
            outputs=3,
        )


def minicpm_layers_gqa(model, h, seqlens_k, total_seq, cos_cache, sin_cache, past_k, past_v):
    """minicpm_layers 的 GQA contrib-op 版: past_k/past_v 为逐层张量列表
    ([B,KV,P,Dh] × L), 返回逐层 present 列表 (host 逐层回喂, 无堆叠/拆分拷贝)。"""
    cfg = model.config
    nh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    dh = cfg.hidden_size // nh if cfg.kv_channels is None else cfg.kv_channels
    scale = dh**-0.5

    pks, pvs = [], []
    for i, layer in enumerate(model.layers):
        attn = layer.self_attn
        res = h
        x = layer.input_layernorm(h)
        o, pk, pv = _GqaAttention.apply(
            attn.q_proj(x), attn.k_proj(x), attn.v_proj(x), past_k[i], past_v[i],
            seqlens_k, total_seq, cos_cache, sin_cache, nh, nkv, scale)
        pks.append(pk)
        pvs.append(pv)
        h = res + attn.o_proj(o)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return model.norm(h), pks, pvs


class LmModule(nn.Module):
    """Unified prefill/decode graph: base_lm -> FSQ mix -> residual_lm -> heads.

    Prefill: S=T, P=0, fsq_mask = audio_mask, feat_embed_masked = audio_mask*feat_embed,
             inputs_embeds = text_mask*text_embed + feat_embed_masked (host-built).
    Decode : S=1, P=n,  fsq_mask = 1, feat_embed_masked = inputs_embeds = curr_embed.

    Outputs feed the NEXT host-loop iteration: dit_hidden -> cfm, stop_logits -> stop check.
    """

    def __init__(self, vox, max_len=4096):
        super().__init__()
        self.base = vox.base_lm
        self.res = vox.residual_lm
        self.fsq_in = vox.fsq_layer.in_proj
        self.fsq_out = vox.fsq_layer.out_proj
        self.fsq_scale = float(vox.fsq_layer.scale)
        self.lm_to_dit = vox.lm_to_dit_proj
        self.res_to_dit = vox.res_to_dit_proj
        self.stop_proj = vox.stop_proj
        self.stop_head = vox.stop_head
        cos, sin = rope_table(vox.base_lm, max_len)
        self.register_buffer("cos_t", cos, persistent=False)
        self.register_buffer("sin_t", sin, persistent=False)

    def _fsq(self, h):
        h = torch.tanh(self.fsq_in(h))
        h = torch.round(h * self.fsq_scale) / self.fsq_scale
        return self.fsq_out(h)

    def forward(
        self,
        inputs_embeds,  # [1,S,H]
        feat_embed_masked,  # [1,S,H]
        fsq_mask,  # [1,S] fp32 (1=audio position)
        attention_mask,  # [1,P+S] all-ones length carrier
        position_ids,  # [1,S] int64 absolute positions (P..P+S-1)
        base_past_k,  # [Lb,1,KV,P,Dh]
        base_past_v,
        res_past_k,  # [Lr,1,KV,P,Dh]
        res_past_v,
    ):
        cos = self.cos_t[position_ids[0]]
        sin = self.sin_t[position_ids[0]]
        mask = causal_mask_with_past(inputs_embeds, attention_mask)

        h, bk, bv = minicpm_layers(self.base, inputs_embeds, cos, sin, mask, base_past_k, base_past_v)
        m = fsq_mask.unsqueeze(-1)
        hm = self._fsq(h) * m + h * (1.0 - m)

        rin = hm + feat_embed_masked
        rh, rk, rv = minicpm_layers(self.res, rin, cos, sin, mask, res_past_k, res_past_v)

        lm_hidden = hm[:, -1, :]
        residual_hidden = rh[:, -1, :]
        dit_hidden = self.lm_to_dit(lm_hidden) + self.res_to_dit(residual_hidden)
        stop_logits = self.stop_head(F.silu(self.stop_proj(lm_hidden)))
        return (
            dit_hidden,
            stop_logits,
            lm_hidden,
            residual_hidden,
            torch.stack(bk, 0),
            torch.stack(bv, 0),
            torch.stack(rk, 0),
            torch.stack(rv, 0),
        )


class LmGqaModule(LmModule):
    """LmModule 的 GQA contrib-op 变体 (lm_gqa.onnx, CPU 专用加速图)。

    与 LmModule 数学等价 (非比特一致: 融合 kernel 归约顺序不同), 接口差异:
      * attention_mask / position_ids -> seqlens_k [1] (int32, =P+S-1) +
        total_sequence_length (int32 标量, =P+S); 因果 mask 与 rope 位置在 op 内部处理
      * KV 逐层输入/输出 (base 24 + residual 6, 各 k/v [1,KV,P,Dh]) —— 免去堆叠
        KV 每步的 Split/Concat 全量拷贝
    """

    def __init__(self, vox, max_len=4096):
        super().__init__(vox, max_len)
        dh = vox.config.lm_config.hidden_size // vox.config.lm_config.num_attention_heads
        cos, sin = self.cos_t, self.sin_t
        # rotate_half 形式的全维表 = 半表复制两份; GQA 的 cos/sin cache 取半表
        assert torch.equal(cos[:, : dh // 2], cos[:, dh // 2:]), "rope 表不是复制半表"
        assert torch.equal(sin[:, : dh // 2], sin[:, dh // 2:]), "rope 表不是复制半表"
        self.register_buffer("cos_half", cos[:, : dh // 2].contiguous(), persistent=False)
        self.register_buffer("sin_half", sin[:, : dh // 2].contiguous(), persistent=False)

    def forward(
        self,
        inputs_embeds,            # [1,S,H]
        feat_embed_masked,        # [1,S,H]
        fsq_mask,                 # [1,S] fp32
        seqlens_k,                # [1] int32 = P+S-1
        total_sequence_length,    # int32 标量 = P+S
        base_past_k,              # tuple(Lb × [1,KV,P,Dh])
        base_past_v,
        res_past_k,               # tuple(Lr × [1,KV,P,Dh])
        res_past_v,
    ):
        h, bk, bv = minicpm_layers_gqa(self.base, inputs_embeds, seqlens_k, total_sequence_length,
                                       self.cos_half, self.sin_half, base_past_k, base_past_v)
        m = fsq_mask.unsqueeze(-1)
        hm = self._fsq(h) * m + h * (1.0 - m)

        rin = hm + feat_embed_masked
        rh, rk, rv = minicpm_layers_gqa(self.res, rin, seqlens_k, total_sequence_length,
                                        self.cos_half, self.sin_half, res_past_k, res_past_v)

        lm_hidden = hm[:, -1, :]
        residual_hidden = rh[:, -1, :]
        dit_hidden = self.lm_to_dit(lm_hidden) + self.res_to_dit(residual_hidden)
        stop_logits = self.stop_head(F.silu(self.stop_proj(lm_hidden)))
        return (dit_hidden, stop_logits, lm_hidden, residual_hidden, *bk, *bv, *rk, *rv)


class FeatEncoderModule(nn.Module):
    """feat [1,T,Pz,Dz] -> feat_embed [1,T,H]  (VoxCPMLocEnc + enc_to_lm_proj).

    T is dynamic: T=prompt patches at prefill, T=1 per decode step. The 4-layer local
    encoder runs non-causally over static (1+Pz)=3-token sequences with batch T."""

    def __init__(self, vox):
        super().__init__()
        self.enc = vox.feat_encoder
        self.proj = vox.enc_to_lm_proj
        pz = vox.patch_size
        cos, sin = rope_table(vox.feat_encoder.encoder, pz + 1)
        self.register_buffer("cos_t", cos, persistent=False)
        self.register_buffer("sin_t", sin, persistent=False)
        self.seq = pz + 1

    def forward(self, feat):
        x = self.enc.in_proj(feat)  # [1,T,Pz,H]
        sp = torch.zeros_like(x[:, :, :1, :]) + self.enc.special_token  # [1,T,1,H]
        x = torch.cat([sp, x], dim=2).flatten(0, 1)  # [T,3,H]
        h, _, _ = minicpm_layers(self.enc.encoder, x, self.cos_t, self.sin_t, static_seq=self.seq)
        cls = h[:, 0, :].unsqueeze(0)  # [1,T,H]
        return self.proj(cls)


class CfmDecodeModule(nn.Module):
    """One full CFM patch generation, Euler loop unrolled (solve_euler semantics).

    n_timesteps Euler steps over the sway-warped t-span; step 1 is the cfg-zero-star
    zero-init (x stays = noise), steps 2..n call the DiT estimator with CFG batch=2
    and the optimized-scale combine. dt into the estimator is always zero (mean_mode
    is False for this checkpoint), matching UnifiedCFM.forward defaults:
    temperature=1.0, sway_sampling_coef=1.0, use_cfg_zero_star=True."""

    def __init__(self, vox, n_timesteps=10):
        super().__init__()
        self.est = vox.feat_decoder.estimator
        self.n_timesteps = int(n_timesteps)
        pz = vox.patch_size
        seq = 1 + pz + pz
        cos, sin = rope_table(self.est.decoder, seq)
        self.register_buffer("cos_t", cos, persistent=False)
        self.register_buffer("sin_t", sin, persistent=False)
        self.seq = seq
        self.prefix = pz
        t_span = torch.linspace(1, 0, self.n_timesteps + 1, dtype=torch.float64)
        t_span = t_span + 1.0 * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        self.t_span = [float(t) for t in t_span]
        self.zero_init_steps = max(1, int((self.n_timesteps + 1) * 0.04))

    def _estimator(self, x, mu, t, cond_in):
        e = self.est
        xi = e.in_proj(x.transpose(1, 2))  # [2,Pz,H]
        ci = e.cond_proj(cond_in.transpose(1, 2))  # [2,Pz,H]
        # sinusoidal args reach ~1000*t -> compute in fp32 even on a half trace
        temb = e.time_mlp(e.time_embeddings(t.float()).to(xi.dtype))
        dtemb = e.delta_time_mlp(e.time_embeddings(torch.zeros_like(t).float()).to(xi.dtype))
        temb = temb + dtemb
        seq = torch.cat([(mu + temb).unsqueeze(1), ci, xi], dim=1)  # [2,5,H]
        h, _, _ = minicpm_layers(e.decoder, seq, self.cos_t, self.sin_t, static_seq=self.seq)
        h = h[:, self.prefix + 1 :, :]
        return e.out_proj(h).transpose(1, 2)  # [2,Dz,Pz]

    def forward(self, dit_hidden, cond, noise, cfg_value):
        # dit_hidden [1,H], cond/noise [1,Dz,Pz], cfg_value [1]
        x = noise  # step 1..zero_init_steps: dphi=0 -> x unchanged, t advances
        mu_in = torch.cat([dit_hidden, torch.zeros_like(dit_hidden)], dim=0)  # [2,H]
        cond_in = torch.cat([cond, cond], dim=0)  # [2,Dz,Pz]
        for step in range(self.zero_init_steps + 1, self.n_timesteps + 1):
            t_k = self.t_span[step - 1]
            dt_k = self.t_span[step - 1] - self.t_span[step]
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.full((2,), t_k, dtype=torch.float32, device=x.device)
            dphi = self._estimator(x_in, mu_in, t_in, cond_in)
            # cfg-zero-star combine in fp32 (sums are reduction-precision sensitive)
            pos, neg = dphi[0:1].float(), dphi[1:2].float()
            dot = (pos * neg).sum(dim=(1, 2), keepdim=True)
            sq = (neg * neg).sum(dim=(1, 2), keepdim=True) + 1e-8
            st = dot / sq
            comb = neg * st + cfg_value.float() * (pos - neg * st)
            x = x - dt_k * comb.to(x.dtype)
        return x


def _snake_plain(x, alpha):
    # exact equivalent of the @torch.jit.script snake for 3D [B,C,T] inputs
    return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


def prepare_vae_for_export(vox):
    """Fold weight_norm into plain weights and swap the scripted snake. Idempotent-ish:
    call once on the shared master copy before exporting either VAE graph."""
    from voxcpm.modules.audiovae import audio_vae as AV

    AV.snake = _snake_plain  # Snake1d.forward resolves `snake` at module-global scope
    folded = 0
    for m in vox.audio_vae.modules():
        try:
            torch.nn.utils.remove_weight_norm(m)
            folded += 1
        except (ValueError, AttributeError):
            pass
    return folded


class VaeDecoderModule(nn.Module):
    """latents [1,Dz,T] -> wav [1,1,T*chunk]; always fp32."""

    def __init__(self, vox):
        super().__init__()
        self.dec = vox.audio_vae.decoder

    def forward(self, latents):
        return self.dec(latents)


class VaeEncoderModule(nn.Module):
    """wav [1,1,S] (S % chunk == 0, host pads) -> latents [1,Dz,S/chunk]; always fp32."""

    def __init__(self, vox):
        super().__init__()
        self.enc = vox.audio_vae.encoder

    def forward(self, wav):
        return self.enc(wav)["mu"]


# ----------------------------------------------------------------------------
# Host-side prefill input construction (shared by parity/golden and _engine)
# ----------------------------------------------------------------------------
def build_prefill_arrays(text_ids, prompt_feat, patch_size, feat_dim, audio_start):
    """Replicates VoxCPMModel._generate_with_prompt_cache input packing with numpy.

    text_ids: list[int] target(+prompt) token ids WITHOUT audio_start (appended here).
    prompt_feat: [Tp,Pz,Dz] float32 or None.
    audio_start: audio_start token id (from the manifest / vox.audio_start_token).
    Returns dict(text_ids[1,T], text_mask[1,T], fsq_mask[1,T], feat[1,T,Pz,Dz])."""
    import numpy as np

    ids = list(text_ids) + [audio_start]
    tlen = len(ids)
    plen = 0 if prompt_feat is None else int(prompt_feat.shape[0])
    total = tlen + plen
    out_ids = np.zeros((1, total), dtype=np.int64)
    out_ids[0, :tlen] = ids
    text_mask = np.zeros((1, total), dtype=np.float32)
    text_mask[0, :tlen] = 1.0
    fsq_mask = 1.0 - text_mask  # == audio_mask
    feat = np.zeros((1, total, patch_size, feat_dim), dtype=np.float32)
    if plen:
        feat[0, tlen:] = prompt_feat
    return dict(text_ids=out_ids, text_mask=text_mask, fsq_mask=fsq_mask, feat=feat)
