# Copyright (c) 2026
# Pure numpy + ONNXRuntime reference engine for the exported VoxCPM package.
# Mirrors VoxCPMModel._generate_with_prompt_cache / _inference step by step and is the
# executable spec for the C# pipeline (VoiceBox.TTS.VoxCPM.Internal.VoxCpmPipeline):
#
#   prefill: tokenize(+zh split)+audio_start -> embeds -> lm.onnx (P=0)
#   loop   : cfm_decode.onnx (noise from host RNG) -> feat_encoder.onnx (T=1)
#            -> stop check -> lm.onnx (S=1)
#   output : audio_vae_decoder.onnx (final decode, or 3-patch sliding window when
#            streaming: decode last 3 patches, keep the trailing 1280 samples)
#
# Noise per step: one [1,64,2] standard normal from a seeded numpy Generator --
# the SAME sequence the C# GaussianNoise must produce for golden parity.
import os
import json

import numpy as np


class VoxCpmOnnxEngine:
    def __init__(self, pkg_dir, providers=None):
        self.pkg = pkg_dir
        with open(os.path.join(pkg_dir, "config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        import onnxruntime as ort
        from tokenizers import Tokenizer

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        pv = providers or ["CPUExecutionProvider"]
        ses = lambda n: ort.InferenceSession(os.path.join(pkg_dir, self.cfg["files"][n]), so, providers=pv)
        self.lm = ses("lm")
        self.fe = ses("feat_encoder")
        self.cfm = ses("cfm_decode")
        self.vae_dec = ses("audio_vae_decoder")
        self.vae_enc = ses("audio_vae_encoder")

        self.tok = Tokenizer.from_file(os.path.join(pkg_dir, self.cfg["files"]["tokenizer"]))
        with open(os.path.join(pkg_dir, self.cfg["files"]["multichar_zh"]), encoding="utf-8") as f:
            self.zh_split = {int(k): v for k, v in json.load(f).items()}
        eshape = self.cfg["text_embedding_shape"]
        self.emb = np.fromfile(
            os.path.join(pkg_dir, self.cfg["files"]["text_embedding"]), dtype=np.float32
        ).reshape(eshape)

        g = self.cfg["generation"]
        self.patch = self.cfg["patch_size"]
        self.feat_dim = self.cfg["feat_dim"]
        self.hidden = self.cfg["hidden_size"]
        self.chunk = self.cfg["vae_chunk_size"]
        self.patch_len = self.patch * self.chunk
        self.audio_start = self.cfg["audio_start_token"]
        self.cfg_value = float(g["cfg_value"])
        self.min_len = int(g["min_len"])
        self.ratio = float(g["max_len_text_ratio"])
        self.bias = int(g["max_len_text_bias"])
        self.max_length = int(self.cfg["max_length"])
        self.stream_prefix = int(g["streaming_prefix_patches"])
        # graphs may be fp16 -> feed/read accordingly
        self.lm_dtype = np.float16 if self.lm.get_inputs()[0].type == "tensor(float16)" else np.float32
        self.fe_dtype = np.float16 if self.fe.get_inputs()[0].type == "tensor(float16)" else np.float32
        self.cfm_dtype = np.float16 if self.cfm.get_inputs()[0].type == "tensor(float16)" else np.float32

    # ------------------------------------------------------------------ #
    def tokenize(self, text):
        """tokenizer.json BPE + multi-char CJK split (mask_multichar_chinese_tokens port).

        NO special tokens: the python wrapper goes through tokenizer.tokenize()
        (add_special_tokens=False), so no bos despite add_bos_token=true in the
        tokenizer config. C# must match this."""
        ids = self.tok.encode(text, add_special_tokens=False).ids
        out = []
        for tid in ids:
            out.extend(self.zh_split.get(tid, [tid]))
        return out

    def encode_prompt(self, wav):
        """wav float32 [S] 16 kHz -> prompt feat [Tp, patch, feat_dim] (left-pad like torch)."""
        n = wav.shape[0]
        rem = n % self.patch_len
        if rem:
            wav = np.concatenate([np.zeros(self.patch_len - rem, dtype=np.float32), wav])
        lat = self._vae_encode(wav)
        # [1,Dz,T] -> view(Dz,-1,patch).permute(1,2,0) == transpose chain below
        lat = lat[0]  # [Dz,T]
        tp = lat.shape[1] // self.patch
        return lat.reshape(self.feat_dim, tp, self.patch).transpose(1, 2, 0).copy()

    def _vae_encode(self, wav):
        return self.vae_enc.run(None, {"wav": wav.reshape(1, 1, -1).astype(np.float32)})[0]

    # ------------------------------------------------------------------ #
    def _feat_embed(self, feat):
        """feat [1,T,patch,feat_dim] -> [1,T,H] via feat_encoder.onnx."""
        return self.fe.run(None, {"feat": feat.astype(self.fe_dtype)})[0].astype(np.float32)

    def _empty_past(self, layers):
        kv = self.cfg["num_key_value_heads"]
        d = self.cfg["head_dim"]
        z = np.zeros((layers, 1, kv, 0, d), dtype=self.lm_dtype)
        return z, z.copy()

    def _lm(self, embeds, fem, fsq_mask, pos_start, past):
        S = embeds.shape[1]
        P = past[0].shape[3]
        feeds = {
            "inputs_embeds": embeds.astype(self.lm_dtype),
            "feat_embed_masked": fem.astype(self.lm_dtype),
            "fsq_mask": fsq_mask.astype(self.lm_dtype),
            "attention_mask": np.ones((1, P + S), dtype=np.int64),
            "position_ids": np.arange(pos_start, pos_start + S, dtype=np.int64).reshape(1, S),
            "base_past_k": past[0], "base_past_v": past[1],
            "res_past_k": past[2], "res_past_v": past[3],
        }
        o = self.lm.run(None, feeds)
        return (
            o[0].astype(np.float32),  # dit_hidden [1,H]
            o[1].astype(np.float32),  # stop_logits [1,2]
            (o[4], o[5], o[6], o[7]),
        )

    # ------------------------------------------------------------------ #
    def generate(self, text, prompt_text=None, prompt_wav=None, seed=1234,
                 max_len=None, streaming=False, collect_latents=False):
        """Returns dict(wav[S], latents[T*patch? optional], steps).

        streaming=True additionally returns chunks: list of 1280-sample arrays
        (3-patch window decode, trailing patch kept) exactly like the torch path."""
        full_text = (prompt_text or "") + text
        ids = self.tokenize(full_text)
        target_len = len(self.tokenize(text))
        prompt_feat = self.encode_prompt(prompt_wav) if prompt_wav is not None else None

        from _modules import build_prefill_arrays

        arrs = build_prefill_arrays(ids, prompt_feat, self.patch, self.feat_dim, self.audio_start)
        T = arrs["text_ids"].shape[1]
        if T >= self.max_length:
            raise ValueError(f"input too long: prefill {T} >= max_length {self.max_length}")
        limit = min(int(target_len * self.ratio + self.bias), max_len or self.max_length)
        # lm.onnx 的 RoPE 表只有 max_length 行: prefill + steps 必须不越界
        limit = max(1, min(limit, self.max_length - T))

        text_embed = self.emb[arrs["text_ids"][0]][None]  # [1,T,H]
        feat_embed = self._feat_embed(arrs["feat"])
        tm = arrs["text_mask"][..., None]
        am = arrs["fsq_mask"][..., None]
        combined = tm * text_embed + am * feat_embed
        fem = am * feat_embed

        bpk, bpv = self._empty_past(self.cfg["base_lm_layers"])
        rpk, rpv = self._empty_past(self.cfg["residual_lm_layers"])
        dit_hidden, stop_logits, past = self._lm(
            combined, fem, arrs["fsq_mask"], 0, (bpk, bpv, rpk, rpv)
        )

        rng = np.random.default_rng(seed)
        cond = arrs["feat"][:, -1].transpose(0, 2, 1).copy()  # [1,Dz,patch] (last patch, zeros w/o prompt)
        patches = []  # each [1,patch,Dz]
        context = []  # decode context for streaming (prompt tail patches)
        if prompt_feat is not None:
            ctx = min(self.stream_prefix - 1, prompt_feat.shape[0])
            context = [prompt_feat[-ctx + i][None] for i in range(ctx)] if ctx else []
        chunks = []
        pos = T
        steps = 0
        one = np.ones((1, 1), dtype=np.float32)
        for i in range(limit):
            noise = rng.standard_normal((1, self.feat_dim, self.patch), dtype=np.float32)
            pred = self.cfm.run(None, {
                "dit_hidden": dit_hidden.astype(self.cfm_dtype),
                "cond": cond.astype(self.cfm_dtype),
                "noise": noise.astype(self.cfm_dtype),
                "cfg_value": np.array([self.cfg_value], dtype=self.cfm_dtype),
            })[0].astype(np.float32)  # [1,Dz,patch]
            patch_tp = pred.transpose(0, 2, 1)  # [1,patch,Dz]
            patches.append(patch_tp)
            cond = pred
            steps = i + 1

            if streaming:
                window = (context + patches)[-self.stream_prefix:]
                lat = np.concatenate(window, axis=1)  # [1,t,patch,Dz] pieces -> [1,t*patch? no]
                lat = lat.reshape(1, -1, self.feat_dim).transpose(0, 2, 1)  # [1,Dz,t*patch]
                wav = self.vae_dec.run(None, {"latents": lat.astype(np.float32)})[0]
                chunks.append(wav[0, 0, -self.patch_len:].copy())

            stop = int(np.argmax(stop_logits[0]))
            if i > self.min_len and stop == 1:
                break

            curr = self._feat_embed(patch_tp[:, None])  # [1,1,patch,Dz] -> [1,1,H]
            dit_hidden, stop_logits, past = self._lm(curr, curr, one, pos, past)
            pos += 1

        # final decode: with a prompt, torch prepends the (stream_prefix-1) prompt context
        # patches and drops their samples after decoding (continuity at the boundary).
        full = context + patches
        lat_all = np.concatenate(full, axis=1).reshape(1, -1, self.feat_dim).transpose(0, 2, 1)
        wav = self.vae_dec.run(None, {"latents": lat_all.astype(np.float32)})[0][0, 0]
        if context:
            wav = wav[self.patch_len * len(context):]
        gen_lat = np.concatenate(patches, axis=1).reshape(1, -1, self.feat_dim).transpose(0, 2, 1)
        out = dict(wav=wav, steps=steps)
        if streaming:
            out["chunks"] = chunks
        if collect_latents:
            out["latents"] = gen_lat[0]
        return out
