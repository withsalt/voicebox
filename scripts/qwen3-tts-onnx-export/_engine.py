# Copyright (c) 2026
# Pure-numpy + ONNXRuntime reference engine for the exported Qwen3-TTS package.
# This mirrors EXACTLY what the C# VoiceBox.TTS.Qwen3 pipeline does, so it doubles as:
#   * the e2e parity reference (vs the original PyTorch model), and
#   * the executable spec for the C# port.
#
# Embeddings + prompt construction + sampling live here (numpy); the talker / code_predictor
# / codec_decoder transformers live in ONNX.
import os
import json
import numpy as np
import onnxruntime as ort


def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


class EmbeddingStore:
    """Loads the raw embedding bins and reproduces the talker input-embedding arithmetic."""

    def __init__(self, pkg_dir):
        with open(os.path.join(pkg_dir, "embeddings", "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        ed = os.path.join(pkg_dir, "embeddings")
        m = self.config["embeddings"]

        def load(entry):
            arr = np.fromfile(os.path.join(ed, entry["file"]), dtype=np.float32)
            return arr.reshape(entry["shape"])

        self.text_projected = load(m["textProjectedEmbedding"]) if "textProjectedEmbedding" in m else None
        if self.text_projected is None:
            self.text_emb = load(m["textEmbedding"])        # [Vtxt, 2048]
            self.fc1_w = load(m["textProjFc1Weight"])       # [2048, 2048]
            self.fc1_b = load(m["textProjFc1Bias"])         # [2048]
            self.fc2_w = load(m["textProjFc2Weight"])       # [H, 2048]
            self.fc2_b = load(m["textProjFc2Bias"])         # [H]
        self.codec_emb = load(m["codecEmbedding"])          # [Vc, H]
        self.cp_emb = load(m["cpCodecEmbedding"])           # [G-1, Vcp, H]
        self.H = self.codec_emb.shape[1]

    def text_proj(self, ids):
        """text token ids -> talker-hidden embeddings via text_embedding + ResizeMLP(SiLU)."""
        if self.text_projected is not None:
            return self.text_projected[np.asarray(ids, dtype=np.int64)]   # [N, H]
        x = self.text_emb[np.asarray(ids, dtype=np.int64)]   # [N, 2048]
        x = _silu(x @ self.fc1_w.T + self.fc1_b)
        return x @ self.fc2_w.T + self.fc2_b                 # [N, H]

    def codec(self, ids):
        return self.codec_emb[np.asarray(ids, dtype=np.int64)]   # [N, H]

    def cp_group(self, group_idx, ids):
        return self.cp_emb[group_idx][np.asarray(ids, dtype=np.int64)]   # [N, H]


def build_prompt(store, text_ids, speaker, language, instruct_ids=None):
    """Reproduce Qwen3TTSForConditionalGeneration.generate() prompt construction for
    CustomVoice (non_streaming_mode=True). Returns (inputs_embeds[1,Tp,H], tts_pad_embed[1,1,H]).
    This is the executable spec for the C# prompt builder.

      text_ids:     full tokenized assistant prompt ids
                    [im_start, assistant, \n, <text...>, im_end, \n, im_start, assistant, \n]
      speaker:      speaker name (lower-cased key into speakerId), or None
      language:     language name (or "auto"); dialect speakers override it
      instruct_ids: optional 1.7B instruct prompt ids, else None
    """
    cfg = store.config
    sp = cfg["specialTokens"]
    H = store.H
    text_ids = np.asarray(text_ids, dtype=np.int64)

    # tts_bos / tts_eos / tts_pad (text-projected)
    bos, eos, pad = store.text_proj([sp["ttsBos"], sp["ttsEos"], sp["ttsPad"]])  # [3,H]
    bos, eos, pad = bos[None], eos[None], pad[None]                              # each [1,H]

    # language id (+ dialect override)
    lang = (language or "auto").lower()
    lang_id = None if lang == "auto" else cfg["codecLanguageId"][lang]
    spk_key = speaker.lower() if speaker else None
    if spk_key and lang in ("chinese", "auto"):
        dial = cfg["speakerIsDialect"].get(spk_key, False)
        if dial:
            lang_id = cfg["codecLanguageId"][dial]

    # codec control tokens + speaker + (pad, bos)
    if lang_id is None:
        ctrl = [sp["codecNothink"], sp["codecThinkBos"], sp["codecThinkEos"]]
    else:
        ctrl = [sp["codecThink"], sp["codecThinkBos"], lang_id, sp["codecThinkEos"]]
    emb0 = store.codec(ctrl)                                  # [k,H]
    emb1 = store.codec([sp["codecPad"], sp["codecBos"]])     # [2,H]
    if spk_key:
        spk = store.codec([cfg["speakerId"][spk_key]])       # [1,H]
        codec_in = np.concatenate([emb0, spk, emb1], 0)
    else:
        codec_in = np.concatenate([emb0, emb1], 0)           # [codec_len,H]
    codec_len = codec_in.shape[0]

    role = store.text_proj(text_ids[:3])                     # [3,H]
    A = np.concatenate([np.repeat(pad, codec_len - 2, 0), bos], 0) + codec_in[:-1]  # [codec_len-1,H]

    text_body = text_ids[3:-5]                               # actual text tokens
    tlen = text_body.shape[0]
    C_text = np.concatenate([store.text_proj(text_body), eos], 0)                   # [tlen+1,H]
    C = C_text + store.codec([sp["codecPad"]] * (tlen + 1))                         # [tlen+1,H]
    D = pad + store.codec([sp["codecBos"]])                                          # [1,H]

    parts = []
    if instruct_ids is not None and len(instruct_ids) > 0:
        parts.append(store.text_proj(np.asarray(instruct_ids, dtype=np.int64)))
    parts += [role, A, C, D]
    prompt = np.concatenate(parts, 0)[None].astype(np.float32)   # [1,Tp,H]
    return prompt, pad[None].astype(np.float32)                  # trailing = tts_pad


class Qwen3OnnxEngine:
    def __init__(self, pkg_dir, providers=None):
        self.store = EmbeddingStore(pkg_dir)
        self.cfg = self.store.config
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = providers or ["CPUExecutionProvider"]
        self.talker = ort.InferenceSession(os.path.join(pkg_dir, "talker.onnx"), so, providers=prov)
        self.cp = ort.InferenceSession(os.path.join(pkg_dir, "code_predictor.onnx"), so, providers=prov)
        self.codec = ort.InferenceSession(os.path.join(pkg_dir, "codec_decoder.onnx"), so, providers=prov)
        self._io_fp16 = self.talker.get_inputs()[0].type == "tensor(float16)"
        t = self.cfg["talker"]
        self.L, self.kv, self.D, self.H = t["numLayers"], t["numKvHeads"], t["headDim"], t["hiddenSize"]
        cp = self.cfg["codePredictor"]
        self.cpL, self.cpkv, self.cpD = cp["numLayers"], cp["numKvHeads"], cp["headDim"]
        self.G = self.cfg["numCodeGroups"]
        self.eos = self.cfg["specialTokens"]["codecEos"]
        sup = self.cfg["suppressTokens"]
        self.sup_lo, self.sup_hi, self.sup_keep = sup["rangeStart"], sup["rangeEnd"], set(sup["keep"])
        gen = self.cfg.get("generation", {})
        self.rep_penalty = float(gen.get("repetitionPenalty", 1.05))
        self.min_new_tokens = int(gen.get("minNewTokens", 2))
        self.sub_temperature = float(gen.get("subtalkerTemperature", 0.9))

    def _ge(self, arr):
        return arr.astype(np.float16) if self._io_fp16 else arr.astype(np.float32)

    def _empty_kv(self, L, kv, D):
        z = np.zeros((L, 1, kv, 0, D), dtype=np.float16 if self._io_fp16 else np.float32)
        return z, z

    def _process_logits(self, logits, history, n_emitted):
        """Greedy logits processing matching HF generate: repetition penalty over the
        generated group0 history, min_new_tokens eos-suppression, then suppress_tokens."""
        logits = logits.astype(np.float64).copy()
        if self.rep_penalty != 1.0 and history:
            for tid in set(history):
                s = logits[tid]
                logits[tid] = s * self.rep_penalty if s < 0 else s / self.rep_penalty
        if n_emitted < self.min_new_tokens:
            logits[self.eos] = -np.inf
        for t in range(self.sup_lo, self.sup_hi):
            if t not in self.sup_keep:
                logits[t] = -np.inf
        return logits

    def _talker(self, embeds, attn_len, pos_start, past_k, past_v):
        S = embeds.shape[1]
        am = np.ones((1, attn_len), dtype=np.int64)
        pos = np.broadcast_to(np.arange(pos_start, pos_start + S, dtype=np.int64), (3, 1, S)).copy()
        out = self.talker.run(None, {"inputs_embeds": self._ge(embeds), "attention_mask": am,
                                     "position_ids": pos, "past_k": past_k, "past_v": past_v})
        return out  # logits, hidden, present_k, present_v

    def _cp_frame(self, past_hidden, group0):
        """Fused sub-talker: ONE Run -> (fine_codes[15], next_sum[1,1,H]). Greedy (do_sample=0)
        for parity. next_sum = codec(group0) + Σ cp_codec_embedding[gs][code_gs] (talker next
        input embedding sum; caller adds tts_pad)."""
        store = self.store
        emb0 = store.codec([group0]).reshape(1, 1, self.H)
        n_steps = self.G - 1
        out = self.cp.run(["fine_codes", "next_sum"], {
            "past_hidden": self._ge(past_hidden),
            "emb0": self._ge(emb0),
            "do_sample": np.array(0, dtype=np.int64),
            "temperature": np.array(self.sub_temperature, dtype=np.float32),
            "randoms": np.zeros(n_steps, dtype=np.float32),
        })
        fine_codes = [int(x) for x in np.asarray(out[0]).reshape(-1).tolist()]
        next_sum = np.asarray(out[1]).astype(np.float32)
        return fine_codes, next_sum

    def generate(self, prompt_embeds, attention_len, trailing_pad_embed, max_frames=2048):
        """prompt_embeds [1,Tp,H] -> codes [T,16] (greedy)."""
        store = self.store
        logits, hidden, pk, pv = self._talker(prompt_embeds, attention_len, 0,
                                               *self._empty_kv(self.L, self.kv, self.D))
        history = []
        group0 = int(np.argmax(self._process_logits(logits[0, -1], history, 0)))
        history.append(group0)
        past_hidden = hidden[:, -1:].astype(np.float32)
        total = attention_len
        codes = []
        for _ in range(max_frames):
            if group0 == self.eos:
                break
            g, next_sum = self._cp_frame(past_hidden, group0)
            frame = [group0] + g
            codes.append(frame)
            # next talker input = sum(16 code embeds) + tts_pad; the fused cp returns the code sum.
            nxt = next_sum + trailing_pad_embed.reshape(1, 1, self.H)
            total += 1
            logits, hidden, pk, pv = self._talker(nxt, total, total - 1, pk, pv)
            group0 = int(np.argmax(self._process_logits(logits[0, -1], history, len(codes))))
            history.append(group0)
            past_hidden = hidden[:, -1:].astype(np.float32)
        return np.asarray(codes, dtype=np.int64)            # [T, 16]

    def decode(self, codes, chunk=300, left=25):
        """codes [T,16] -> wav (float32), replicating chunked_decode(chunk_size, left_context)."""
        up = self.cfg["samplesPerFrame"]
        c = np.clip(codes, 0, None).T[None]                 # [1,16,T]
        T = c.shape[2]
        out = []
        start = 0
        while start < T:
            end = min(start + chunk, T)
            ctx = left if start - left > 0 else start
            chunk_codes = c[:, :, start - ctx:end]
            wav = self.codec.run(["wav"], {"codes": chunk_codes.astype(np.int64)})[0]
            out.append(wav[..., ctx * up:])
            start = end
        return np.concatenate(out, axis=-1).reshape(-1)
