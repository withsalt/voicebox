# Copyright (c) 2026
# Dump the talker/code-predictor embedding + text-projection weights as raw little-endian
# .bin files, plus a comprehensive config.json the C# EmbeddingStore/ModelConfig consume.
#
# The C# side builds the talker/cp `inputs_embeds` (the intricate prompt arithmetic in
# Qwen3TTSForConditionalGeneration.generate) from these tensors; only the transformers
# (+ codec_head / cp lm_heads) live in ONNX.
import os
import json
import argparse
import numpy as np

import _common as C


def _emb_entry(arr, fname, out_dir, dtype=np.float32):
    shape = C.save_bin(arr, os.path.join(out_dir, fname), dtype=dtype)
    return {"file": fname, "dtype": "float32", "shape": shape}


def _projected_text_emb_entry(talker, out_dir, chunk_size=1024):
    """Precompute text_embedding -> text_projection in fp32.

    The C# runtime only ever consumes the projected [text_vocab, hidden] table. Exporting it
    removes the per-request CPU MLP while preserving the exact PyTorch fp32 projection math.
    """
    import torch

    path = os.path.join(out_dir, "text_projected_embedding.bin")
    text_weight = talker.model.text_embedding.weight.detach()
    vocab = int(text_weight.shape[0])
    hidden = int(talker.config.hidden_size)
    print(f"   precomputing text projection: [{vocab},{text_weight.shape[1]}] -> [{vocab},{hidden}] ...")
    with open(path, "wb") as f, torch.inference_mode():
        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            y = talker.text_projection(text_weight[start:end].float()).detach().cpu().numpy().astype(np.float32, copy=False)
            y.tofile(f)
            if start == 0 or end == vocab or (end % (chunk_size * 32) == 0):
                print(f"     projected {end}/{vocab}")
    return {"file": os.path.basename(path), "dtype": "float32", "shape": [vocab, hidden]}


def run(tts=None):
    emb_dir = os.path.join(C.PKG_DIR, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    cfg = C.load_config()
    tcfg = cfg["talker_config"]
    cpcfg = tcfg["code_predictor_config"]
    gen_path = os.path.join(C.MODEL, "generation_config.json")
    gen = {}
    if os.path.isfile(gen_path):
        with open(gen_path, encoding="utf-8") as f:
            gen = json.load(f)

    if tts is None:
        print("== loading model (fp32) for embeddings ...")
        tts = C.load_tts(device="cpu", dtype="float32")
    talker = tts.model.talker
    cp = talker.code_predictor
    H = int(tcfg["hidden_size"])

    print("== dumping embedding bins ...")
    n = lambda t: C.to_numpy(t)
    manifest = {
        "textProjectedEmbedding": _projected_text_emb_entry(talker, emb_dir),
        "codecEmbedding": _emb_entry(n(talker.model.codec_embedding.weight), "codec_embedding.bin", emb_dir),
    }
    # code-predictor: 15 input embeddings stacked -> [num_code_groups-1, cp_vocab, H]
    cp_emb = np.stack([n(e.weight) for e in cp.model.codec_embedding], axis=0)
    manifest["cpCodecEmbedding"] = _emb_entry(cp_emb, "cp_codec_embedding.bin", emb_dir)

    # suppress: range(vocab-1024, vocab) except codec_eos (see generate() suppress_tokens)
    talker_vocab = int(tcfg["vocab_size"])
    config = {
        "ttsModelSize": cfg.get("tts_model_size"),
        "ttsModelType": cfg.get("tts_model_type"),
        "tokenizerType": cfg.get("tokenizer_type"),
        "sampleRate": 24000,
        "frameRate": 12.5,
        "samplesPerFrame": 1920,
        "numCodeGroups": int(tcfg["num_code_groups"]),
        "staticCacheLen": C.STATIC_MAX_LEN,
        "specialTokens": {
            "imStart": cfg["im_start_token_id"], "imEnd": cfg["im_end_token_id"],
            "assistant": cfg["assistant_token_id"],
            "ttsBos": cfg["tts_bos_token_id"], "ttsEos": cfg["tts_eos_token_id"],
            "ttsPad": cfg["tts_pad_token_id"],
            "codecBos": tcfg["codec_bos_id"], "codecEos": tcfg["codec_eos_token_id"],
            "codecPad": tcfg["codec_pad_id"], "codecThink": tcfg["codec_think_id"],
            "codecNothink": tcfg["codec_nothink_id"],
            "codecThinkBos": tcfg["codec_think_bos_id"], "codecThinkEos": tcfg["codec_think_eos_id"],
        },
        "codecLanguageId": tcfg["codec_language_id"],
        "speakerId": tcfg["spk_id"],
        "speakerIsDialect": tcfg["spk_is_dialect"],
        "suppressTokens": {"rangeStart": talker_vocab - 1024, "rangeEnd": talker_vocab,
                           "keep": [tcfg["codec_eos_token_id"]]},
        "talker": {
            "hiddenSize": H, "numLayers": int(tcfg["num_hidden_layers"]),
            "numHeads": int(tcfg["num_attention_heads"]), "numKvHeads": int(tcfg["num_key_value_heads"]),
            "headDim": int(tcfg["head_dim"]), "textHiddenSize": int(tcfg["text_hidden_size"]),
            "textVocabSize": int(tcfg["text_vocab_size"]), "codecVocabSize": talker_vocab,
            "ropeTheta": float(tcfg["rope_theta"]), "rmsNormEps": float(tcfg["rms_norm_eps"]),
            "ropeInterleaved": bool(tcfg["rope_scaling"].get("interleaved", False)),
            "mropeSection": tcfg["rope_scaling"]["mrope_section"],
            "positionIdPerSeconds": int(tcfg.get("position_id_per_seconds", 13)),
            "intermediateSize": int(tcfg["intermediate_size"]),
        },
        "codePredictor": {
            "hiddenSize": int(cpcfg["hidden_size"]), "numLayers": int(cpcfg["num_hidden_layers"]),
            "numHeads": int(cpcfg["num_attention_heads"]), "numKvHeads": int(cpcfg["num_key_value_heads"]),
            "headDim": int(cpcfg["head_dim"]), "vocabSize": int(cpcfg["vocab_size"]),
            "ropeTheta": float(cpcfg["rope_theta"]), "rmsNormEps": float(cpcfg["rms_norm_eps"]),
            "intermediateSize": int(cpcfg["intermediate_size"]),
            "inputDim": H,   # cp inputs_embeds are at talker hidden; graph projects internally
        },
        "generation": {
            "doSample": bool(gen.get("do_sample", True)),
            "temperature": float(gen.get("temperature", 0.9)),
            "topK": int(gen.get("top_k", 50)),
            "topP": float(gen.get("top_p", 1.0)),
            "repetitionPenalty": float(gen.get("repetition_penalty", 1.05)),
            "subtalkerDoSample": bool(gen.get("subtalker_dosample", True)),
            "subtalkerTemperature": float(gen.get("subtalker_temperature", 0.9)),
            "subtalkerTopK": int(gen.get("subtalker_top_k", 50)),
            "subtalkerTopP": float(gen.get("subtalker_top_p", 1.0)),
            "maxNewTokens": int(gen.get("max_new_tokens", 8192)),
            "minNewTokens": 2,
        },
        "embeddings": manifest,
    }
    with open(os.path.join(emb_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    total_mb = sum(os.path.getsize(os.path.join(emb_dir, m["file"])) for m in manifest.values()) / 1e6
    print(f"   wrote {len(manifest)} bins ({total_mb:.0f} MB) + config.json -> {emb_dir}")
    print(f"   talker H={H} layers={config['talker']['numLayers']}  cp H={config['codePredictor']['hiddenSize']} "
          f"layers={config['codePredictor']['numLayers']}  cp_emb={cp_emb.shape}")
    return config


def main():
    ap = argparse.ArgumentParser(description="Export Qwen3-TTS embeddings + config for C#.")
    C.add_io_args(ap)
    args = ap.parse_args()
    C.apply_io_args(args)
    run()


if __name__ == "__main__":
    main()
