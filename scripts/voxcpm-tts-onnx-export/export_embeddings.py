# Copyright (c) 2026
# Export host-side lookup tables + the C# manifest:
#
#   embeddings/text_embedding.bin  [V,H] fp32   base_lm token embedding (host lookup;
#                                               keeps the lm graph int4-friendly)
#   embeddings/multichar_zh.json                vocab tokens of >=2 pure-Chinese chars
#                                               (id + per-char split ids) for the C#
#                                               mask_multichar_chinese_tokens port
#   config.json                                 model dims + generation defaults + files map
import os
import json
import argparse

import numpy as np

import _common as C


def _multichar_zh_table(vox):
    """token -> single-char ids table replicating model/utils.mask_multichar_chinese_tokens.

    C# tokenizes with tokenizer.json, then for each produced id present in this table
    substitutes the per-char id list. unk (id 0) chars are kept like the python wrapper
    (convert_tokens_to_ids -> unk)."""
    tok = vox.text_tokenizer.tokenizer  # unwrap CharTokenizerWrapper
    vocab = tok.get_vocab()
    # step 1: the wrapper's precomputed set = vocab keys that are pure CJK, len>=2
    multichar = {t for t in vocab if len(t) >= 2 and all("一" <= c <= "鿿" for c in t)}
    # step 2: the wrapper strips "▁" from each PRODUCED token before the set lookup, so
    # "▁你好" splits when "你好" is in the set -> key the table by the produced token's id.
    table = {}
    for token, tid in vocab.items():
        clean = token.replace("▁", "")
        if clean in multichar:
            table[str(tid)] = tok.convert_tokens_to_ids(list(clean))
    return table


def run(vox=None):
    if vox is None:
        print("== loading model (fp32) for embeddings ...")
        vox = C.load_vox()
    cfg = vox.config

    emb = C.to_numpy(vox.base_lm.embed_tokens.weight)  # [V,H] (use_mup=False -> no scale_emb)
    p = os.path.join(C.PKG_DIR, "embeddings", "text_embedding.bin")
    shape = C.save_bin(emb, p)
    print(f"   wrote {p} shape={shape}")

    zh = _multichar_zh_table(vox)
    zp = os.path.join(C.PKG_DIR, "embeddings", "multichar_zh.json")
    with open(zp, "w", encoding="utf-8") as f:
        json.dump(zh, f, ensure_ascii=False)
    print(f"   wrote {zp} ({len(zh)} multi-char zh tokens)")

    manifest = {
        "model_type": "voxcpm",
        "version": "0.5b",
        "sample_rate": vox.sample_rate,
        "hidden_size": cfg.lm_config.hidden_size,
        "feat_dim": cfg.feat_dim,
        "patch_size": cfg.patch_size,
        "vae_chunk_size": vox.chunk_size,
        "samples_per_step": cfg.patch_size * vox.chunk_size,
        "vocab_size": cfg.lm_config.vocab_size,
        "base_lm_layers": cfg.lm_config.num_hidden_layers,
        "residual_lm_layers": cfg.residual_lm_num_layers,
        "num_attention_heads": cfg.lm_config.num_attention_heads,
        "num_key_value_heads": cfg.lm_config.num_key_value_heads,
        "head_dim": cfg.lm_config.hidden_size // cfg.lm_config.num_attention_heads,
        "max_length": cfg.max_length,
        "audio_start_token": vox.audio_start_token,
        "generation": dict(C.GENERATION_DEFAULTS),
        "files": {
            "lm": "lm.onnx",
            "lm_gqa": "lm_gqa.onnx",   # CPU 加速变体 (GQA contrib op, 可选; C# 按存在性启用)
            "feat_encoder": "feat_encoder.onnx",
            "cfm_decode": "cfm_decode.onnx",
            "audio_vae_decoder": "audio_vae_decoder.onnx",
            "audio_vae_encoder": "audio_vae_encoder.onnx",
            "text_embedding": "embeddings/text_embedding.bin",
            "multichar_zh": "embeddings/multichar_zh.json",
            "tokenizer": "tokenizer/tokenizer.json",
        },
        "text_embedding_shape": shape,
    }
    mp = os.path.join(C.PKG_DIR, "config.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"   wrote {mp}")


def main():
    ap = argparse.ArgumentParser(description="Export VoxCPM embeddings + C# manifest.")
    C.add_io_args(ap)
    args = ap.parse_args()
    C.apply_io_args(args)
    run()


if __name__ == "__main__":
    main()
