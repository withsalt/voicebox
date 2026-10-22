# -*- coding: utf-8 -*-
"""Export xiaoyun FSMN-KWS to a streaming ONNX model with explicit per-layer caches.

Streaming contract (exact steady-state equivalence with full-utterance forward):
  inputs : feats   [1, T, 400]   (fbank80 -> LFR(5,3) -> CMVN, T >= 1)
           cache0..cache3 [1, 128, 11, 1]  (zeros at stream reset)
  outputs: probs   [1, T, 2599]  (softmax posteriors)
           new_cache0..new_cache3
Each FSMN layer keeps (lorder-1) + rorder = 9 + 2 = 11 projected frames.
Output frame j corresponds to input frame j - 8 (4 layers x rorder 2 lookahead).
The first ~35 output frames after reset are warm-up and should be ignored.
"""
import json
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model", "speech_charctc_kws_phone-xiaoyun")
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "onnx")

LORDER, RORDER = 10, 2
CACHE_LEN = (LORDER - 1) + RORDER  # 11
PROJ_DIM = 128
NUM_LAYERS = 4

# 部署唤醒词（官方预训练目标词）。char 全集建模，可改任意中文词，但实际唤醒率取决于与训练数据匹配度。
WAKE_WORD = "小云小云"
# 部署 keywords.json 的命中阈值（命中分 = 关键词各 token 峰值后验的几何平均, scale (0,1]）。
# 官方预训练模型无训练期 DET 选阈过程，故取与 C# 侧 FsmnKwsOptions.Threshold 默认值一致的 0.5
# （亦贴近 kws-train 自训模型导出的 threshold1≈0.534）；如需调阈直接改此值。
# 注意：模型自带 keywords.json 的 threshold1=0.069 是闭源 kwsbp 引擎的分数标度，与本 ONNX 解码器不通用。
DEFAULT_THRESHOLD1 = 0.5
# ONNX opset（torch.onnx.export 与 meta 共用，避免两处漂移）。
OPSET = 19


class StreamingFsmnKws(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.in_linear1 = encoder.in_linear1
        self.in_linear2 = encoder.in_linear2
        self.blocks = nn.ModuleList(list(encoder.fsmn))
        self.out_linear1 = encoder.out_linear1
        self.out_linear2 = encoder.out_linear2

    @staticmethod
    def _block_forward(block, x, cache):
        p = block.linear(x)                       # [1, T, 128]
        b = p.transpose(1, 2).unsqueeze(-1)       # [1, 128, T, 1]
        buf = torch.cat([cache, b], dim=2)        # [1, 128, T+11, 1]
        new_cache = buf[:, :, -CACHE_LEN:, :]
        center = buf[:, :, LORDER - 1:-RORDER, :]                       # T frames
        y_left = block.fsmn_block.conv_left(buf[:, :, :-RORDER, :])     # T frames
        y_right = block.fsmn_block.conv_right(buf[:, :, LORDER:, :])    # T frames
        out = center + y_left + y_right
        out = out.squeeze(-1).transpose(1, 2)     # [1, T, 128]
        out = F.relu(block.affine(out))           # [1, T, 250]
        return out, new_cache

    def forward(self, feats, cache0, cache1, cache2, cache3):
        x = F.relu(self.in_linear2(self.in_linear1(feats)))
        caches = [cache0, cache1, cache2, cache3]
        new_caches = []
        for block, cache in zip(self.blocks, caches):
            x, nc = self._block_forward(block, x, cache)
            new_caches.append(nc)
        logits = self.out_linear2(self.out_linear1(x))
        probs = torch.softmax(logits, dim=-1)
        return (probs, *new_caches)


def load_encoder():
    from funasr import AutoModel
    am = AutoModel(model=MODEL_DIR, keywords=WAKE_WORD, device="cpu", disable_update=True)
    encoder = am.model.encoder
    encoder.eval()
    return encoder


def parse_cmvn(path):
    from funasr.frontends.wav_frontend import load_cmvn
    cmvn = load_cmvn(path)  # [2, 400]: row0 add_shift (neg mean), row1 rescale (inv std)
    return cmvn[0].tolist(), cmvn[1].tolist()


def _read_lexicon(path):
    """读 funasr seg_dict（每行 `word\\ttok1 tok2 ...`）为 {word: [tokens]}。"""
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) >= 2:
                d[parts[0]] = parts[1].split()
    return d


def keyword_token_ids(wake, tokens_path, lexicon_path=None):
    """唤醒词 → token id 序列（id = tokens.txt 的 0 基行号）。

    逐字查 tokens；缺字时用 lexicon 展开成子 token 再查；仍缺记入 OOV。
    与 kws-train 导出 (src/kws_export/exporter.py::keyword_token_ids) 同逻辑。返回 (ids, oov)。
    """
    with open(tokens_path, encoding="utf-8") as f:
        tokens = [line.rstrip("\n") for line in f]
    tok2id = {t: i for i, t in enumerate(tokens)}
    lex = _read_lexicon(lexicon_path) if lexicon_path and os.path.exists(lexicon_path) else {}
    ids, oov = [], []
    for ch in wake:
        if ch in tok2id:
            ids.append(tok2id[ch])
        elif ch in lex and all(t in tok2id for t in lex[ch]):
            ids.extend(tok2id[t] for t in lex[ch])
        else:
            oov.append(ch)
    return ids, oov


def source_checkpoint():
    """官方 init_param 路径（相对 model 目录），从 configuration.json 读，随再微调自动跟随。"""
    with open(os.path.join(MODEL_DIR, "configuration.json"), encoding="utf-8") as f:
        conf = json.load(f)
    return conf.get("file_path_metas", {}).get("init_param", "")


def write_keywords_json(out_dir, wake=WAKE_WORD, threshold1=DEFAULT_THRESHOLD1):
    """写部署用 keywords.json（字段对齐 kws-train 导出 output/<词>/onnx/keywords.json）。

    自训模型在训练第七步由 DET 曲线自动选阈并回写 threshold1；官方预训练模型无此过程，
    故用 DEFAULT_THRESHOLD1 兜底（见其注释）。其余字段沿用基线 kwsbp_resource 模板常量。
    """
    entry = {
        "name": wake,
        "type": "wakeup",
        "activation": True,
        "is_main": True,
        "lm_boost": 0.0,
        "am_boost": 0.0,
        "threshold1": round(float(threshold1), 6),
        "threshold2": -1.0,
        "subseg_threshold": -0.6,
        "high_threshold": 90.0,
        "min_dur": 0.25,
        "max_dur": 2.5,
        "cc_name": "commoncc",
    }
    path = os.path.join(out_dir, "keywords.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"word_list": [entry]}, f, ensure_ascii=False, indent=4)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    encoder = load_encoder()
    model = StreamingFsmnKws(encoder).eval()

    # sanity: streaming forward == funasr full forward (steady state)
    torch.manual_seed(0)
    T = 50
    feats = torch.randn(1, T, 400)
    with torch.no_grad():
        full = encoder(feats)  # use_softmax=False -> logits
        full_probs = torch.softmax(full, dim=-1)
        caches = [torch.zeros(1, PROJ_DIM, CACHE_LEN, 1) for _ in range(NUM_LAYERS)]
        outs = []
        for t in range(T):  # frame-by-frame, worst case
            res = model(feats[:, t:t + 1, :], *caches)
            outs.append(res[0])
            caches = list(res[1:])
        stream_probs = torch.cat(outs, dim=1)
    delay = NUM_LAYERS * RORDER  # 8
    warmup = (NUM_LAYERS - 1) * (LORDER - 1) + delay  # 35
    diff = (stream_probs[0, warmup:T, :] - full_probs[0, warmup - delay:T - delay, :]).abs().max()
    print(f"steady-state max diff (torch stream vs full): {diff:.3e}")
    assert diff < 1e-5, "streaming wrapper does not match full forward"

    onnx_path = os.path.join(OUT_DIR, "kws_streaming.onnx")
    cache_inputs = [f"cache{i}" for i in range(NUM_LAYERS)]
    cache_outputs = [f"new_cache{i}" for i in range(NUM_LAYERS)]
    torch.onnx.export(
        model,
        (feats[:, :3, :], *[torch.zeros(1, PROJ_DIM, CACHE_LEN, 1) for _ in range(NUM_LAYERS)]),
        onnx_path,
        input_names=["feats", *cache_inputs],
        output_names=["probs", *cache_outputs],
        dynamic_axes={"feats": {1: "T"}, "probs": {1: "T"}},
        opset_version=OPSET,
    )

    # 新版 torch (>=2.9 dynamo 导出) 会把权重放进 .onnx.data 外部文件;
    # 模型仅 ~3MB, 合并为单文件便于部署 (与 C#/README 的单文件契约一致)。
    import onnx
    m = onnx.load(onnx_path)  # 自动加载外部数据
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    onnx.save_model(m, onnx_path, save_as_external_data=False)
    print(f"exported: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB, single file)")

    # onnxruntime check vs torch
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    caches = [np.zeros((1, PROJ_DIM, CACHE_LEN, 1), dtype=np.float32) for _ in range(NUM_LAYERS)]
    outs = []
    feats_np = feats.numpy()
    for s in range(0, T, 7):  # uneven chunks
        chunk = feats_np[:, s:s + 7, :]
        res = sess.run(None, {"feats": chunk, **{f"cache{i}": caches[i] for i in range(NUM_LAYERS)}})
        outs.append(res[0])
        caches = res[1:]
    ort_probs = np.concatenate(outs, axis=1)
    diff = abs(ort_probs - stream_probs.numpy()).max()
    print(f"onnxruntime vs torch max diff: {diff:.3e}")
    assert diff < 1e-4

    # resources for C#
    shutil.copy(os.path.join(MODEL_DIR, "funasr", "tokens_2599.txt"), os.path.join(OUT_DIR, "tokens.txt"))
    shutil.copy(os.path.join(MODEL_DIR, "funasr", "lexicon.txt"), os.path.join(OUT_DIR, "lexicon.txt"))
    shutil.copy(os.path.join(MODEL_DIR, "funasr", "am.mvn.dim80_l2r2"), os.path.join(OUT_DIR, "am.mvn"))
    add_shift, rescale = parse_cmvn(os.path.join(MODEL_DIR, "funasr", "am.mvn.dim80_l2r2"))

    # 唤醒词 token ids（字段对齐 kws-train 导出 build_meta：keyword / token ids / OOV）。
    kw_ids, kw_oov = keyword_token_ids(
        WAKE_WORD, os.path.join(OUT_DIR, "tokens.txt"), os.path.join(OUT_DIR, "lexicon.txt"))
    if kw_oov:
        print(f"⚠ 唤醒词含 OOV 字（不在 token 表）：{kw_oov}；这些字无法被该模型唤醒。")
    meta = {
        "keyword": WAKE_WORD,
        "keyword_token_ids": kw_ids,
        "keyword_oov_chars": kw_oov,
        "sample_rate": 16000,
        "fbank": {"num_mel_bins": 80, "frame_length_ms": 25, "frame_shift_ms": 10,
                  "window": "hamming", "dither": 0.0, "snip_edges": True,
                  "waveform_scale": 32768.0},
        "lfr": {"m": 5, "n": 3},
        "cmvn": {"add_shift": add_shift, "rescale": rescale},
        "model": {"input_dim": 400, "output_dim": 2599, "num_layers": NUM_LAYERS,
                  "proj_dim": PROJ_DIM, "left_order": LORDER, "right_order": RORDER,
                  "cache_len": CACHE_LEN,
                  "output_delay_frames": delay, "warmup_frames": warmup,
                  "blank_id": 0},
        "source_checkpoint": source_checkpoint(),
        "checkpoint_kind": "pretrained",
        "opset": OPSET,
    }
    with open(os.path.join(OUT_DIR, "kws_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    kw_path = write_keywords_json(OUT_DIR)
    print(f"keywords.json: {kw_path} (keyword={WAKE_WORD}, threshold1={DEFAULT_THRESHOLD1})")
    print(f"resources written to {OUT_DIR}")


if __name__ == "__main__":
    main()
