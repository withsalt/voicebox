# -*- coding: utf-8 -*-
"""Export CAM++ (speech_campplus_sv_zh-cn_16k-common) to a dynamic-length ONNX model.

部署契约 (C# 直接可用):
  inputs : feats     [batch, time, 80]   Kaldi fbank 80 维、逐段减均值 (CMN) 后的特征
  outputs: embedding [batch, 192]        说话人 embedding (未归一化, 打分用余弦相似度)

导出坑 (本脚本的存在理由):
  CAMLayer.seg_pooling 的 avg_pool1d(ceil_mode=True) 导出成 ONNX AveragePool 后,
  最后一个不完整分段窗口的除数语义与 PyTorch 不一致; 且 FCM 头部把时间轴减半,
  导致只有内部长度恰为 seg_len(100) 整数倍的输入才精确 (其余长度 cosine 跌到 0.4~0.98)。
  解决: 导出前把 seg_pooling 换成 cumsum 分段均值 + index_select 广播 —— 数学等价
  (本脚本会先在 eager 模式下证明等价), 且所有 shape 计算都是动态算子。

用法:
  python 01_export_onnx.py --model-dir <原始模型目录> --out-dir <onnx 部署目录>
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "speech_campplus_sv_zh-cn_16k-common")
DEFAULT_OUT_DIR = os.path.normpath(os.path.join(
    SCRIPT_DIR, "..", "..", "src", "VoiceBox.Speaker.CampPlus", "runtimes", "models", "speaker", "campplus"))

OPSET = 17
SEG_LEN = 100          # CAMLayer.seg_pooling 的分段长度 (模型结构常量)
TRACE_FRAMES = 298     # trace 用长度, 与部署窗口无关 (动态轴)
# 对拍长度集: 覆盖 <1 段 / 整数段 / 部分段 / 内部(时间减半后)整百与非整百
PARITY_FRAMES = [50, 100, 148, 150, 198, 200, 298, 350, 500, 977, 1000]


def parse_args():
    p = argparse.ArgumentParser(description="Export CAM++ SV model to dynamic-length ONNX")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                   help="原始 modelscope 模型目录 (含 campplus_cn_common.bin / configuration.json)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="ONNX 部署目录 (C# 推理直接使用: campplus.onnx + config.ini)")
    return p.parse_args()


def read_model_config(model_dir):
    import json
    with open(os.path.join(model_dir, "configuration.json"), encoding="utf-8") as f:
        conf = json.load(f)
    mc = conf["model"]["model_config"]
    return {
        "sample_rate": int(mc["sample_rate"]),
        "fbank_dim": int(mc["fbank_dim"]),
        "emb_size": int(mc["emb_size"]),
        "threshold": float(conf["model"].get("yesOrno_thr", 0.31)),
        "model_file": conf["model"].get("pretrained_model", "campplus_cn_common.bin"),
    }


def build_model(model_dir, cfg):
    from modelscope.models.audio.sv.DTDNN import CAMPPlus
    m = CAMPPlus(feat_dim=cfg["fbank_dim"], embedding_size=cfg["emb_size"])
    sd = torch.load(os.path.join(model_dir, cfg["model_file"]), map_location="cpu")
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def _seg_pooling_cumsum(x, seg_len):
    """avg_pool1d(k=s=seg_len, ceil_mode=True) + 广播回原长, 的动态 shape 安全等价实现。

    分段均值用 cumsum 前缀和差分求 (末段除数 = 实际元素数, 与 PyTorch 边界裁剪语义一致),
    广播用 index_select(arange//seg_len)。全部算子在 trace 下保持 shape 动态。
    """
    t = x.shape[-1]
    cs = torch.cumsum(x, dim=-1)
    cs = torch.cat([torch.zeros_like(cs[..., :1]), cs], dim=-1)  # cs[i] = 前 i 个之和
    starts = torch.arange(0, t, seg_len, device=x.device)
    ends = torch.clamp(starts + seg_len, max=t)
    seg = (torch.index_select(cs, -1, ends) - torch.index_select(cs, -1, starts)) \
        / (ends - starts).to(x.dtype)
    idx = torch.arange(t, device=x.device) // seg_len
    return torch.index_select(seg, -1, idx)


def patch_seg_pooling(model):
    """把 model 内所有 CAMLayer 实例的 seg_pooling 换成 cumsum 版 (实例级绑定)。

    只 patch 传入的实例、不改类, 因此"原始模型"参考天然不可能被污染,
    等价性证明不依赖任何语句顺序。
    """
    import types

    from modelscope.models.audio.sv.DTDNN_layers import CAMLayer

    def seg_pooling(self, x, seg_len=SEG_LEN, stype="avg"):
        assert stype == "avg", f"unexpected seg_pooling stype: {stype}"
        return _seg_pooling_cumsum(x, seg_len)

    patched = 0
    for m in model.modules():
        if isinstance(m, CAMLayer):
            m.seg_pooling = types.MethodType(seg_pooling, m)
            patched += 1
    assert patched > 0, "no CAMLayer found to patch"
    return patched


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def write_config_ini(path, cfg):
    lines = [
        "[model]",
        "model_type=campplus",
        "model_file=campplus.onnx",
        f"sample_rate={cfg['sample_rate']}",
        f"feature_dim={cfg['fbank_dim']}",
        f"embedding_size={cfg['emb_size']}",
        f"threshold={cfg['threshold']}",
        f"opset={OPSET}",
        "",
        "[fbank]",
        "# torchaudio.compliance.kaldi.fbank 默认参数 (modelscope SV pipeline 同款)",
        "frame_length_ms=25",
        "frame_shift_ms=10",
        "num_mel_bins=80",
        "window_type=povey",
        "dither=0",
        "preemphasis=0.97",
        "remove_dc_offset=1",
        "snip_edges=1",
        "low_freq=20",
        "high_freq=0",
        "waveform_scale=32768",
        "# 推理前逐段减均值 (CMN); 该步同时消除输入增益差异",
        "apply_cmn=1",
        "",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    cfg = read_model_config(args.model_dir)
    print(f"model config: {cfg}")

    # 1) 原始模型 (未打补丁) 的参考输出
    torch.manual_seed(20260712)
    original = build_model(args.model_dir, cfg)
    inputs = {t: torch.randn(1, t, cfg["fbank_dim"]) for t in PARITY_FRAMES}
    with torch.no_grad():
        refs = {t: original(x).numpy() for t, x in inputs.items()}

    # 2) 实例级替换 seg_pooling, eager 等价性证明 (original 未被改动, 参考始终干净)
    patched = build_model(args.model_dir, cfg)
    n_patched = patch_seg_pooling(patched)
    max_patch_diff = 0.0
    with torch.no_grad():
        for t, x in inputs.items():
            d = float(np.abs(patched(x).numpy() - refs[t]).max())
            max_patch_diff = max(max_patch_diff, d)
    print(f"seg_pooling cumsum patch vs original (eager, {n_patched} CAM layers, "
          f"{len(inputs)} lengths): max diff {max_patch_diff:.3e}")
    assert max_patch_diff < 1e-4, "cumsum seg_pooling is NOT equivalent to the original"

    # 3) 导出 (TorchScript trace 路径; dynamo 导出器对本模型仍有 AveragePool 语义问题)
    os.makedirs(args.out_dir, exist_ok=True)
    onnx_path = os.path.join(args.out_dir, "campplus.onnx")
    torch.onnx.export(
        patched,
        torch.randn(1, TRACE_FRAMES, cfg["fbank_dim"]),
        onnx_path,
        input_names=["feats"],
        output_names=["embedding"],
        dynamic_axes={"feats": {0: "batch", 1: "time"}, "embedding": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):  # 保障单文件部署契约
        import onnx
        m = onnx.load(onnx_path)
        os.remove(data_file)
        onnx.save_model(m, onnx_path, save_as_external_data=False)
    print(f"exported: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB, single file)")

    # 4) onnxruntime 多长度对拍 (vs 原始 eager 模型)
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst_diff, worst_cos = 0.0, 1.0
    for t, x in inputs.items():
        out = sess.run(None, {"feats": x.numpy()})[0]
        d = float(np.abs(out - refs[t]).max())
        c = cosine(out, refs[t])
        worst_diff = max(worst_diff, d)
        worst_cos = min(worst_cos, c)
        print(f"  T={t}: max_diff={d:.3e} cos={c:.8f}")
    print(f"onnxruntime vs original torch: worst max_diff={worst_diff:.3e} worst cos={worst_cos:.8f}")
    assert worst_diff < 1e-3 and worst_cos > 0.99999, "ONNX parity check failed"

    # 5) 批维 sanity (batch=2 与逐条一致)
    xb = torch.cat([inputs[198], inputs[198].flip(1)], dim=0)
    out_b = sess.run(None, {"feats": xb.numpy()})[0]
    out_1 = sess.run(None, {"feats": xb[:1].numpy()})[0]
    out_2 = sess.run(None, {"feats": xb[1:].numpy()})[0]
    bd = float(np.abs(out_b - np.concatenate([out_1, out_2])).max())
    print(f"batch=2 vs per-item: max_diff={bd:.3e}")
    assert bd < 1e-4

    write_config_ini(os.path.join(args.out_dir, "config.ini"), cfg)
    print(f"config.ini written; deployment dir ready: {args.out_dir}")


if __name__ == "__main__":
    main()
