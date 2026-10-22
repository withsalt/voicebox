# Copyright (c) 2026
# Shared utilities for the VoxCPM-0.5B ONNX export + parity pipeline.
#
# Centralises:
#   * path resolution (model dir / source dir / output dirs)
#   * model loading (VoxCPMModel.from_local, fp32 master copy)
#   * dtype post-processing (fp16 via onnxconverter_common,
#     int8 weight-only block-wise via ORT MatMulNBits accuracy_level=4)
#   * numeric comparison helpers (SNR / cosine) used by every parity script
#
# Nothing here mutates the vendored VoxCPM/ source tree.
import os
import sys
import json

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----------------------------------------------------------------------------
# Paths (defaults; repointed via configure() / CLI --model-dir/--out-dir)
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))  # scripts/voxcpm-tts-onnx-export
SRC = os.path.join(HERE, "VoxCPM", "src")  # vendored model source (package root)
DEFAULT_MODEL = os.path.join(HERE, "VoxCPM-0.5B")

MODEL = DEFAULT_MODEL
OUT = os.path.join(HERE, "out")
PKG_DIR = None  # final C# package dir (set by configure, depends on dtype)
GOLD_DIR = os.path.join(OUT, "golden")
DTYPE = "fp32"
OPSET = 20
DEVICE = "cpu"
# Block size for weight-only quantization (MatMulNBits int8). Must divide every quantized
# K dim (lm K dims are 1024/4096/128-multiples -> all divisible by 32).
QUANT_BLOCK_SIZE = 32

# 生成默认值的单一事实来源: export_cfm 按 inference_timesteps 烘焙 Euler 步数,
# export_embeddings 把整块写进 manifest 的 generation (供 C# 读)。
# 其余值抄自上游 core.py::_generate 的默认参数。
GENERATION_DEFAULTS = {
    "inference_timesteps": 10,
    "cfg_value": 2.0,
    "min_len": 2,
    "max_len_text_ratio": 6.0,
    "max_len_text_bias": 10,
    "streaming_prefix_patches": 3,
}

if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)

MODEL_REQUIRED = [
    "config.json",
    "pytorch_model.bin",
    "audiovae.pth",
    "tokenizer.json",
]


def validate_model_dir(path: str) -> str:
    if not path:
        raise ValueError("模型目录不能为空。")
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise NotADirectoryError(f"模型目录不存在: {path}")
    missing = [f for f in MODEL_REQUIRED if not os.path.isfile(os.path.join(path, f))]
    if missing:
        raise FileNotFoundError(f"不像 VoxCPM 模型目录 (缺少: {', '.join(missing)}): {path}")
    return path


def add_io_args(ap):
    ap.add_argument("--model-dir", default=None, help="原始模型目录; 默认 ./VoxCPM-0.5B。")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="导出基目录; 默认 ./out。部署包 -> <out-dir>/voxcpm-0.5b-onnx-<dtype>, golden -> <out-dir>/golden。",
    )
    ap.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="lm/cfm/feat_encoder 精度; 默认 fp32 (数值对拍基准)。"
        "fp16 = 权重+计算半精 (体积减半)。"
        "int8 = lm/cfm/feat_encoder 权重-only int8 (MatMulNBits accuracy_level=4);"
        "体积/带宽 ~1/4, CPU RTF ~0.48; 曾评估的 int4 lm 判停头受损 (stop SNR 36dB,"
        "长度跑飞) 已弃用。audio_vae 始终 fp32。",
    )
    ap.add_argument("--opset", type=int, default=20, help="ONNX opset; 默认 20。")
    ap.add_argument("--device", default="cpu", help="导出/trace 设备; 默认 cpu (fp32 trace)。")
    return ap


def configure(model_dir=None, out_dir=None, dtype=None, opset=None, device=None):
    """Resolve global paths/options. Returns (MODEL, PKG_DIR)."""
    global MODEL, OUT, PKG_DIR, GOLD_DIR, DTYPE, OPSET, DEVICE
    if model_dir:
        MODEL = os.path.abspath(model_dir)
    MODEL = validate_model_dir(MODEL)
    if out_dir:
        OUT = os.path.abspath(out_dir)
    if dtype:
        DTYPE = dtype
    if opset:
        OPSET = opset
    if device:
        DEVICE = device
    PKG_DIR = os.path.join(OUT, f"voxcpm-0.5b-onnx-{DTYPE}")
    GOLD_DIR = os.path.join(OUT, "golden")
    os.makedirs(PKG_DIR, exist_ok=True)
    os.makedirs(os.path.join(PKG_DIR, "embeddings"), exist_ok=True)
    os.makedirs(os.path.join(PKG_DIR, "tokenizer"), exist_ok=True)
    os.makedirs(GOLD_DIR, exist_ok=True)
    return MODEL, PKG_DIR


def apply_io_args(args):
    configure(
        getattr(args, "model_dir", None),
        getattr(args, "out_dir", None),
        getattr(args, "dtype", None),
        getattr(args, "opset", None),
        getattr(args, "device", None),
    )


# ----------------------------------------------------------------------------
# Model / config loading
# ----------------------------------------------------------------------------
_VOX = None


def load_config(model_dir=None) -> dict:
    with open(os.path.join(model_dir or MODEL, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def load_vox(force_reload=False):
    """Load VoxCPMModel as an fp32 CPU master copy (cached).

    fp32 on CPU: the checkpoint is bf16; every graph is exported at fp32 and
    reduced afterwards (fp16/int8 post-processing), matching the qwen3-tts flow.
    """
    global _VOX
    if _VOX is not None and not force_reload:
        return _VOX
    import torch

    torch.backends.cudnn.enabled = False  # this box: cuDNN sublibrary mismatch on conv1d
    from voxcpm.model.voxcpm import VoxCPMModel

    model = VoxCPMModel.from_local(MODEL, optimize=False, device="cpu")
    model = model.float().eval()
    model.config.dtype = "float32"
    # StaticKVCache buffers were allocated bf16 at construction -> rebuild fp32
    model.base_lm.setup_cache(1, model.config.max_length, "cpu", torch.float32)
    model.residual_lm.setup_cache(1, model.config.max_length, "cpu", torch.float32)
    _VOX = model
    return _VOX


def seed_all(seed: int = 0):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# ONNX export + dtype post-processing
# ----------------------------------------------------------------------------
def export_onnx(module, args, path, input_names, output_names, dynamic_axes, opset=None):
    """torch.onnx.export wrapper (legacy static tracing, external data when large)."""
    import torch

    opset = opset or OPSET
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            module,
            args,
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    import onnx

    # torch auto-externalizes >2GB models to MANY per-tensor files; consolidate them into a
    # single <name>.data so the package stays clean. No-op for <2GB single-file models.
    d = os.path.dirname(path) or "."
    meta = onnx.load(path, load_external_data=False)
    locs = {e.value for t in meta.graph.initializer for e in t.external_data if e.key == "location"}
    if locs:
        model = onnx.load(path)  # pulls the loose external files into memory
        data_file = os.path.basename(path) + ".data"
        os.remove(path)
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file,
            size_threshold=1024,
            convert_attribute=False,
        )
        for loc in locs:
            fp = os.path.join(d, loc)
            if os.path.isfile(fp):
                os.remove(fp)
    onnx.checker.check_model(path)
    return path


def _fold_identity_weights(model):
    """Rewire consumers that read a weight through Identity(initializer) to read the
    initializer directly, then drop those Identity nodes (pure pass-through -> exact).

    The unrolled cfm graph shares each estimator weight across 9 unrolled steps via
    Identity passthroughs; the lm graph is a single forward (no-op there). Folding makes
    every step point at one initializer so MatMulNBits quantizes each weight exactly once."""
    g = model.graph
    init_names = {i.name for i in g.initializer}
    id_in = {n.output[0]: n.input[0] for n in g.node if n.op_type == "Identity"}

    def root(name):
        seen = set()
        while name in id_in and name not in seen:
            seen.add(name)
            name = id_in[name]
        return name

    weight_ident = {o for o in id_in if root(o) in init_names}
    if not weight_ident:
        return 0
    for n in g.node:
        for j, inp in enumerate(n.input):
            if inp in weight_ident:
                n.input[j] = root(inp)
    dead = [n for n in g.node if n.op_type == "Identity" and n.output[0] in weight_ident]
    for n in dead:
        g.node.remove(n)
    return len(dead)


def postprocess_dtype(path, dtype, keep_io_types=False):
    """Convert a freshly-exported fp32 ONNX in-place to fp16/int8. fp32 is a no-op."""
    if dtype == "fp32":
        return path
    if dtype == "fp16":
        import onnx
        from onnxconverter_common import float16

        model = onnx.load(path)
        model16 = float16.convert_float_to_float16(
            model, keep_io_types=keep_io_types, disable_shape_infer=True
        )
        data_old = path + ".data"
        onnx.save(model16, path, save_as_external_data=False)
        if os.path.isfile(data_old):
            os.remove(data_old)
        return path
    if dtype == "int8":
        # weight-only block-wise int8 -> com.microsoft.MatMulNBits (CPU/CUDA/DirectML)。
        # RTN (data-free). Only initializer-B MatMuls are touched; activation x activation
        # MatMuls (attention q.k / w.v) and Conv (vae) are left untouched.
        #
        # 必须 accuracy_level=4 (激活 int8 计算): MLAS 只为该组合提供快 kernel,
        # 默认 (fp32 反量化) 慢 ~10x (实测 cfm 37→376ms/Run; acc4 为 17ms)。
        # 代价是激活量化误差 (cfm 10 步复合 ~26dB, fe ~33dB, lm dit ~31dB) —— 均显著
        # 好于曾评估的 int4 (lm dit 18.7dB / stop 36dB, 判停不稳导致长度跑飞, 已弃用)。
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

        model = onnx.load(path)  # pulls any external .data into memory
        folded = _fold_identity_weights(model)
        if folded:
            print(f"   {dtype} prep: folded {folded} identity-shared weights")
        quant = MatMulNBitsQuantizer(model, bits=8, block_size=QUANT_BLOCK_SIZE,
                                     is_symmetric=False, accuracy_level=4)
        quant.process()
        g = quant.model.model.graph
        seen, keep = set(), []
        for it in g.initializer:
            if it.name in seen:
                continue
            seen.add(it.name)
            keep.append(it)
        if len(keep) != len(g.initializer):
            del g.initializer[:]
            g.initializer.extend(keep)
        data_old = path + ".data"  # orphaned external data from the fp32 export
        tmp = path + ".quant.tmp"
        onnx.save_model(quant.model.model, tmp, save_as_external_data=False)
        os.replace(tmp, path)
        if os.path.isfile(data_old):
            os.remove(data_old)
        return path
    raise ValueError(f"unknown dtype {dtype}")


def dedup_initializers(base_path, variant_path):
    """variant 图中与 base 图逐字节相同的 initializer 改写为共享 base 的外部数据。

    用途: lm_gqa.onnx 与 lm.onnx 权重完全相同 (仅注意力子图结构不同), 直接双份会让
    包体积翻倍。做法: ① base 若还是单文件 protobuf, 先转成 <base>.onnx + <base>.onnx.data
    (ORT 加载透明); ② variant 中逐字节相同的张量改为指向同一 data 文件的 external ref,
    其余 (如 GQA 的 rope cache) 保持内联。

    匹配按 (dtype, dims, SHA256) 内容寻址 —— 常量折叠后的权重名是 onnx::MatMul_
    流水号, 两张图编号不同, 按名字对不上。"""
    import hashlib

    import onnx
    from onnx import TensorProto

    d = os.path.dirname(base_path) or "."
    data_file = os.path.basename(base_path) + ".data"

    meta = onnx.load(base_path, load_external_data=False)
    if not any(t.data_location == TensorProto.EXTERNAL for t in meta.graph.initializer):
        m = onnx.load(base_path)
        onnx.save_model(m, base_path, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=data_file, size_threshold=1024, convert_attribute=False)
        meta = onnx.load(base_path, load_external_data=False)

    table = {}
    with open(os.path.join(d, data_file), "rb") as fdata:
        for t in meta.graph.initializer:
            if t.data_location != TensorProto.EXTERNAL:
                continue
            kv = {e.key: e.value for e in t.external_data}
            if kv.get("location") != data_file:
                continue
            off, ln = int(kv.get("offset", "0")), int(kv["length"])
            fdata.seek(off)
            digest = hashlib.sha256(fdata.read(ln)).hexdigest()
            table.setdefault((t.data_type, tuple(t.dims), ln, digest), (off, ln))

    var = onnx.load(variant_path)  # 松散外部数据一并载入内存
    hits, saved = 0, 0
    for t in var.graph.initializer:
        raw = onnx.numpy_helper.to_array(t).tobytes()
        info = table.get((t.data_type, tuple(t.dims), len(raw), hashlib.sha256(raw).hexdigest()))
        if info is None:
            continue
        off, ln = info
        t.ClearField("raw_data")
        t.ClearField("float_data")
        t.ClearField("int32_data")
        t.ClearField("int64_data")
        del t.external_data[:]
        for k, v in (("location", data_file), ("offset", str(off)), ("length", str(ln))):
            e = t.external_data.add()
            e.key, e.value = k, v
        t.data_location = TensorProto.EXTERNAL
        hits += 1
        saved += ln

    inline_left = sum(len(t.raw_data) for t in var.graph.initializer
                      if t.data_location != TensorProto.EXTERNAL)
    if inline_left > 1_500_000_000:
        raise RuntimeError(f"dedup 后 variant 仍有 {inline_left / 1e6:.0f}MB 内联权重, 拒绝写出巨型文件。")
    loose = variant_path + ".data"   # variant 自己导出时可能带的外部数据文件
    onnx.save(var, variant_path)
    if os.path.isfile(loose):
        os.remove(loose)
    print(f"   dedup: {hits} tensors ({saved / 1e6:.0f} MB) -> 共享 {data_file}; "
          f"variant 内联剩余 {inline_left / 1e6:.1f} MB")
    return hits


# ----------------------------------------------------------------------------
# Raw .bin writers (little-endian) for the C# side
# ----------------------------------------------------------------------------
def save_bin(arr, path, dtype=np.float32):
    a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    a.tofile(path)
    return list(a.shape)


def to_numpy(t):
    import torch

    if isinstance(t, torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()
    return np.asarray(t)


# ----------------------------------------------------------------------------
# Comparison helpers (parity gate: SNR>=40dB, cosine>=0.999)
# ----------------------------------------------------------------------------
_RESULTS = []


def compare(name, a, b, atol_warn=1e-3, min_snr=None, min_cos=None):
    """min_snr/min_cos: per-result gate overrides (e2e wav is gated looser than
    per-graph checks; see gate_and_exit)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    len_mismatch = a.size != b.size
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    diff = np.abs(a - b)
    sig = np.sum(a**2)
    err = np.sum((a - b) ** 2)
    snr = 10 * np.log10(sig / err) if err > 0 else float("inf")
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    out = dict(
        name=name,
        n=n,
        max_abs=float(diff.max() if n else 0.0),
        snr_db=float(snr),
        cos=cos,
        len_mismatch=len_mismatch,
        min_snr=min_snr,
        min_cos=min_cos,
    )
    flag = "ok " if out["max_abs"] <= atol_warn else "CHK"
    extra = "  LEN-MISMATCH!" if len_mismatch else ""
    print(
        f"[{flag}] {name:40} n={n:<9} max_abs={out['max_abs']:.3e} "
        f"snr={out['snr_db']:.1f}dB cos={out['cos']:.6f}{extra}"
    )
    _RESULTS.append(out)
    return out


def gate_and_exit(min_snr=40.0, min_cos=0.999):
    failed = [
        r
        for r in _RESULTS
        if r["len_mismatch"]
        or (r["snr_db"] < (r.get("min_snr") or min_snr))
        or (r["cos"] < (r.get("min_cos") or min_cos))
    ]
    if failed:
        print(
            f"\nPARITY GATE FAILED ({len(failed)} 项): "
            + ", ".join(
                f"{r['name']}(snr={r['snr_db']:.1f},cos={r['cos']:.5f}"
                f"{',LEN' if r['len_mismatch'] else ''})"
                for r in failed
            )
        )
        sys.exit(1)
    print(f"\nPARITY GATE PASSED ({len(_RESULTS)} 项, min_snr={min_snr}, min_cos={min_cos}).")


def ort_session(path, providers=None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    return ort.InferenceSession(path, sess_options=so, providers=providers or ["CPUExecutionProvider"])
