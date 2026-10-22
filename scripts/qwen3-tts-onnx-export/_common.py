# Copyright (c) 2026
# Shared utilities for the Qwen3-TTS (12Hz CustomVoice) ONNX export + parity pipeline.
#
# Centralises:
#   * path resolution (model dir / source dir / output dirs)
#   * model loading (Qwen3TTSModel.from_pretrained, eager attn, fp32 master copy)
#   * dtype post-processing (fp16 via onnxconverter_common, int8 via ORT dynamic-quant,
#     int4 weight-only block-wise via ORT MatMulNBits)
#   * numeric comparison helpers (SNR / cosine) used by every parity script
#   * little-endian raw .bin writers for the C# EmbeddingStore
#
# Nothing here mutates the vendored Qwen3-TTS/ source tree.
import os
import sys
import json
import shutil

import numpy as np

# ---- isolate from a broken user-site as early as possible is not feasible (numpy/onnx
#      live there); instead we just make sure HF runs fully offline against local dirs. ----
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----------------------------------------------------------------------------
# Paths (defaults; repointed via configure() / CLI --model-dir/--out-dir)
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))  # scripts/qwen3-tts-onnx-export
SRC = os.path.join(HERE, "Qwen3-TTS")  # vendored model source
DEFAULT_MODEL = os.path.join(HERE, "Qwen3-TTS-12Hz-0.6B-CustomVoice")

MODEL = DEFAULT_MODEL
OUT = os.path.join(HERE, "out")
PKG_DIR = None  # final C# package dir (set by configure, depends on size+dtype)
GOLD_DIR = os.path.join(OUT, "golden")
DTYPE = "fp16"
OPSET = 20
DEVICE = "cuda:0"
# Block size for int4 weight-only quantization (MatMulNBits). Must divide every quantized K dim
# (talker/cp K are 1024/2048/3072 -> all divisible by 32). 32 keeps per-block scales fine-grained
# (better accuracy than 128) at a small size cost; the model is tiny so the extra scales are cheap.
INT4_BLOCK_SIZE = 32
# Fixed KV length for the static-KV talker decode graph (talker_static.onnx, CUDA-graph path).
# Generation stops at codec_eos well before this; 12.5Hz so 2048 frames ≈ 164s of audio.
STATIC_MAX_LEN = 2048

if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)

MODEL_REQUIRED = [
    "config.json",
    "model.safetensors",
    "vocab.json",
    "merges.txt",
    os.path.join("speech_tokenizer", "config.json"),
]


def validate_model_dir(path: str) -> str:
    if not path:
        raise ValueError("模型目录不能为空。")
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise NotADirectoryError(f"模型目录不存在: {path}")
    missing = [f for f in MODEL_REQUIRED if not os.path.isfile(os.path.join(path, f))]
    if missing:
        raise FileNotFoundError(
            f"不像 Qwen3-TTS 模型目录 (缺少: {', '.join(missing)}): {path}"
        )
    return path


def add_io_args(ap):
    ap.add_argument(
        "--model-dir",
        default=None,
        help="原始模型目录; 默认 ./Qwen3-TTS-12Hz-0.6B-CustomVoice。",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="导出基目录; 默认 ./out。部署包 -> <out-dir>/qwen3-tts-<size>-onnx-<dtype>, golden -> <out-dir>/golden。",
    )
    ap.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16", "int8", "int4"],
        default="fp32",
        help="talker/code_predictor 精度; 默认 bf16 (该模型 bf16 训练, CUDA 上数值稳定)。"
        "int4 = 权重-only block-wise (MatMulNBits), 体积最小、CPU/DirectML 提速、保真度下降。"
        "codec_decoder 始终 fp32。",
    )
    ap.add_argument("--opset", type=int, default=20, help="ONNX opset; 默认 20。")
    ap.add_argument("--device", default="cuda:0", help="导出/trace 设备; 默认 cuda:0。")
    return ap


def configure(
    model_dir=None, out_dir=None, dtype=None, opset=None, device=None, size_tag=None
):
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
    tag = size_tag or _infer_size_tag(MODEL)
    PKG_DIR = os.path.join(OUT, f"qwen3-tts-{tag}-onnx-{DTYPE}")
    GOLD_DIR = os.path.join(OUT, "golden", f"{tag}")
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


def _infer_size_tag(model_dir: str) -> str:
    cfg = load_config(model_dir)
    size = str(cfg.get("tts_model_size", "")).lower()
    return {"0b6": "0.6b", "1b7": "1.7b"}.get(size, size or "model")


# ----------------------------------------------------------------------------
# Model / config loading
# ----------------------------------------------------------------------------
def load_config(model_dir=None) -> dict:
    with open(os.path.join(model_dir or MODEL, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def load_tts(device=None, dtype="float32", attn_implementation="eager"):
    """Load the full Qwen3-TTS model (master copy is fp32 on the given device)."""
    import torch
    from qwen_tts import Qwen3TTSModel

    dt = {"float32": torch.float32, "float16": torch.float16}[dtype]
    dev = device or DEVICE
    tts = Qwen3TTSModel.from_pretrained(
        MODEL, device_map=dev, dtype=dt, attn_implementation=attn_implementation
    )
    tts.model.eval()
    return tts


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
def export_onnx(
    module, args, path, input_names, output_names, dynamic_axes, opset=None
):
    """torch.onnx.export wrapper (static tracing, external data when large)."""
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

    # torch auto-externalizes >2GB models to MANY per-tensor files; consolidate them into a single
    # <name>.data so the package stays clean. size_threshold keeps scalars inline (avoids onnx 1.21's
    # scalar-merge bug). No-op for <2GB single-file models (e.g. 0.6B), so those stay byte-identical.
    d = os.path.dirname(path) or "."
    meta = onnx.load(path, load_external_data=False)
    locs = {
        e.value
        for t in meta.graph.initializer
        for e in t.external_data
        if e.key == "location"
    }
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
    """Rewire consumers that read a weight through Identity(initializer) to read the initializer
    directly, then drop those Identity nodes (Identity is a pure pass-through -> exact).

    The fused code_predictor shares each of its 5 transformer layers' weights across the 15
    unrolled steps via Identity passthroughs. Left in place, MatMulNBits can only quantize the one
    DIRECT reference per weight and must keep the fp32 original alive for the Identity-fed copies
    -> the graph ends up BIGGER (fp32 weights + int4 duplicates). Folding makes every step point at
    one initializer, so each weight is quantized once and the fp32 copy is freed. No-op for the
    talker (single forward, no Identity-fed weights). Returns the number of Identity nodes removed."""
    g = model.graph
    init_names = {i.name for i in g.initializer}
    id_in = {n.output[0]: n.input[0] for n in g.node if n.op_type == "Identity"}

    def root(name):  # collapse Identity chains down to the ultimate source
        seen = set()
        while name in id_in and name not in seen:
            seen.add(name)
            name = id_in[name]
        return name

    weight_ident = {o for o in id_in if root(o) in init_names}  # Identity outputs resolving to a weight
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


def _hoist_const_matmul_weights(model):
    """Move any MatMul B (input[1]) that is produced by a Constant node into a graph initializer.

    MatMulNBitsQuantizer only quantizes MatMuls whose B is an *initializer*. The fused
    code_predictor folds its head_w[gs] / cp_codec_embedding[gs] lookup tables into Constant
    NODES (do_constant_folding collapses the per-step Slice+Transpose), so without this pass
    those big tables (15x2048x1024 each) would be skipped. The talker has no such case -> no-op.
    Renaming a Constant's output to an initializer of the same name keeps every consumer valid.
    Returns the number of tables hoisted."""
    import onnx

    g = model.graph
    init_names = {i.name for i in g.initializer}
    const_by_out = {
        n.output[0]: n
        for n in g.node
        if n.op_type == "Constant" and len(n.output) == 1
    }
    wanted = {
        n.input[1]
        for n in g.node
        if n.op_type == "MatMul"
        and len(n.input) >= 2
        and n.input[1] not in init_names
        and n.input[1] in const_by_out
    }
    moved = []
    for out_name in wanted:
        cnode = const_by_out[out_name]
        val = next((a for a in cnode.attribute if a.name == "value"
                    and a.type == onnx.AttributeProto.TENSOR), None)
        if val is None:
            continue  # value_float / sparse_value etc. -> not one of our weight tables
        t = onnx.TensorProto()
        t.CopyFrom(val.t)
        t.name = out_name
        g.initializer.append(t)
        moved.append(cnode)
    for cnode in moved:
        g.node.remove(cnode)
    return len(moved)


def postprocess_dtype(path, dtype):
    """Convert a freshly-exported fp32 ONNX in-place to fp16/int8/int4. fp32 is a no-op.
    Returns the final path (same as input)."""
    if dtype == "fp32":
        return path
    if dtype == "fp16":
        import onnx
        from onnxconverter_common import float16

        model = onnx.load(path)
        model16 = float16.convert_float_to_float16(
            model, keep_io_types=False, disable_shape_infer=True
        )
        onnx.save(model16, path, save_as_external_data=False)
        return path
    if dtype == "int8":
        from onnxruntime.quantization import quantize_dynamic, QuantType

        tmp = path + ".fp32.onnx"
        os.replace(path, tmp)
        quantize_dynamic(
            model_input=tmp,
            model_output=path,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        os.remove(tmp)
        return path
    if dtype == "int4":
        # weight-only block-wise int4 -> com.microsoft.MatMulNBits (CPU/CUDA/DirectML).
        # RTN (data-free, default algo). Only initializer-B MatMuls are touched: all talker
        # weights + cp transformer weights, plus the cp lookup tables once hoisted. Activation
        # x activation MatMuls (attention q.kᵀ / w.v) and Conv (codec) are left untouched.
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

        model = onnx.load(path)  # pulls any external .data into memory
        folded = _fold_identity_weights(model)       # cp: un-share the 15-step unrolled weights
        hoisted = _hoist_const_matmul_weights(model)  # cp: hoist head/codec lookup tables
        if folded or hoisted:
            print(f"   int4 prep: folded {folded} identity-shared weights, "
                  f"hoisted {hoisted} constant lookup tables")
        quant = MatMulNBitsQuantizer(
            model, bits=4, block_size=INT4_BLOCK_SIZE, is_symmetric=False
        )
        quant.process()
        # safety net: MatMulNBits already dedups quantized weights by name (shared cp weights ->
        # one <name>_Q4); drop any exact-name duplicate initializer just in case a future ORT doesn't.
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
        data_old = path + ".data"  # orphaned external data from the fp32 export (>2GB models)
        tmp = path + ".int4.tmp"
        onnx.save_model(quant.model.model, tmp, save_as_external_data=False)
        os.replace(tmp, path)
        if os.path.isfile(data_old):
            os.remove(data_old)
        return path
    raise ValueError(f"unknown dtype {dtype}")


# ----------------------------------------------------------------------------
# Raw .bin writers (little-endian) for the C# EmbeddingStore
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


def compare(name, a, b, atol_warn=1e-3):
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
    )
    flag = "ok " if out["max_abs"] <= atol_warn else "CHK"
    extra = "  LEN-MISMATCH!" if len_mismatch else ""
    print(
        f"[{flag}] {name:34} n={n:<9} max_abs={out['max_abs']:.3e} "
        f"snr={out['snr_db']:.1f}dB cos={out['cos']:.6f}{extra}"
    )
    _RESULTS.append(out)
    return out


def gate_and_exit(min_snr=40.0, min_cos=0.999):
    failed = [
        r
        for r in _RESULTS
        if r["len_mismatch"] or (r["snr_db"] < min_snr) or (r["cos"] < min_cos)
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
    print(
        f"\nPARITY GATE PASSED ({len(_RESULTS)} 项, min_snr={min_snr}, min_cos={min_cos})."
    )


def ort_session(path, providers=None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    return ort.InferenceSession(
        path, sess_options=so, providers=providers or ["CPUExecutionProvider"]
    )
