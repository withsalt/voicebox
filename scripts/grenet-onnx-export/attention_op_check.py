"""Check standard ONNX Attention against its explicit FP32 decomposition."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def main():
    batch, sequence, heads, width = 1, 5, 4, 24
    hidden = heads * width
    shape = helper.make_tensor_value_info("shape", TensorProto.INT64, [4])
    shape_t = helper.make_tensor("shape", TensorProto.INT64, [4], [batch, sequence, heads, width])
    output_shape_t = helper.make_tensor("output_shape", TensorProto.INT64, [3], [batch, sequence, hidden])
    nodes = [
        helper.make_node("Reshape", ["Q", "shape"], ["q_3d"]),
        helper.make_node("Transpose", ["q_3d"], ["q"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["K", "shape"], ["k_3d"]),
        helper.make_node("Transpose", ["k_3d"], ["kt"], perm=[0, 2, 3, 1]),
        helper.make_node("Reshape", ["V", "shape"], ["v_3d"]),
        helper.make_node("Transpose", ["v_3d"], ["v"], perm=[0, 2, 1, 3]),
        helper.make_node("MatMul", ["q", "kt"], ["scores"]),
        helper.make_node("Softmax", ["scores"], ["weights"], axis=-1),
        helper.make_node("MatMul", ["weights", "v"], ["attended"]),
        helper.make_node("Transpose", ["attended"], ["attended_3d"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["attended_3d", "output_shape"], ["reference"]),
        helper.make_node("Attention", ["Q", "K", "V"], ["candidate"], q_num_heads=heads, kv_num_heads=heads, scale=1.0),
    ]
    graph = helper.make_graph(
        nodes,
        "attention-check",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [batch, sequence, hidden]) for name in ("Q", "K", "V")],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [batch, sequence, hidden]) for name in ("reference", "candidate")],
        [shape_t, output_shape_t],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)], producer_name="voicebox-attention-check")
    onnx.checker.check_model(model)
    path = Path(__file__).with_name("attention-check.onnx")
    onnx.save(model, path)
    generator = np.random.default_rng(42)
    q, k, v = [generator.standard_normal((batch, sequence, hidden), dtype=np.float32) for _ in range(3)]
    q *= width**-0.5
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    reference, candidate = session.run(None, {"Q": q, "K": k, "V": v})
    print("max_abs_delta", float(np.abs(reference - candidate).max()))
    np.testing.assert_allclose(candidate, reference, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    main()
