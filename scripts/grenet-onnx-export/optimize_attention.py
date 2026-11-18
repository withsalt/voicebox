"""Fuse GRE-Net frequency attention into com.microsoft.MultiHeadAttention.

Q/K/V stay in their original three-dimensional projected layout and the fused
node's [1, 101, 96] output feeds the existing output projection, removing six
input-layout nodes, the merge transpose and the MatMul/Softmax/MatMul group per
transformer. The exporter folds head_dim**-0.5 into the packed query weights, so
the fused node must use scale=1.0; the default scale would double-apply it.

The graph stays at opset 17 and only adds the com.microsoft domain. Time
attention, caches and every non-frequency node are untouched.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import NodeProto, helper, shape_inference


@dataclass
class Fusion:
    qk: NodeProto
    attention: NodeProto
    removed: list[NodeProto]
    tail: NodeProto | None


def static_shape(value):
    tensor = value.type.tensor_type
    if not tensor.shape.dim:
        return None
    dimensions = []
    for dim in tensor.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        dimensions.append(dim.dim_value)
    return dimensions


def permutation(node):
    return next((list(attribute.ints) for attribute in node.attribute if attribute.name == "perm"), None)


def fuse_frequency_attention(model):
    if not any(opset.domain == "com.microsoft" for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    shapes = {value.name: static_shape(value) for value in (*graph.input, *graph.output, *graph.value_info)}
    consumers = Counter(input_name for node in graph.node for input_name in node.input)
    children = defaultdict(list)
    by_output = {}
    for node in graph.node:
        for output in node.output:
            by_output[output] = node
        for input_name in node.input:
            children[input_name].append(node)

    def sole_child(name, op_type):
        nodes = children[name]
        return nodes[0] if len(nodes) == 1 and nodes[0].op_type == op_type else None

    def projected_tensor(name):
        nodes = []
        current = name
        for _ in range(3):
            shape = shapes.get(current)
            if shape is not None and len(shape) == 3:
                return current, nodes
            node = by_output.get(current)
            if node is None or node.op_type not in ("Reshape", "Transpose") or consumers[current] != 1:
                return None
            nodes.append(node)
            current = node.input[0]
        return None

    replacements = []
    for qk in graph.node:
        if qk.op_type != "MatMul" or consumers[qk.output[0]] != 1:
            continue
        softmax = sole_child(qk.output[0], "Softmax")
        if softmax is None or consumers[softmax.output[0]] != 1:
            continue
        av = sole_child(softmax.output[0], "MatMul")
        if av is None or av.input[0] != softmax.output[0] or consumers[av.output[0]] != 1:
            continue
        query_shape = shapes.get(qk.input[0])
        key_shape = shapes.get(qk.input[1])
        value_shape = shapes.get(av.input[1])
        output_shape = shapes.get(av.output[0])
        if not all((query_shape, key_shape, value_shape, output_shape)) or len(query_shape) != 4:
            continue
        if query_shape != value_shape or query_shape != output_shape:
            continue
        batch, heads, sequence, head_size = query_shape
        if batch != 1 or key_shape != [batch, heads, head_size, sequence]:
            continue

        query = projected_tensor(qk.input[0])
        key = projected_tensor(qk.input[1])
        value = projected_tensor(av.input[1])
        if not all((query, key, value)):
            if qk.name == "/att/MatMul_1":
                print("projection match", query, key, value)
            continue
        expected_projection = [batch, sequence, heads * head_size]
        if any(shapes.get(item[0]) != expected_projection for item in (query, key, value)):
            if qk.name == "/att/MatMul_1":
                print("projection shapes", [shapes.get(item[0]) for item in (query, key, value)], expected_projection)
            continue

        merge_transpose = sole_child(av.output[0], "Transpose")
        if merge_transpose is None or permutation(merge_transpose) != [0, 2, 1, 3] or consumers[merge_transpose.output[0]] != 1:
            if qk.name == "/att/MatMul_1":
                print("merge transpose", merge_transpose)
            continue
        flatten_nodes = children[merge_transpose.output[0]]
        if len(flatten_nodes) != 1 or flatten_nodes[0].op_type not in ("Flatten", "Reshape"):
            continue
        flatten = flatten_nodes[0]
        merge_shape = shapes.get(flatten.output[0])
        if merge_shape == expected_projection:
            tail = None
            attention_output = flatten.output[0]
        elif merge_shape == expected_projection[1:]:
            # The legacy exporter squeezes static B=1 before the output Gemm.
            # Keep that reshape and feed it Attention's standard 3-D output.
            tail = flatten
            attention_output = qk.output[0] + "_attention"
        else:
            continue

        attention = helper.make_node(
            "MultiHeadAttention",
            [query[0], key[0], value[0]],
            [attention_output],
            domain="com.microsoft",
            name=qk.name.replace("/MatMul", "/Attention"),
            num_heads=heads,
            scale=1.0,
        )
        removed = [qk, softmax, av, *query[1], *key[1], *value[1], merge_transpose]
        if tail is None:
            removed.append(flatten)
        replacements.append(Fusion(qk, attention, removed, tail))

    if not replacements:
        return 0, model

    removed = {node.name for fusion in replacements for node in fusion.removed}
    replacement_by_qk = {fusion.qk.name: fusion.attention for fusion in replacements}
    rewired_tails = {fusion.tail.name: fusion.attention.output[0] for fusion in replacements if fusion.tail is not None}
    nodes = []
    for node in list(graph.node):
        if node.name in rewired_tails:
            node.input[0] = rewired_tails[node.name]
        if node.name in replacement_by_qk:
            nodes.append(replacement_by_qk[node.name])
        elif node.name not in removed:
            nodes.append(node)
    graph.ClearField("node")
    graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input}
    used.update(value.name for value in (*graph.input, *graph.output))
    initializers = [tensor for tensor in graph.initializer if tensor.name in used]
    graph.ClearField("initializer")
    graph.initializer.extend(initializers)
    graph.ClearField("value_info")
    onnx.checker.check_model(model)
    return len(replacements), model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count, model = fuse_frequency_attention(onnx.load(args.input))
    if count != 4:
        raise RuntimeError(f"Expected four frequency attention subgraphs, matched {count}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"Fused {count} frequency attention subgraphs: {args.output}")


if __name__ == "__main__":
    main()
