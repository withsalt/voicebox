"""Inspect MatMul/Softmax attention patterns in a GRE-Net ONNX graph."""

import argparse
from collections import Counter
from pathlib import Path

import onnx
from onnx import shape_inference


def dimensions(graph, name):
    values = {value.name: value for value in (*graph.input, *graph.output, *graph.value_info)}
    if name not in values:
        return None
    tensor = values[name].type.tensor_type
    return [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in tensor.shape.dim]


def ancestry(graph, name, depth=4):
    producers = {output: node for node in graph.node for output in node.output}
    result = []
    for _ in range(depth):
        node = producers.get(name)
        if node is None:
            break
        result.append({"op": node.op_type, "name": node.name, "input": dimensions(graph, node.input[0]), "output": dimensions(graph, node.output[0]), "inputs": list(node.input)})
        name = node.input[0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    model = shape_inference.infer_shapes(onnx.load(args.model))
    graph = model.graph
    consumers = Counter(input_name for node in graph.node for input_name in node.input)
    by_output = {output: node for node in graph.node for output in node.output}
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        following = [candidate for candidate in graph.node if node.output[0] in candidate.input]
        if len(following) != 1 or following[0].op_type != "Softmax":
            continue
        softmax = following[0]
        projections = [candidate for candidate in graph.node if softmax.output[0] in candidate.input and candidate.op_type == "MatMul"]
        for projection in projections:
            transpose = by_output.get(node.input[1])
            print({
                "qk": node.name,
                "softmax": softmax.name,
                "av": projection.name,
                "q": dimensions(graph, node.input[0]),
                "k": dimensions(graph, node.input[1]),
                "v": dimensions(graph, projection.input[1]),
                "output": dimensions(graph, projection.output[0]),
                "transpose": transpose.op_type if transpose else None,
                "transpose_permutation": list(transpose.attribute[0].ints) if transpose and transpose.attribute else None,
                "transpose_consumers": consumers[node.input[1]],
                "q_ancestry": ancestry(graph, node.input[0]),
                "k_ancestry": ancestry(graph, transpose.input[0]) if transpose else None,
                "v_ancestry": ancestry(graph, projection.input[1]),
                "output_consumers": [{"op": candidate.op_type, "name": candidate.name, "inputs": list(candidate.input)} for candidate in graph.node if projection.output[0] in candidate.input],
            })


if __name__ == "__main__":
    main()
