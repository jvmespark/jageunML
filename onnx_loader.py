# load ONNX models

import onnx
from onnx import numpy_helper, TensorProto
import numpy as np
import torch
from jageun.ir import Graph, Node, OpType, TensorType, GraphBuilder

ONNX_OP_MAP = {
    "Gemm": OpType.MATMUL,
    "MatMul": OpType.MATMUL,
    "Add": OpType.ADD,
    "Relu": OpType.RELU,
    "Gelu": OpType.GELU,
    "Sigmoid": OpType.SILU,
    "LayerNorm": OpType.LAYERNORM,
    "Transpose": OpType.TRANSPOSE,
    "Reshape": OpType.RESHAPE,
    "Cast": OpType.CAST,
}

ONNX_DTYPE_MAP = {
    TensorProto.FLOAT: torch.float32,
    TensorProto.FLOAT16: torch.float16,
    TensorProto.BFLOAT16: torch.bfloat16,
    TensorProto.INT8: torch.int8,
    TensorProto.INT32: torch.int32,
    TensorProto.INT64: torch.int64,
}

def load_onnx(path: str) -> tuple[Graph, dict]:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    graph = model.graph

    g = Graph()
    name_to_node: dict[str, Node] = {}
    weights: dict[str, np.ndarray] = {}

    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        weights[init.name] = arr
        dtype = ONNX_DTYPE_MAP.get(init.data_type, torch.float32)
        tt = TensorType(tuple(arr.shape), dtype)
        node = Node(OpType.CONST, [], attrs={'type': tt}, name=init.name)
        g.add_node(node)
        name_to_node[init.name] = node

    initializer_names = {i.name for i in graph.initializer}
    for inp in graph.input:
        if inp.name in initializer_names:
            continue
        shape, dtype = _parse_type_proto(inp.type)
        tt = TensorType(shape, dtype)
        node = Node(OpType.INPUT, [], attrs={'type': tt}, name=inp.name)
        g.add_node(node)
        g.inputs.append(node)
        name_to_node[inp.name] = node

    for onnx_node in graph.node:
        op_type = ONNX_OP_MAP.get(onnx_node.op_type)
        if op_type is None:
            print(f"  WARN: unsupported op {onnx_node.op_type}, skipping")
            continue

        input_nodes = [name_to_node[n] for n in onnx_node.input if n in name_to_node]
        attrs = _parse_attrs(onnx_node)

        out_name = onnx_node.output[0] if onnx_node.output else None
        node = Node(op_type, input_nodes, attrs=attrs, name=out_name)
        g.add_node(node)
        if out_name:
            name_to_node[out_name] = node

    g.type_infer()
    return g, weights

def _parse_type_proto(type_proto) -> tuple:
    t = type_proto.tensor_type
    shape = tuple(
        d.dim_value if d.dim_value > 0 else None
        for d in t.shape.dim
    )
    dtype = ONNX_DTYPE_MAP.get(t.elem_type, torch.float32)
    return shape, dtype

def _parse_attrs(onnx_node) -> dict:
    attrs = {}
    for attr in onnx_node.attribute:
        if attr.type == 1: attrs[attr.name] = attr.f 
        elif attr.type == 2: attrs[attr.name] = attr.i  
        elif attr.type == 3: attrs[attr.name] = attr.s 
        elif attr.type == 7: attrs[attr.name] = list(attr.ints) 
    return attrs