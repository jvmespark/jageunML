# graph IR

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import torch

class OpType(Enum):
    # Memory ops
    INPUT = "input"      # graph input tensor
    CONST = "const"     # constant tensor (weights)
    OUTPUT = "output"    # graph output (sink)
    # Compute ops
    MATMUL = "matmul"    # C = A @ B
    ADD = "add"       # elementwise add (bias)
    RELU = "relu"      # elementwise relu
    GELU = "gelu"      # elementwise gelu
    SILU = "silu"      # elementwise silu
    SOFTMAX = "softmax"   # softmax over last dim
    LAYERNORM = "layernorm" # layer normalization
    TRANSPOSE = "transpose" # transpose dims
    RESHAPE = "reshape"   # reshape tensor
    CAST = "cast"      # dtype cast

@dataclass
class TensorType:
    shape: Tuple[Optional[int], ...]
    dtype: torch.dtype

    def num_elements(self) -> int:
        s = 1
        for d in self.shape:
            if d is None:
                return -1
            s *= d
        return s

    def num_bytes(self) -> int:
        el = self.num_elements()
        if el < 0:
            return -1
        return el * (torch.finfo(self.dtype).bits // 8)

class Node:
    _counter = 0

    def __init__(self, op: OpType, inputs: List[Node], attrs: Dict[str, Any] = None, name: str = None):
        self.op = op
        self.inputs = inputs          
        self.users = []             
        self.attrs = attrs or {}
        self.output_type: Optional[TensorType] = None 
        Node._counter += 1
        self.name = name or f"{op.value}_{Node._counter}"

    def __repr__(self):
        ins = ", ".join(n.name for n in self.inputs)
        ty = str(self.output_type.shape) if self.output_type else "?"
        return f"%{self.name} = {self.op.value}({ins}) : {ty}"

class Graph:
    def __init__(self):
        self.nodes: List[Node] = []
        self.inputs: List[Node]  = []
        self.outputs: List[Node] = []

    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        for inp in node.inputs:
            inp.users.append(node)
        return node

    def type_infer(self):
        for node in self.nodes:
            node.output_type = self._infer_node_type(node)

    def _infer_node_type(self, node: Node) -> TensorType:
        match node.op:
            case OpType.INPUT | OpType.CONST:
                return node.attrs['type']
            
            case OpType.MATMUL:
                a, b = node.inputs[0].output_type, node.inputs[1].output_type
                out_shape = a.shape[:-1] + (b.shape[-1],)
                return TensorType(out_shape, torch.float32)
            
            case OpType.ADD | OpType.RELU | OpType.GELU | OpType.SILU:
                return node.inputs[0].output_type
            
            case OpType.CAST:
                src = node.inputs[0].output_type
                return TensorType(src.shape, node.attrs['dtype'])
            
            case _:
                raise NotImplementedError(f"Type inference for {node.op}")

    def dump(self) -> str:
        lines = ["graph {"]
        for node in self.nodes:
            lines.append(f"  {node}")
        lines.append("}")
        return "\n".join(lines)

class GraphBuilder:
    def __init__(self):
        self.g = Graph()

    def input(self, name: str, shape: tuple, dtype=torch.float16) -> Node:
        n = Node(OpType.INPUT, [], attrs={'type': TensorType(shape, dtype)}, name=name)
        return self.g.add_node(n)

    def weight(self, name: str, shape: tuple, dtype=torch.float16) -> Node:
        n = Node(OpType.CONST, [], attrs={'type': TensorType(shape, dtype)}, name=name)
        return self.g.add_node(n)

    def matmul(self, a: Node, b: Node) -> Node:
        n = Node(OpType.MATMUL, [a, b])
        return self.g.add_node(n)

    def add(self, a: Node, b: Node) -> Node:
        n = Node(OpType.ADD, [a, b])
        return self.g.add_node(n)

    def gelu(self, x: Node) -> Node:
        n = Node(OpType.GELU, [x])
        return self.g.add_node(n)

    def build(self) -> Graph:
        self.g.type_infer()
        return self.g

def build_ffn_graph(seq: int = 512, d_model: int = 1024, d_ff: int = 4096) -> Graph:
    b = GraphBuilder()
    x = b.input("x", (seq, d_model))
    w1 = b.weight("w1", (d_model, d_ff))
    bias1 = b.weight("b1", (d_ff,))
    w2 = b.weight("w2", (d_ff, d_model))
    bias2 = b.weight("b2", (d_model,))

    mm1 = b.matmul(x, w1)
    add1 = b.add(mm1, bias1)
    act = b.gelu(add1)
    mm2 = b.matmul(act, w2)
    out = b.add(mm2, bias2)

    return b.build()