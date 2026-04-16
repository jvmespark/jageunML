# operator fusion passes

from jageun.ir import Graph, Node, OpType
from typing import List, Set, Optional
from dataclasses import dataclass

FUSION_PATTERNS = [
    ([OpType.MATMUL, OpType.ADD, OpType.GELU], "matmul_bias_gelu"),
    ([OpType.MATMUL, OpType.ADD, OpType.SILU], "matmul_bias_silu"),
    ([OpType.MATMUL, OpType.ADD, OpType.RELU], "matmul_bias_relu"),
    ([OpType.MATMUL, OpType.ADD], "matmul_bias"),
    ([OpType.ADD, OpType.GELU], "add_gelu"),
    ([OpType.ADD, OpType.RELU], "add_relu"),
]

@dataclass
class FusionGroup:
    name: str
    nodes: List[Node]
    root: Node
    epilogue: List[Node] 

class FusionPass:
    def run(self, graph: Graph) -> List[FusionGroup]:
        groups = []
        fused_nodes: Set[Node] = set()

        for node in graph.nodes:
            if node in fused_nodes:
                continue
            if node.op != OpType.MATMUL:
                continue

            group = self._try_match_from(node, fused_nodes)
            if group:
                groups.append(group)
                fused_nodes.update(group.nodes)
                for n in group.nodes:
                    n.attrs['fusion_group'] = group.name
                    n.attrs['fusion_root'] = (n == group.root)

        print(f"Fusion pass: found {len(groups)} fusion groups")
        for g in groups:
            node_names = " → ".join(n.op.value for n in g.nodes)
            print(f"  [{g.name}]: {node_names}")
        return groups

    def _try_match_from(self, anchor: Node, fused: Set[Node]) -> Optional[FusionGroup]:
        for pattern_ops, pattern_name in FUSION_PATTERNS:
            result = self._match_chain(anchor, pattern_ops, fused)
            if result:
                root = result[0]
                epilogue = result[1:]
                return FusionGroup(
                    name=pattern_name,
                    nodes=result,
                    root=root,
                    epilogue=epilogue,
                )
        return None

    def _match_chain(self, start: Node, ops: List[OpType], fused: Set[Node]) -> Optional[List[Node]]:
        if start.op != ops[0]:
            return None
        if start in fused:
            return None

        chain = [start]
        current = start

        for expected_op in ops[1:]:
            users = [u for u in current.users if u not in fused]
            if len(users) != 1:
                return None 

            next_node = users[0]
            if next_node.op != expected_op:
                return None
            if next_node in fused:
                return None

            if (next_node.output_type and current.output_type and next_node.output_type.shape != current.output_type.shape):
                if expected_op not in {OpType.ADD}:
                    return None

            chain.append(next_node)
            current = next_node

        return chain