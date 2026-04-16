# public API

from jageun.ir import Graph, GraphBuilder
from jageun.frontend.onnx_loader import load_onnx
from jageun.passes.fusion import FusionPass
from jageun.codegen.triton_codegen import TritonCodegen
from jageun.bench import benchmark, plot_roofline, BenchResult, _detect_hardware
import torch
from typing import Callable, Dict, List
from dataclasses import dataclass

@dataclass
class CompilationResult:
    graph: Graph
    fusion_groups: list
    kernels: Dict[str, Callable]
    bench_results: List[BenchResult] = None

    def run(self, inputs: Dict[str, torch.Tensor], weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Full executor coming in v0.2")

    def benchmark_vs_pytorch(self, example_inputs: Dict[str, torch.Tensor], example_weights: Dict[str, torch.Tensor]):
        hw = _detect_hardware()
        results = []

        for group_name, kernel_fn in self.kernels.items():
            group = next(g for g in self.fusion_groups
                        if g.name == group_name)
            a_node = group.root.inputs[0]
            b_node = group.root.inputs[1]
            a = example_weights.get(a_node.name,
                torch.randn(*a_node.output_type.shape, device='cuda', dtype=torch.float16))
            b = example_weights.get(b_node.name,
                torch.randn(*b_node.output_type.shape, device='cuda', dtype=torch.float16))

            M, K = a.shape
            K2, N = b.shape
            flops = 2 * M * N * K  # matmul FLOPs
            bytes_accessed = (a.numel() + b.numel()) * 2 

            r = benchmark(
                lambda: kernel_fn(a, b),
                flops=flops, bytes_accessed=bytes_accessed,
                name=f"jageun_{group_name}", hw=hw)
            results.append(r)
            print(r)

            def pytorch_ref():
                out = torch.nn.functional.linear(a, b.T)
                return torch.nn.functional.gelu(out)

            r_pt = benchmark(
                pytorch_ref,
                flops=flops, bytes_accessed=bytes_accessed,
                name=f"pytorch_{group_name}", hw=hw)
            results.append(r_pt)
            print(r_pt)

            speedup = r_pt.latency_p50_us / r.latency_p50_us
            print(f"  jageun speedup: {speedup:.2f}x")

        plot_roofline(results, hw, "jageun_roofline.png")
        self.bench_results = results
        return results


def compile(source: str, verbose: bool = True) -> CompilationResult:
    if verbose:
        print(f"[jageun] Loading {source}")
    graph, weights = load_onnx(source)

    if verbose:
        print(f"[jageun] Graph: {len(graph.nodes)} nodes")
    if verbose:
        print(graph.dump())

    if verbose:
        print("\n[jageun] Running fusion pass...")
    fusion_pass = FusionPass()
    groups = fusion_pass.run(graph)

    if verbose:
        print(f"\n[jageun] Generating Triton kernels...")
    codegen = TritonCodegen()
    kernels = {}
    for group in groups:
        kernel = codegen.compile_and_load(group)
        kernels[group.name] = kernel
        if verbose:
            print(f"  Compiled: {group.name}")

    return CompilationResult(
        graph=graph,
        fusion_groups=groups,
        kernels=kernels,
    )