# numerical correctness tests

import pytest
import torch
import torch.nn.functional as F
from jageun.codegen.triton_codegen import TritonCodegen
from jageun.passes.fusion import FusionGroup
from jageun.ir import Node, OpType, TensorType

RTOL_FP16 = 1e-2
ATOL_FP16 = 1e-2

def make_test_tensors(M, K, N, device='cuda'):
    a = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
    b = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
    bias = torch.randn(N, device=device, dtype=torch.float16) * 0.01
    return a, b, bias

def pytorch_matmul_bias_gelu(a, b, bias):
    out = (a.to(torch.float32) @ b.to(torch.float32)).to(torch.float16)
    out = out + bias
    out = F.gelu(out.to(torch.float32)).to(torch.float16)
    return out

class TestMatmulBiasGelu:
    @pytest.fixture
    def kernel(self):
        mm_node = Node(OpType.MATMUL, [])
        add_node = Node(OpType.ADD, [mm_node])
        gelu_node = Node(OpType.GELU, [add_node])
        mm_node.output_type = TensorType((512, 4096), torch.float16)
        add_node.output_type = TensorType((512, 4096), torch.float16)
        gelu_node.output_type = TensorType((512, 4096), torch.float16)

        group = FusionGroup(
            name="matmul_bias_gelu",
            nodes=[mm_node, add_node, gelu_node],
            root=mm_node,
            epilogue=[add_node, gelu_node]
        )
        codegen = TritonCodegen()
        return codegen.compile_and_load(group)

    @pytest.mark.parametrize("M,K,N", [
        (64,  1024, 4096),    # small batch
        (128, 1024, 4096),   # standard batch
        (512, 1024, 4096),   # large batch
        (512, 1024, 4096),   # standard FFN size
        (1,   768,  3072),    # single token (inference batch=1)
        (128, 2048, 8192),   # large model FFN
    ])
    def test_correctness(self, kernel, M, K, N):
        a, b, bias = make_test_tensors(M, K, N)
        ref = pytorch_matmul_bias_gelu(a, b, bias)
        out = kernel(a, b, bias)
        torch.testing.assert_close(out, ref, rtol=RTOL_FP16, atol=ATOL_FP16, msg=f"Failed at M={M},K={K},N={N}")

    def test_no_nan(self, kernel):
        a, b, bias = make_test_tensors(512, 1024, 4096)
        out = kernel(a, b, bias)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_output_shape(self, kernel):
        a, b, bias = make_test_tensors(512, 1024, 4096)
        out = kernel(a, b, bias)
        assert out.shape == (512, 4096), f"Wrong shape: {out.shape}"

    def test_non_multiple_of_block_size(self, kernel):
        a = torch.randn(100, 1000, device='cuda', dtype=torch.float16) * 0.1
        b = torch.randn(1000, 3000, device='cuda', dtype=torch.float16) * 0.1
        bias = torch.randn(3000, device='cuda', dtype=torch.float16) * 0.01
        ref = pytorch_matmul_bias_gelu(a, b, bias)
        out = kernel(a, b, bias)
        torch.testing.assert_close(out, ref, rtol=RTOL_FP16, atol=ATOL_FP16)