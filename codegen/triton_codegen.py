# emit Triton kernels from fusion groups

import triton
import triton.language as tl
import torch
from jageun.passes.fusion import FusionGroup
from jageun.ir import OpType
import textwrap

class TritonCodegen:
    def emit_kernel(self, group: FusionGroup) -> str:
        if group.name in ("matmul_bias_gelu", "matmul_bias_silu", "matmul_bias_relu", "matmul_bias"):
            return self._emit_matmul_epilogue(group)
        raise NotImplementedError(f"Codegen for {group.name}")

    def _emit_matmul_epilogue(self, group: FusionGroup) -> str:
        epilogue_code = self._build_epilogue(group)

        return textwrap.dedent(f"""
import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({{'BLOCK_M':128,'BLOCK_N':256,'BLOCK_K':64}}, num_warps=8, num_stages=3),
        triton.Config({{'BLOCK_M':64, 'BLOCK_N':128,'BLOCK_K':64}}, num_warps=4, num_stages=4),
        triton.Config({{'BLOCK_M':128,'BLOCK_N':128,'BLOCK_K':32}}, num_warps=4, num_stages=4),
        triton.Config({{'BLOCK_M':64, 'BLOCK_N':64, 'BLOCK_K':32}}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def {group.name}_kernel(
    A, B, C, bias,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = tl.make_block_ptr(
        A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0))
    b_ptrs = tl.make_block_ptr(
        B, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, boundary_check=(0, 1))
        b = tl.load(b_ptrs, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc)
        a_ptrs = tl.advance(a_ptrs, (0, BLOCK_K))
        b_ptrs = tl.advance(b_ptrs, (BLOCK_K, 0))

{epilogue_code}

    c_ptrs = tl.make_block_ptr(
        C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    tl.store(c_ptrs, acc.to(tl.float16), boundary_check=(0, 1))


def {group.name}(a: torch.Tensor, b: torch.Tensor,
                  bias: torch.Tensor = None) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    {group.name}_kernel[grid](
        a, b, c, bias if bias is not None else a,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=(bias is not None),
    )
    return c
""")

    def _build_epilogue(self, group: FusionGroup) -> str:
        lines = []
        indent = "    " * 2

        for node in group.epilogue:
            if node.op == OpType.ADD:
                lines += [
                    f"{indent}if HAS_BIAS:",
                    f"{indent}    bias_vec = tl.load(bias + offs_n, mask=offs_n < N)",
                    f"{indent}    acc = acc + bias_vec[None, :]",
                ]
            elif node.op == OpType.RELU:
                lines.append(f"{indent}acc = tl.maximum(acc, 0.0)")
            elif node.op == OpType.GELU:
                lines += [
                    f"{indent}k1 = 0.7978845608028654",
                    f"{indent}k2 = 0.044715",
                    f"{indent}acc = 0.5 * acc * (1.0 + tl.tanh(k1 * (acc + k2 * acc * acc * acc)))",
                ]
            elif node.op == OpType.SILU:
                lines.append(f"{indent}acc = acc * tl.sigmoid(acc)")

        return "\n".join(lines) if lines else f"{indent}pass"

    def compile_and_load(self, group: FusionGroup) -> callable:
        code = self.emit_kernel(group)
        namespace = {}
        exec(code, namespace)
        return namespace[group.name]