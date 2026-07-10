"""自定义算子与 Kernel 示例

展示如何使用 torch.library API 注册自定义算子，
并使其与 torch.compile 无缝集成。
"""

import torch
from torch import library

# ============================================================
# 注册自定义算子
# ============================================================


# --- docs: define_op ---

import torch
from torch import library


# 定义自定义算子的实现（eager 模式）
def my_quadruple_impl(x):
    return x * 4


# 注册为 ATen 算子
library.define(
    "mylib::quadruple",
    "(Tensor x) -> Tensor",
    tags=torch.Tag.pt2_compliant_tag,
)
library.impl("mylib::quadruple", my_quadruple_impl, "CompositeImplicitAutograd")


if __name__ == "__main__":
    x = torch.randn(4)
    out = torch.ops.mylib.quadruple(x)
    print(f"输入: {x}")
    print(f"输出: {out}")

# --- docs: end ---

# ============================================================
# 自定义反向传播
# ============================================================


# --- docs: custom_grad ---

import torch


class MyQuadrupleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 4

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 4


library.impl("mylib::quadruple", MyQuadrupleFunction.apply, "AutogradCPU")
library.impl("mylib::quadruple", MyQuadrupleFunction.apply, "AutogradCUDA")


if __name__ == "__main__":
    x = torch.randn(4, requires_grad=True)
    out = torch.ops.mylib.quadruple(x)
    loss = out.sum()
    loss.backward()
    print(f"梯度: {x.grad}")

# --- docs: end ---

# ============================================================
# 注册 Decomposition
# ============================================================


# --- docs: decomposition ---

from torch._decomp import register_decomposition
from torch._ops import ops


@register_decomposition(ops.mylib.quadruple)
def quadruple_decomp(x):
    return x * 4  # 展开为 aten.mul


if __name__ == "__main__":
    # 验证 decomposition 被触发
    x = torch.randn(4)
    out = torch.ops.mylib.quadruple(x)
    print(f"Decomposition 结果: {out}")

# --- docs: end ---

# ============================================================
# 注册 Lowering
# ============================================================


# --- docs: lowering ---

from torch._inductor.lowering import register_lowering


@register_lowering(ops.mylib.quadruple)
def quadruple_lower(x):
    # 生成 Pointwise IRNode
    from torch._inductor.ir import Pointwise

    return Pointwise(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=lambda idx: ops.mul(
            ops.load(x, idx),
            ops.constant(4.0, x.get_dtype()),
        ),
        ranges=x.get_size(),
    )


if __name__ == "__main__":
    print("Lowering 注册完成")

# --- docs: end ---

# ============================================================
# 注册 Fallback
# ============================================================


# --- docs: fallback ---

from torch._inductor.lowering import make_fallback

make_fallback(ops.mylib.quadruple)

if __name__ == "__main__":
    print("Fallback 注册完成")

# --- docs: end ---

# ============================================================
# 自定义 Triton Kernel
# ============================================================


# --- docs: triton_kernel ---

import triton
import triton.language as tl


@triton.jit
def my_triton_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x * 4, mask=mask)


def quadruple_triton(x: torch.Tensor) -> torch.Tensor:
    """使用 Triton kernel 实现 x * 4。"""
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    my_triton_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    x = torch.randn(100, device="cuda")
    out = quadruple_triton(x)
    assert torch.allclose(out, x * 4), "Triton kernel 结果不正确"
    print(f"✓ Triton kernel 验证通过")
    print(f"  输入: {x[:5]}")
    print(f"  输出: {out[:5]}")

# --- docs: end ---
