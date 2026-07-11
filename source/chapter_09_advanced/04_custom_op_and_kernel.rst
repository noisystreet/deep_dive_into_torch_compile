.. _custom-op-and-kernel:

=========================
自定义算子与 Kernel
=========================

在实际项目中，你可能需要引入 PyTorch 没有提供的自定义算子（custom operator），并将其与 torch.compile 无缝集成。这一节介绍如何在 torch.compile 的框架中注册和使用自定义算子。

注册自定义算子
==================

PyTorch 提供了 ``torch.library`` API 来注册自定义算子：

.. synced-code-start:: define_op

   .. code-block:: python
      :linenos:

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

.. synced-code-end::

关键点：

- ``mylib::quadruple`` 是算子的全名（namespace::op_name）
- ``tags=torch.Tag.pt2_compliant_tag`` 标记该算子与 torch.compile 兼容
- ``CompositeImplicitAutograd`` 表示算子可以用 PyTorch 的自动微分自动求导

如果算子需要自定义反向传播（不是纯复合操作），需要注册 ``autograd`` kernel：

.. synced-code-start:: custom_grad

   .. code-block:: python
      :linenos:

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

.. synced-code-end::

让自定义算子支持 torch.compile
========================================

为了让 torch.compile 正确处理自定义算子，需要：

**注册 Decomposition**

通过 decomposition 将自定义算子展开为已知算子：

.. synced-code-start:: decomposition

   .. code-block:: python
      :linenos:

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

.. synced-code-end::

这样 Inductor 在 lowering 时看到的是 ``aten.mul`` ，可以直接处理。

**注册 Lowering**

如果自定义算子有高效的 Triton 或 C++ 实现，可以直接注册 lowering：

.. synced-code-start:: lowering

   .. code-block:: python
      :linenos:

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

.. synced-code-end::

**注册 Fallback**

对于无法分解也无法单独 lowering 的算子，可以注册 fallback 让它回退到 eager：

.. synced-code-start:: fallback

   .. code-block:: python
      :linenos:

   from torch._inductor.lowering import make_fallback

   make_fallback(ops.mylib.quadruple)

   if __name__ == "__main__":
       print("Fallback 注册完成")

.. synced-code-end::

自定义 Triton Kernel
==========================

如果自定义算子需要手写 Triton kernel 获得最佳性能：

.. synced-code-start:: triton_kernel

   .. code-block:: python
      :linenos:

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

.. synced-code-end::

然后按照上面的方式注册这个算子的 lowering 或 decomposition。

通过 torch._dynamo.allow_in_graph 集成
============================================

对于 torch.compile 无法自动识别的操作，可以手动标记允许其在图中出现：

.. code-block:: python

   import torch._dynamo as dynamo

   @dynamo.allow_in_graph
   class MyCustomOp(torch.nn.Module):
       def forward(self, x):
           # 这个函数内部的 Python 操作会在图中保留
           # 不会被 graph break
           return x * 4

   @torch.compile
   def fn(x):
       return MyCustomOp()(x)

当 dynamo 遇到 ``allow_in_graph`` 标记的模块时，会将其作为一个整体捕获到图中，而不是尝试深入分析其内部实现。

使用 torch.library 注册自定义算子的最佳实践
===================================================

1. **总是标记 pt2_compliant_tag** 。如果算子符合 torch.compile 的约束（纯函数式、无 side effect），加上这个标签可以确保编译流畅。

2.**优先提供 decomposition** 。decomposition 让 Inductor 可以自动优化算子内部的算术操作。

3.**为性能关键路径提供 Triton kernel** 。如果 decomposition 生成的代码效率不够，手写 Triton kernel 可以获得最佳性能。

4.**测试 eager 和 compiled 模式的一致性** ：

   .. code-block:: python

      x = torch.randn(100, device='cuda')
      eager_result = my_custom_op(x)
      compiled_fn = torch.compile(my_custom_op)
      compiled_result = compiled_fn(x)
      torch.testing.assert_close(eager_result, compiled_result)
