.. _triton-kernel-examples:

====================
Triton Kernel 示例
====================

这一节通过三个实际可运行的 Triton kernel 示例，展示从简单到复杂的编程模式。所有示例代码位于 ``source/examples/triton_kernel.py``。

示例 1：逐元素加法
============================

最简单的 Triton kernel——将两个张量逐元素相加：

.. code-block:: python
   :caption: source/examples/triton_kernel.py

   @triton.jit
   def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(axis=0)
       block_start = pid * BLOCK_SIZE
       offsets = block_start + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements
       x = tl.load(x_ptr + offsets, mask=mask)
       y = tl.load(y_ptr + offsets, mask=mask)
       tl.store(output_ptr + offsets, x + y, mask=mask)

   def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
       output = torch.empty_like(x)
       n_elements = output.numel()
       grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
       add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
       return output

关键点：

- ``pid`` 标识当前处理的块
- ``tl.arange(0, BLOCK_SIZE)`` 生成块内的连续偏移
- ``mask`` 确保不超过数组边界
- 通过 ``grid`` 定义 launch 的 block 数量

示例 2：融合操作——向量归一化
======================================

将 "计算均值 → 计算方差 → 归一化" 融合为单个 kernel：

.. code-block:: python

   @triton.jit
   def normalize_kernel(x_ptr, output_ptr, n_elements, eps: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(axis=0)
       block_start = pid * BLOCK_SIZE
       offsets = block_start + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements

       x = tl.load(x_ptr + offsets, mask=mask)
       
       # 块内归约：计算均值和方差
       mean = tl.sum(x, axis=0) / n_elements
       x_centered = x - mean
       var = tl.sum(x_centered * x_centered, axis=0) / n_elements
       
       # 归一化
       output = x_centered / tl.sqrt(var + eps)
       tl.store(output_ptr + offsets, output, mask=mask)

这个示例展示了融合的核心优势：中间结果（``x_centered``、``var``）在寄存器中传递，无需写回全局内存。

示例 3：矩阵乘法（GEMM）
===============================

矩阵乘法是 Triton 的代表性用例，也是 Inductor 用 ``TemplateBuffer`` 调用的优化 kernel 的基础：

.. code-block:: python

   @triton.jit
   def matmul_kernel(
       a_ptr, b_ptr, c_ptr,
       M, N, K,
       stride_am, stride_ak,
       stride_bk, stride_bn,
       stride_cm, stride_cn,
       BLOCK_SIZE: tl.constexpr,
   ):
       pid_m = tl.program_id(axis=0)
       pid_n = tl.program_id(axis=1)
       
       # 当前块的起始位置
       m_start = pid_m * BLOCK_SIZE
       n_start = pid_n * BLOCK_SIZE
       
       # 累加器
       acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
       
       # K 维度上的迭代
       for k_start in range(0, K, BLOCK_SIZE):
           k_offsets = k_start + tl.arange(0, BLOCK_SIZE)
           
           # 加载 A 的块
           a_ptrs = a_ptr + m_start[:, None] * stride_am + k_offsets[None, :] * stride_ak
           a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :])
           
           # 加载 B 的块
           b_ptrs = b_ptr + k_offsets[:, None] * stride_bk + n_start[None, :] * stride_bn
           b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :])
           
           # 累加
           acc = tl.dot(a, b, acc)
       
       # 存储结果
       c_ptrs = c_ptr + m_start[:, None] * stride_cm + n_start[None, :] * stride_cn
       tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

这个 kernel 的关键设计：

- **二维网格**：``pid_m`` 和 ``pid_n`` 分别对应输出矩阵的行和列块
- **分块累加**：在 K 维度上迭代，每次加载 BLOCK_SIZE 大小的块
- **tl.dot**：Triton 的矩阵乘原语，编译为 NVIDIA Tensor Core 指令
- **张量核心（Tensor Core）利用**：``tl.dot`` 自动使用 GPU 的 Tensor Core，这是 Triton 相比其他编译器的核心优势

Triton 编译器自动为 ``tl.dot`` 生成使用 Tensor Core 的 PTX 指令。这意味着在 Triton 中编写高效的矩阵乘法 kernel 不需要手动处理 warp-level matrix multiply 的复杂性——这些由编译器处理。
