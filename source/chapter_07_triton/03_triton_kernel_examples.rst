.. _triton-kernel-examples:

====================
Triton Kernel 示例
====================

这一节通过三个实际可运行的 Triton kernel 示例，展示从简单到复杂的编程模式。所有示例代码位于 ``examples/triton_kernel.py`` 。

示例 1：逐元素加法
============================

最简单的 Triton kernel——将两个张量逐元素相加：

.. synced-code-start:: add_kernel

   .. code-block:: python
      :linenos:

   import torch
   import triton
   import triton.language as tl


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

.. synced-code-end::

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

这个示例展示了融合的核心优势：中间结果（ ``x_centered`` 、 ``var`` ）在寄存器中传递，无需写回全局内存。

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

- **二维网格** ： ``pid_m`` 和 ``pid_n`` 分别对应输出矩阵的行和列块
- **分块累加** ：在 K 维度上迭代，每次加载 BLOCK_SIZE 大小的块
- **tl.dot** ：Triton 的矩阵乘原语，编译为 NVIDIA Tensor Core 指令
- **张量核心（Tensor Core）利用** ： ``tl.dot`` 自动使用 GPU 的 Tensor Core，这是 Triton 相比其他编译器的核心优势

Triton 编译器自动为 ``tl.dot`` 生成使用 Tensor Core 的 PTX 指令。这意味着在 Triton 中编写高效的矩阵乘法 kernel 不需要手动处理 warp-level matrix multiply 的复杂性——这些由编译器处理。

示例 4：矩阵乘法与后处理融合（Matmul + Bias + ReLU）
===============================================================

在实际的神经网络推理中，矩阵乘法后通常紧跟 bias 加法和 ReLU 激活函数。将这些操作融合到单个 kernel 中可以避免中间结果写回全局内存：

.. synced-code-start:: matmul_epilogue

   .. code-block:: python
      :linenos:

   import torch
   import triton
   import triton.language as tl


   @triton.jit
   def matmul_bias_relu_kernel(
       a_ptr,
       b_ptr,
       bias_ptr,
       c_ptr,
       M,
       N,
       K,
       stride_am,
       stride_ak,
       stride_bk,
       stride_bn,
       stride_cm,
       stride_cn,
       BLOCK_SIZE: tl.constexpr,
   ):
       """矩阵乘法 (A @ B) + Bias + ReLU 融合 kernel。

       参数:
           a_ptr: A 矩阵指针，形状 (M, K)
           b_ptr: B 矩阵指针，形状 (K, N)
           bias_ptr: bias 向量指针，形状 (N,)
           c_ptr: 输出矩阵指针，形状 (M, N)
           M, N, K: 矩阵维度
           stride_am, stride_ak: A 矩阵的行/列步幅
           stride_bk, stride_bn: B 矩阵的行/列步幅
           stride_cm, stride_cn: C 矩阵的行/列步幅
           BLOCK_SIZE: 分块大小
       """
       pid_m = tl.program_id(axis=0)
       pid_n = tl.program_id(axis=1)

       m_start = pid_m * BLOCK_SIZE
       n_start = pid_n * BLOCK_SIZE

       # 块索引
       m_offsets = m_start + tl.arange(0, BLOCK_SIZE)
       n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
       k_offsets = tl.arange(0, BLOCK_SIZE)

       # Mask
       mask_m = m_offsets < M
       mask_n = n_offsets < N

       # 累加器
       acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

       for k_start in range(0, K, BLOCK_SIZE):
           k_current = k_start + k_offsets
           mask_k = k_current < K

           # 加载 A 的块
           a_ptrs = a_ptr + (
               m_offsets[:, None] * stride_am + k_current[None, :] * stride_ak
           )
           a = tl.load(
               a_ptrs,
               mask=mask_m[:, None] & mask_k[None, :],
           )

           # 加载 B 的块
           b_ptrs = b_ptr + (
               k_current[:, None] * stride_bk + n_offsets[None, :] * stride_bn
           )
           b = tl.load(
               b_ptrs,
               mask=mask_k[:, None] & mask_n[None, :],
           )

           acc = tl.dot(a, b, acc)

       # --- Epilogue 融合 ---
       # 加载 bias 并广播到块的所有行
       bias = tl.load(bias_ptr + n_offsets, mask=mask_n)
       acc = acc + bias[None, :]

       # ReLU 激活
       acc = tl.where(acc > 0, acc, 0.0)

       # 存储最终结果
       c_ptrs = c_ptr + (m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn)
       tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


   def matmul_bias_relu(
       a: torch.Tensor,
       b: torch.Tensor,
       bias: torch.Tensor,
       BLOCK_SIZE: int = 64,
   ) -> torch.Tensor:
       """包装函数: 执行 MatMul + Bias + ReLU。

       BLOCK_SIZE 默认 64 以避免 shared memory 超限(大部分 GPU 限制 96KB)。
       """

       assert a.is_cuda and b.is_cuda and bias.is_cuda
       M, K = a.shape
       _, N = b.shape
       assert a.shape[1] == b.shape[0]
       assert bias.shape[0] == N

       c = torch.empty((M, N), device="cuda", dtype=a.dtype)

       grid = (
           triton.cdiv(M, BLOCK_SIZE),
           triton.cdiv(N, BLOCK_SIZE),
       )

       matmul_bias_relu_kernel[grid](
           a,
           b,
           bias,
           c,
           M,
           N,
           K,
           a.stride(0),
           a.stride(1),
           b.stride(0),
           b.stride(1),
           c.stride(0),
           c.stride(1),
           BLOCK_SIZE=BLOCK_SIZE,
       )
       return c

.. synced-code-end::

融合的关键优势：

- **减少内存带宽消耗** ：矩阵乘法的结果直接在寄存器中传递给 bias 加法和 ReLU，不需要写回再读取
- **降低 kernel launch 开销** ：三个操作（matmul + bias + relu）只需要一次 kernel launch
- **寄存器级数据复用** ：在 CUDA 中，如果分别执行这三个操作，矩阵乘法的结果需要经过全局内存往返

.. note::

**Inductor 中的 epilogue 融合。**
   Inductor 的 ``FusedSchedulerNode`` 会在图级别识别出 matmul 后跟 element-wise 操作的 pattern，将它们融合到同一个 Triton kernel 中。Inductor 生成的 matmul kernel 通常包含一个 epilogue 部分，用于处理 bias、residual、normalization 等后处理操作。

示例 5：融合 Softmax
=============================

Softmax 是注意力机制中的核心操作。一个高效的 fused softmax kernel 需要处理"求最大值 → 计算指数 → 求和 → 归一化"这四步，并且所有中间计算都在寄存器中完成：

.. synced-code-start:: fused_softmax

   .. code-block:: python
      :linenos:

   @triton.jit
   def fused_softmax_kernel(
       x_ptr,
       output_ptr,
       x_row_stride,
       output_row_stride,
       n_cols,
       BLOCK_SIZE: tl.constexpr,
   ):
       """逐行 fused softmax kernel。

       每行由一个 Triton program 处理。使用 "max 减" 技巧确保数值稳定。

       参数:
           x_ptr: 输入张量指针
           output_ptr: 输出张量指针
           x_row_stride: 输入张量的行步幅（bytes）
           output_row_stride: 输出张量的行步幅
           n_cols: 列数
           BLOCK_SIZE: 每行分块大小（需要 >= n_cols）
       """
       row_idx = tl.program_id(axis=0)
       row_start_x = row_idx * x_row_stride
       row_start_out = row_idx * output_row_stride

       col_offsets = tl.arange(0, BLOCK_SIZE)
       col_mask = col_offsets < n_cols

       # 加载一行数据
       x = tl.load(x_ptr + row_start_x + col_offsets, mask=col_mask)

       # 数值稳定的 softmax：
       # 1. 减去最大值，避免 exp(大正数) 溢出
       # 用 -1e38 替换 masked 元素，避免默认值 0 污染 max
       x_masked = tl.where(col_mask, x, -1e38)
       x_max = tl.max(x_masked, axis=0)
       x_sub = x - x_max

       # 2. 计算指数
       x_exp = tl.exp(x_sub)

       # 3. 求和（masked 元素贡献 0）
       x_exp_masked = tl.where(col_mask, x_exp, 0.0)
       x_sum = tl.sum(x_exp_masked, axis=0)

       # 4. 归一化
       y = x_exp / x_sum

       tl.store(output_ptr + row_start_out + col_offsets, y, mask=col_mask)

.. synced-code-end::

.. tip::

   **数值稳定性。**
   上述实现使用了 "max 减" 技巧：先减去最大值再计算指数，确保最大指数为 ``exp(0) = 1`` ，避免 ``exp(大正数)`` 导致的浮点溢出。这是所有数值稳定的 softmax 实现的标准做法。

更完整的版本还需要处理 ``BLOCK_SIZE > n_cols`` 的情况，以及支持二维网格来并行处理多行：

.. synced-code-start:: fused_softmax_2d

   .. code-block:: python
      :linenos:

   @triton.jit
   def fused_softmax_kernel_2d(
       x_ptr,
       output_ptr,
       x_row_stride,
       output_row_stride,
       n_rows,
       n_cols,
       BLOCK_SIZE: tl.constexpr,
   ):
       """二维网格 fused softmax kernel。

       支持多行并行处理，每行可被多个 block 分块处理。
       当 BLOCK_SIZE < n_cols 时，一行由多个 program 协作处理。

       参数:
           x_ptr: 输入张量指针
           output_ptr: 输出张量指针
           x_row_stride: 输入张量的行步幅
           output_row_stride: 输出张量的行步幅
           n_rows: 行数
           n_cols: 列数
           BLOCK_SIZE: 分块大小
       """
       row_idx = tl.program_id(axis=0)
       col_idx = tl.program_id(axis=1)

       # 计算起始位置
       col_start = col_idx * BLOCK_SIZE
       row_start_x = row_idx * x_row_stride
       row_start_out = row_idx * output_row_stride

       offsets = col_start + tl.arange(0, BLOCK_SIZE)
       mask = (row_idx < n_rows) & (offsets < n_cols)

       # 加载数据块
       x = tl.load(x_ptr + row_start_x + offsets, mask=mask)

       # Softmax 计算（单 block 覆盖整行）
       x_masked = tl.where(mask, x, -1e38)
       x_max = tl.max(x_masked, axis=0)
       x_exp = tl.exp(x - x_max)
       x_exp_masked = tl.where(mask, x_exp, 0.0)
       x_sum = tl.sum(x_exp_masked, axis=0)

       tl.store(
           output_ptr + row_start_out + offsets,
           x_exp / x_sum,
           mask=mask,
       )


   def fused_softmax(x: torch.Tensor, BLOCK_SIZE: int = 4096) -> torch.Tensor:
       """包装函数：在输入张量的最后一维执行 fused softmax。

       参数:
           x: 输入张量，形状 (..., n_cols)
           BLOCK_SIZE: Triton kernel 的分块大小

       返回:
           在最后一维执行 softmax 后的结果
       """
       assert x.is_cuda
       x_contiguous = x.contiguous()
       orig_shape = x_contiguous.shape
       n_cols = orig_shape[-1]

       # 展平前面的维度
       x_2d = x_contiguous.view(-1, n_cols)
       n_rows = x_2d.shape[0]

       output = torch.empty_like(x_2d)

       assert BLOCK_SIZE >= n_cols, f"BLOCK_SIZE ({BLOCK_SIZE}) 必须 >= n_cols ({n_cols})"

       grid = (n_rows,)

       fused_softmax_kernel[grid](
           x_2d,
           output,
           x_2d.stride(0),
           output.stride(0),
           n_cols,
           BLOCK_SIZE=BLOCK_SIZE,
       )

       return output.view(orig_shape)

.. synced-code-end::

.. note::

   **多 block softmax 的挑战。 **
   当 ``n_cols > BLOCK_SIZE`` 时，一行数据被多个 block 处理，需要跨 block 通信来计算全局 max 和 sum。这可以通过两遍扫描实现：第一遍计算局部 max/sum 并写回 scratch 空间，第二遍读取全局 max/sum 完成归一化。Triton 的 ``tl.atomic_add`` 可以用于安全地累加跨 block 的 sum。

Flash Attention 简化示例（概念性理解）
=================================================

Flash Attention 是 Triton 展示其能力的最具代表性的例子之一。虽然完整的 Flash Attention kernel 非常复杂（涉及 online softmax、分块矩阵乘法、因果掩码等），但其核心思想可以通过一个简化的概念性描述来理解。

核心思想
------------

Flash Attention 的关键洞见是：**将注意力计算分块进行，在 SRAM 中完成所有中间计算，避免读写 HBM** 。传统的注意力计算需要三步：

1. 计算 ``S = Q @ K^T`` ，写回 HBM
2. 读取 ``S`` ，计算 ``P = softmax(S)`` ，写回 HBM
3. 读取 ``P`` ，计算 ``O = P @ V`` ，写回 HBM

Flash Attention 通过分块 tiling 和 online softmax 将这三步合并：

.. code-block:: text

   传统实现（三步 HBM 往返）：
       Q, K, V (HBM)
         │
         ▼
       S = Q @ K^T → 写 HBM  ← 一次 HBM 写入
         │
         ▼
       P = softmax(S) → 写 HBM ← 二次 HBM 写入
         │
         ▼
       O = P @ V → 写 HBM    ← 三次 HBM 写入

   Flash Attention（单步在线计算）：
       Q, K, V (HBM)
         │
         ▼（逐块加载到 SRAM）
       ┌─────────────────────────────────────┐
       │ 在 SRAM 中完成：                      │
       │   S_block = Q_block @ K_block^T      │
       │   P_block = softmax(S_block)          │
       │   O_block += P_block @ V_block        │
       └─────────────────────────────────────┘
         │
         ▼
       O (HBM)  ← 仅一次 HBM 写入

Triton 中的简化的 Flash Attention kernel 结构
--------------------------------------------------------

.. code-block:: python

   @triton.jit
   def flash_attention_kernel(
       q_ptr, k_ptr, v_ptr, o_ptr,
       N_CTX, D_HEAD,
       BLOCK_SIZE: tl.constexpr,
   ):
       # 当前处理的 query 块
       pid_m = tl.program_id(axis=0)
       m_start = pid_m * BLOCK_SIZE
       
       # 初始化累加器和统计量
       acc = tl.zeros((BLOCK_SIZE, D_HEAD), dtype=tl.float32)
       m_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) - float("inf")
       z_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
       
       # 在 key/value 维度上分块迭代
       for pid_n in range(0, N_CTX, BLOCK_SIZE):
           n_start = pid_n * BLOCK_SIZE
           
           # 加载 Q、K、V 的块
           q = tl.load(q_ptr + offsets_q)
           k = tl.load(k_ptr + offsets_k)
           v = tl.load(v_ptr + offsets_v)
           
           # 计算 S = Q @ K^T
           s = tl.dot(q, tl.trans(k))
           
           # --- online softmax ---
           # 更新局部统计量
           m_new = tl.maximum(m_i, tl.max(s, axis=1))
           alpha = tl.exp(m_i - m_new)
           beta = tl.exp(s - m_new[:, None])
           
           # 更新累加器
           acc = acc * alpha[:, None] + tl.dot(beta, v)
           m_i = m_new
           z_i = z_i * alpha + tl.sum(beta, axis=1)
       
       # 最终归一化
       acc = acc / z_i[:, None]
       
       tl.store(o_ptr + offsets_o, acc)

.. warning::

   **上述代码是概念性说明，并非完整可运行的 Flash Attention 实现。 **
   完整的 Flash Attention kernel 需要处理：因果掩码（causal mask）、head 维度分块、dropout、更精确的 online softmax 算法（由 Rabe & Staats 2018 和 FlashAttention 论文提出）等。实际的 Triton Flash Attention 实现可以在 ``triton/benchmarks/tutorials/flash_attention.py`` 中找到。

Triton 的分块策略详解
==============================

理解 Triton 如何将 block 内的计算映射到 warp 上，对于编写高性能 kernel 至关重要。

从 block 到 warp 的映射
-------------------------------

每个 Triton program 处理一个数据 block。Triton 编译器自动将这个 block 的计算分配给 GPU 上的 warp（通常每个 block 分配 4 或 8 个 warp）。分配策略遵循以下原则：

.. mermaid::

   flowchart TD
       A["Triton Program (Block)<br/>例如: 处理 128x128 矩阵"] --> B["Warp 0<br/>行 0-15, 列 0-127"]
       A --> C["Warp 1<br/>行 16-31, 列 0-127"]
       A --> D["Warp 2<br/>行 32-47, 列 0-127"]
       A --> E["...<br/>更多 warps"]

       B --> F["Thread 0: 元素 (0,0)"]
       B --> G["Thread 1: 元素 (0,1)"]
       B --> H["Thread 31: 元素 (0,31)"]

共享内存分块（Shared Memory Tiling）
-----------------------------------------

对于矩阵乘法这样的操作，Triton 使用共享内存分块来减少全局内存访问。每个 warp 将数据从全局内存加载到共享内存，然后在共享内存上执行计算：

.. code-block:: text

   全局内存 (HBM)                    共享内存 (SRAM)                  寄存器
   ┌──────────────┐                ┌──────────────┐               ┌──────────┐
   │ A 矩阵        │  ──block──→   │ A_block       │  ──warp──→   │ Warp 的   │
   │ (M x K)      │               │ (BLOCK x BLK)│              │ 私有片段  │
   │              │               │              │              │          │
   │ B 矩阵        │  ──block──→   │ B_block       │  ──warp──→   │ 累加器    │
   │ (K x N)      │               │ (BLOCK x BLK)│              │ (fp32)   │
   └──────────────┘               └──────────────┘              └──────────┘

编译器自动处理：

- **Tiled 加载** ：从全局内存加载数据到共享内存，确保合并访问
- **同步** ：在 warp 之间同步共享内存的写入（通过 ``__syncthreads`` ）
- **寄存器分配** ：将共享内存中的数据分配到 warp 的寄存器文件
- **流水线** ：通过软件流水线隐藏内存延迟（overlap 计算和数据加载）

Block Size 的选择策略
-----------------------------

Block size 是 Triton kernel 最重要的性能参数之一。选择策略涉及多个权衡：

.. list-table::
   :header-rows: 1

   * - Block Size
     - 优势
     - 劣势
     - 适用场景
   * - 64
     - 寄存器压力小，适合小矩阵
     - Tensor Core 利用率低
     - 小 batch、小 head_dim
   * - 128
     - 良好的 Tensor Core 利用率
     - 中等寄存器压力
     - 大多数通用场景（默认选择）
   * - 256
     - 最大化 Tensor Core 吞吐
     - 高寄存器压力，可能 spill
     - 大矩阵、T4/A100 等高端 GPU

Inductor 的 autotune 进程会枚举这些 block size 选项，通过实际的 benchmark 选择最优值。

Block Size 与 num_warps 的关系
--------------------------------------

``num_warps`` 控制每个 Triton program 分配的 warp 数量。它与 block size 共同决定了每个线程处理的工作量：

.. code-block:: text

   每个线程处理的元素数 = BLOCK_SIZE * BLOCK_SIZE / (num_warps * 32)

   例如：
   - BLOCK_SIZE=128, num_warps=4: 每个线程处理 128*128/(4*32) = 128 个元素
   - BLOCK_SIZE=128, num_warps=8: 每个线程处理 128*128/(8*32) = 64 个元素

增加 ``num_warps`` 可以提高占用率（occupancy），但会减少每个线程的寄存器数量，可能导致寄存器溢出（spill）到本地内存。
