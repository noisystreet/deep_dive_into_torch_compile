.. _triton-intro:

=============
Triton 简介
=============

.. note::

   **Triton 的作者 Philippe Tillet 只花了 6 个月就写出了第一个版本。**
   Tillet 在 Harvard 读博士期间研究 GPU 编程，发现 CUDA 的编程模型过于底层。他的论文 "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" 获评 PPoPP 2019 的最佳论文。有趣的是，Triton 最初并没有被 OpenAI 重视——Tillet 加入 OpenAI 后以业余项目的形式继续开发，直到大家发现它可以让 Flash Attention 的编写变得异常简单，才被正式纳入推理堆栈的核心。如今 Triton 已经成为 AI 基础设施领域增长最快的编译器项目之一。

.. tip::

   **Inductor 和 Triton 的耦合有多深？看提交历史就知道了。**
   在 Inductor 的所有提交中，直接提及 "triton" 的超过 650 次（约 7.4%）。这些提交涵盖：适配 Triton 编译器新版本、利用 Triton 的新特性（如 Hopper 架构的 TMA 指令）、修复 Triton 编译器兼容性问题等。Triton 和 Inductor 是共同演进的——Triton 编译器每有变动（如升级 PTX 版本、改变 `tl.arange` 的行为），Inductor 都必须跟着适配，否则生成的 Triton 代码可能无法编译或性能下降。在 PyTorch 的 CI 中，Inductor 的测试依赖于 Triton 的特定版本，升级 Triton 需要协调两个仓库的发布节奏。

Triton 是一种面向 GPU 编程的语言和编译器，由 OpenAI 开发。在 PyTorch 2.x 中，Triton 是 Inductor 后端在 GPU 上的默认代码生成目标——Inductor 将 IRNode 翻译为 Triton 代码，然后由 Triton 编译器编译为 NVIDIA GPU 上的 PTX 指令。

Triton 的核心设计理念
=========================

Triton 的设计目标是 **让 GPU kernel 编程更简单，同时不牺牲性能**。它通过以下方式实现：

**块级编程（Block-level Programming）**。Triton 让开发者以 "块"（block）为单位思考，而不是单个线程。每个 Triton program 处理一个数据块（如 1024 个元素），块内的操作自动并行化：

.. code-block:: python

   # Triton：以块为单位编程
   @triton.jit
   def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(axis=0)  # 当前块的 ID
       block_start = pid * BLOCK_SIZE
       offsets = block_start + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements
       x = tl.load(x_ptr + offsets, mask=mask)
       y = tl.load(y_ptr + offsets, mask=mask)
       tl.store(output_ptr + offsets, x + y, mask=mask)

对比 CUDA 的线程级编程：

.. code-block:: cuda

   // CUDA：以线程为单位编程
   __global__ void add_kernel(float* x, float* y, float* output, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) {
           output[idx] = x[idx] + y[idx];
       }
   }

在 Triton 中，开发者不需要管理 ``blockIdx``、``threadIdx``、``blockDim`` 之间的映射——这些由 Triton 编译器自动处理。

**自动内存合并（Automatic Memory Coalescing）**。GPU 性能的关键之一是全局内存访问的合并（coalescing）。在 CUDA 中，开发者需要手动确保相邻线程访问相邻地址。在 Triton 中，编译器自动分析块内的访问模式，生成合并的内存访问指令。

**自动调度（Automatic Scheduling）**。Triton 编译器自动决定如何将块内的计算映射到 warp（线程束）上，并管理寄存器分配、指令流水线等底层细节。

Triton 与 PyTorch 的关系
==============================

Triton 是 PyTorch 2.x 编译生态中的关键组件，但不是 PyTorch 的一部分。它是一个独立的开源项目，PyTorch 通过 ``triton`` Python 包引入它：

.. code-block:: text

   PyTorch 编译栈                  Triton 生态
   ┌──────────────────────┐      ┌────────────────────┐
   │ torch.compile        │      │ Triton 编译器       │
   │   → Inductor         │ ──→  │   → AST → PTX      │
   │   → Triton kernel    │      │   → PTX → SASS     │
   │     (@triton.jit)    │      │                    │
   └──────────────────────┘      │ Triton 语言        │
                                 │   tl.load/store    │
   Triton 在 PyTorch 中          │   tl.sin/cos/add   │
   的使用场景:                    │   tl.atomic_add    │
   • Inductor GPU 后端           │                    │
   • Flash Attention 实现         │ triton 包自带      │
   • 自定义 Triton kernel        │ 的 Python API     │
   • SDPA 的底层 kernel          └────────────────────┘

Triton 的历史与发展
=======================

- **2019**：Triton 论文发表（Philippe Tillet 等，Harvard）
- **2021**：OpenAI 开源 Triton，PyTorch 开始评估作为编译器后端
- **2022**：PyTorch 2.0 发布，Inductor 默认使用 Triton 作为 GPU 代码生成目标
- **2023-2024**：Triton 广泛用于 Flash Attention、vLLM 等推理框架，成为 AI 推理/训练的基础设施
- **2025+**：Triton 持续演进，扩展对 Hopper 架构（SM90）的支持，改进编译器优化

安装和验证
================

.. code-block:: bash

   # triton 随 PyTorch 2.x 自动安装
   pip install torch==2.12.1

   # 验证
   python -c "import triton; print(triton.__version__)"

如果 ``triton`` 可用，Inductor 的 GPU 后端会自动使用它。可以通过配置强制禁用：

.. code-block:: python

   import torch
   # 禁用 Triton，降级到 eager
   torch._inductor.config.triton.autotune = False
