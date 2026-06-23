.. _triton-vs-cuda:

=================
Triton vs CUDA
=================

Triton 和 CUDA 都是 GPU 编程工具，但它们的抽象层级和设计哲学不同。这一节从多个维度对比两者，帮助理解什么时候用 Triton、什么时候用 CUDA。

抽象层级
=============

.. list-table::
   :header-rows: 1

   * - 维度
     - Triton
     - CUDA
   * - 编程单位
     - 块（block），每个块处理一个数据块
     - 线程（thread），每个线程处理一个元素
   * - 内存管理
     - 自动合并内存访问
     - 手动管理合并
   * - 并行粒度
     - 编译器决定 warp 映射
     - 开发者控制 thread/block/warp
   * - 性能调优
     - 主要调 BLOCK_SIZE、num_warps
     - 调 block/grid 维度、shared memory、warp 同步
   * - 学习曲线
     - 较低（Python 语法）
     - 较高（C++ 语法、GPU 架构知识）
   * - 编译方式
     - JIT 编译（@triton.jit）
     - AOT 编译（nvcc）或 JIT（NVRTC）

**Triton 代码更简洁**。同一功能的 kernel，Triton 代码行数通常是 CUDA 的 1/3 到 1/2。对比矩阵乘法的实现：

- CUDA 实现需要手动管理 shared memory tiling、warp-level matrix multiply、bank conflict 避免——约 200-300 行代码
- Triton 实现只需约 50 行（见 7.3 节），编译器自动处理 shared memory 和 Tensor Core 映射

性能对比
============

Triton 的性能通常接近手写 CUDA，在某些场景下甚至更优：

- **逐元素操作**：Triton 和 CUDA 性能相同（都是受内存带宽限制）
- **归约操作**：Triton 性能接近 CUDA（差异 < 5%）
- **矩阵乘法**：Triton 使用 ``tl.dot`` 调用 Tensor Core，性能与 cuBLAS 相当
- **复杂融合**：Triton 在有大量融合的场景下可能优于 CUDA（因为编译器可以自动优化跨操作的寄存器分配）

Triton 的局限性
====================

Triton 不是 CUDA 的完全替代。以下场景可能需要 CUDA：

**细粒度控制**。Triton 提供了高级抽象，但也屏蔽了对底层硬件的直接控制。如果需要：

- 操作 shared memory 的特定 bank
- 使用 warp-level shuffle 指令（``__shfl_sync``）
- 控制指令级的顺序和流水线

这些在 CUDA 中可以直接编码，但在 Triton 中无法直接表达。

**特殊指令**。Triton 的 ``tl.*`` API 是经过精心设计的子集。如果用到 CUDA 中较特殊的内置函数（如 ``__match_any_sync``、``__nanosleep``），可能没有对应的 Triton 原生操作。

**非 NVIDIA GPU**。Triton 目前主要支持 NVIDIA GPU（通过 PTX）。对 AMD GPU 的支持（通过 ROCm）在开发中，但成熟度不及 NVIDIA 平台。CUDA 只能运行在 NVIDIA GPU 上，而 Triton 理论上可以移植到其他 GPU 架构（因为它的 IR 设计是设备无关的）。

Triton 的优势场景
====================

- **快速原型开发**：想在 GPU 上运行自定义操作，不想花时间优化 CUDA kernel
- **融合 kernel**：在 Inductor 生成的代码中，多个操作被融合为一个 kernel——这正好是 Triton 的强项
- **自动调优**：Triton 的 autotune 机制比手动枚举 CUDA kernel 配置更高效
- **训练/推理框架集成**：Triton kernel 可以无缝嵌入 PyTorch 计算图

何时选择 CUDA
=================

- 需要极致的硬件性能（如手写汇编级优化）
- 使用 NVIDIA 专有特性（如 CUDA Graphs 的高级用法）
- 已有现成的 CUDA 库（如 cuDNN、cuBLAS）可以直接调用
- 目标平台不支持 Triton

实践建议
===============

**优先使用 Triton**。对于大多数场景，Triton 的性能已经足够好，开发成本更低。

**必要时回退到 CUDA**。如果 Triton 无法满足需求（性能不达标或功能不支持），再考虑 CUDA。

**混合使用**。一个项目中可以同时使用 Triton 和 CUDA——Inductor 本身就是这么做的：大部分 kernel 用 Triton 生成，矩阵乘法等性能关键路径通过 ``TemplateBuffer`` 调用 cuBLAS 或手写 CUDA kernel。
