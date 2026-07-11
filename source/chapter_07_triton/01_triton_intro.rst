.. _triton-intro:

=============
Triton 简介
=============

.. note::

   **Triton 的作者 Philippe Tillet 只花了 6 个月就写出了第一个版本。 **
   Tillet 在 Harvard 读博士期间研究 GPU 编程，发现 CUDA 的编程模型过于底层。他的论文 "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" 获评 PPoPP 2019 的最佳论文。有趣的是，Triton 最初并没有被 OpenAI 重视——Tillet 加入 OpenAI 后以业余项目的形式继续开发，直到大家发现它可以让 Flash Attention 的编写变得异常简单，才被正式纳入推理堆栈的核心。如今 Triton 已经成为 AI 基础设施领域增长最快的编译器项目之一。

.. tip::

**Inductor 和 Triton 的耦合有多深？看提交历史就知道了。 **
   在 Inductor 的所有提交中，直接提及 "triton" 的超过 650 次（约 7.4%）。这些提交涵盖：适配 Triton 编译器新版本、利用 Triton 的新特性（如 Hopper 架构的 TMA 指令）、修复 Triton 编译器兼容性问题等。Triton 和 Inductor 是共同演进的——Triton 编译器每有变动（如升级 PTX 版本、改变 ``tl.arange`` 的行为），Inductor 都必须跟着适配，否则生成的 Triton 代码可能无法编译或性能下降。在 PyTorch 的 CI 中，Inductor 的测试依赖于 Triton 的特定版本，升级 Triton 需要协调两个仓库的发布节奏。

Triton 是一种面向 GPU 编程的语言和编译器，由 OpenAI 开发。在 PyTorch 2.x 中，Triton 是 Inductor 后端在 GPU 上的默认代码生成目标——Inductor 将 IRNode 翻译为 Triton 代码，然后由 Triton 编译器编译为 NVIDIA GPU 上的 PTX 指令。

Triton 的核心设计理念
=========================

Triton 的设计目标是** 让 GPU kernel 编程更简单，同时不牺牲性能 **。它通过以下方式实现：

** 块级编程（Block-level Programming）** 。Triton 让开发者以 "块"（block）为单位思考，而不是单个线程。每个 Triton program 处理一个数据块（如 1024 个元素），块内的操作自动并行化：

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

在 Triton 中，开发者不需要管理 ``blockIdx`` 、 ``threadIdx`` 、 ``blockDim`` 之间的映射——这些由 Triton 编译器自动处理。

**自动内存合并（Automatic Memory Coalescing）** 。GPU 性能的关键之一是全局内存访问的合并（coalescing）。在 CUDA 中，开发者需要手动确保相邻线程访问相邻地址。在 Triton 中，编译器自动分析块内的访问模式，生成合并的内存访问指令。

**自动调度（Automatic Scheduling）** 。Triton 编译器自动决定如何将块内的计算映射到 warp（线程束）上，并管理寄存器分配、指令流水线等底层细节。

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

- **2019** ：Triton 论文发表（Philippe Tillet 等，Harvard）
- **2021** ：OpenAI 开源 Triton，PyTorch 开始评估作为编译器后端
- **2022** ：PyTorch 2.0 发布，Inductor 默认使用 Triton 作为 GPU 代码生成目标
- **2023-2024** ：Triton 广泛用于 Flash Attention、vLLM 等推理框架，成为 AI 推理/训练的基础设施
- **2025+** ：Triton 持续演进，扩展对 Hopper 架构（SM90）的支持，改进编译器优化

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

Triton 的编译器架构
========================

Triton 编译器的架构采用经典的三阶段设计：前端解析、中间表示优化、后端代码生成。理解这个架构对使用 Triton 编写高性能 kernel 至关重要。

编译流水线概述
--------------------

从 ``@triton.jit`` 修饰的 Python 函数到 GPU 上执行的 SASS 机器码，Triton 编译器经历以下转换阶段：

.. mermaid::

   flowchart LR
       A["Python AST<br/>@triton.jit 函数"] --> B["类型推断<br/>Type Inference"]
       B --> C["Triton IR 生成<br/>(TTIR)"]
       C --> D["TTIR 优化<br/>循环展开·常量折叠<br/>内存合并分析"]
       D --> E["PTX 生成<br/>TensorCore 指令选择"]
       E --> F["ptxas<br/>NVIDIA 汇编器"]
       F --> G["SASS<br/>GPU 机器码"]
       G --> H["cubin<br/>可执行 kernel"]

       style A fill:#4a9eff,color:#fff
       style B fill:#6abf69,color:#fff
       style C fill:#ffa94d,color:#fff
       style D fill:#ffa94d,color:#fff
       style E fill:#d94f8a,color:#fff
       style F fill:#d94f8a,color:#fff
       style G fill:#b07cd8,color:#fff
       style H fill:#b07cd8,color:#fff

各阶段的作用如下：

1.**Python AST 解析** 。 ``@triton.jit`` 装饰器在 Python 层面将函数的 AST 捕获下来。Triton 编译器使用 Python 的 ``ast`` 模块解析函数体，提取其中的 ``tl.*`` 调用和控制流结构。这一步的关键是将 Python 语法树映射到 Triton 的内部表示。

2. **类型推断** 。Triton 是静态类型语言。编译器根据 kernel 参数的类型标注和 ``tl.constexpr`` 的值，推断所有中间变量的类型。例如 ``tl.arange(0, BLOCK_SIZE)`` 的类型是 ``int32`` 的向量， ``tl.load`` 的返回值类型由指针指向的数据类型决定。

3.**Triton IR（TTIR）生成** 。经过类型推断后，编译器将 AST 转换为 Triton 的中间表示——TTIR（Triton Intermediate Representation）。TTIR 是一种基于 MLIR（Multi-Level Intermediate Representation）框架的 dialect，它定义了 Triton 特有的操作（如 ``ttir.load`` 、 ``ttir.store`` 、 ``ttir.dot`` ）。MLIR 框架使得 Triton 可以复用 LLVM 生态中的大量优化 passes。

4.**TTIR 优化** 。在 TTIR 层面，编译器执行一系列与设备无关的优化：循环展开（将 ``tl.arange`` 展开为具体索引）、常量折叠、死代码消除、内存访问模式分析（为后续的合并访问做准备）。

5.**PTX 生成** 。优化后的 TTIR 被翻译为 NVIDIA 的 PTX（Parallel Thread Execution）中间表示。这一步的关键是 Tensor Core 指令选择——``tl.dot`` 被映射为 ``mma.sync`` 系列指令，常规计算被映射为 ``fadd`` 、 ``fmul`` 等标量或向量指令。

6.**ptxas 汇编** 。NVIDIA 的 ``ptxas`` 工具将 PTX 汇编为 GPU 架构相关的 SASS 机器码，最终打包为 cubin（CUDA binary）文件。

这种分层设计的优势在于：Triton 的前端可以独立于后端硬件演进。如果未来 Triton 要支持非 NVIDIA 的 GPU（如 AMD 的 CDNA 或 Intel 的 Xe 架构），只需要更换 PTX 生成之后的阶段，前端和 TTIR 优化可以复用。

与其他 DSL 编译器的比较
==============================

Triton 并非唯一面向深度学习领域的 DSL 编译器。为了更深入地理解 Triton 的设计取舍，我们将其与 TVM 和 Halide 进行对比：

.. list-table::
   :header-rows: 1

   * - 维度
     - Triton
     - TVM
     - Halide
   * - 核心抽象
     - 块级编程（block-level program）
     - 计算与调度分离（compute/schedule）
     - 纯函数式流水线（functional pipeline）
   * - 输入形式
     - Python DSL（@triton.jit）
     - Python DSL / Relay IR / Relax IR
     - C++ DSL / Python 绑定
   * - 中间表示
     - TTIR（基于 MLIR）
     - Relay / Relax / TIR（基于 TVM 自家框架）
     - 自有的 IR（基于 LLVM）
   * - 自动调度
     - 编译器自动映射到 warp
     - 基于 auto-scheduling（Ansor/ AutoTVM）
     - 需手动指定调度策略
   * - 后端支持
     - NVIDIA GPU（主要），AMD GPU（实验性）
     - NVIDIA/AMD/Intel/ARM/FPGA
     - x86/ARM/NVIDIA/AMD/Hexagon
   * - 优化粒度
     - block 级别，编译器负责内部优化
     - thread 级别，手动或自动选择调度
     - 像素/线程级别，调度完全显式
   * - 与 PyTorch 集成
     - 深度集成（Inductor 默认后端）
     - 通过 TVM PyTorch 前端集成
     - 无原生集成

.. tip::

   **三个项目的定位差异本质上是 "易用性 vs 通用性" 的权衡。 **
   Triton 选择了窄而深的路线：只针对 GPU 的块级编程，通过限制表达力换取编译器的自动化能力。TVM 选择了宽而广的路线：支持多种硬件后端，提供从自动调度到手动调优的完整工具链。Halide 则扮演了"先驱"的角色——它的计算/调度分离理念深刻影响了 Triton 和 TVM 的设计。

关键设计差异分析：

**调度策略**。Halide 要求开发者显式指定调度（在哪里并行、在哪里做向量化），这提供了最大的控制力但也增加了使用门槛。TVM 的 Ansor（Auto Scheduling）可以自动搜索调度方案，但搜索空间巨大。Triton 则采取了一种中间路线——开发者只需要指定 block size 等少数参数，编译器自动将 block 内的计算映射到 warp 上。

**硬件抽象层次**。TVM 的 TIR（Tensor Intermediate Representation）是设备无关的，通过 target 描述后端特性。Triton 的 TTIR 虽然也基于 MLIR，但语义上更接近 NVIDIA GPU 的编程模型（如显式的 program_id、warp 语义）。这使得 Triton 在 NVIDIA GPU 上的优化更直接，但移植到其他硬件时需要更多的适配工作。

**与宿主框架的关系**。Triton 与 PyTorch 的关系最为紧密——它不仅是 Inductor 的代码生成目标，还提供了 ``triton.testing`` 等工具用于在 PyTorch 张量上测试 kernel。TVM 与 PyTorch 的集成通过 ONNX 导出或 TVM 的 PyTorch 前端实现，存在一定的语义鸿沟。Halide 则基本不直接与 PyTorch 交互。

**实际生态影响**。从社区活跃度和采用率来看，Triton 的增长显著快于 TVM 和 Halide。这主要得益于：

- **PyTorch 的"捆绑"效应** ：每个 PyTorch 2.x 用户都自动获得了 Triton
- **学习曲线优势** ：Python DSL 比 TVM 的调度原语或 Halide 的 C++ API 更容易上手
- **聚焦的硬件目标** ：只针对 NVIDIA GPU 意味着 Triton 团队可以深入优化，而不是分摊到多个后端

.. note::

**TVM 的三阶段模型 vs Triton 的两阶段模型。**
   TVM 将编译分为三个阶段：前端（Relax/Relay）→ 张量中间表示（TIR）→ 后端代码生成。Triton 则合并了前端和中间表示——``@triton.jit`` 函数直接被解析为 TTIR，省去了高 Level 的图优化阶段。这是因为 Triton 的定位是"生成单个 kernel"，而不是"优化整个计算图"。图级别的优化在 Inductor 中完成，Triton 专注于 kernel 级别的代码生成。
