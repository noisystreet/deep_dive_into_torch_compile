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

为什么块级抽象是更优的 GPU 编程模型？根本原因在于 GPU 的硬件执行单元是 **warp**（32 个线程），而非单个线程。GPU 以 warp 为单位发射指令、合并访存、同步状态。线程级编程模型（CUDA）让开发者控制单个线程，但 GPU 硬件实际执行的是 warp——这中间存在一个 **语义鸿沟**：开发者为单个线程写的代码，实际上是以 warp 为单位执行的。调度器发射一条指令，warp 内所有线程同时执行。这个鸿沟导致了大量 CUDA 优化技巧（如确保相邻线程访问相邻地址以实现内存合并）本质上是 "绕过编译器，直接与硬件对话"。

Triton 的块级抽象直接消除了这个鸿沟：一个 block 对应一组 warp，编译器负责将 block 内的计算映射到 warp 内的线程。这使得编译器获得了完整的 block 级别视野——它能看到整个 block 内的所有内存访问模式、数据流和控制流，从而做出比人类更优的线程映射决策。这正是 Triton "不牺牲性能" 的底气来源：**限制开发者的表达力，换取编译器的全局视野**。

.. tip::

   **"限制表达力以换取自动化"** 是 Triton 设计哲学的精髓。开发者只能表达 "处理哪个数据块"，不能控制 "哪个线程处理哪个元素"。这听起来像是一种限制，但实际上它移除了编译器优化路径上的最大障碍——不确定性。当编译器确定知道 block 内每个元素的访问模式时，它可以做出激进的优化决策（如精确的寄存器分配、warp 级归约的代码生成），而这些在 CUDA 中需要开发者手动实现。

**自动内存合并（Automatic Memory Coalescing）** 。GPU 性能的关键之一是全局内存的合并访问（coalescing）——当 warp 内 32 个线程访问连续地址时，硬件将这些访问合并为一次或少数几次内存事务。在 CUDA 中，开发者必须手动确保 ``threadIdx.x`` 与地址之间的对应关系满足合并条件。在 Triton 中，编译器自动分析 block 内的访问模式并生成合并的指令。

Triton 实现自动内存合并的核心机制是 ``tl.arange`` ：它不是一个普通的 Python 函数，而是一个 **编译器内置原语**，在编译时被展开为具体的线程索引序列。编译器通过追踪 ``tl.arange`` 在地址计算中的使用方式，推断出 block 内所有线程的访存模式。例如，``tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))`` 中，地址相对于 ``x_ptr`` 是连续递增的，编译器立即判断这是一个完全合并的访问。而 ``tl.load(x_ptr + tl.arange(0, BLOCK_SIZE) * 2)`` 中的步长为 2，编译器仍然可以生成合并的指令，但内存事务的带宽利用率会减半——因为每个线程跳过了一个元素。

自动内存合并在规则模式（连续、固定步长）下表现完美，但在非规则模式（如 gather/scatter、索引数组间接寻址）下失效。当编译器无法推断出访存模式时，它会退化为最保守的策略——每个线程发射独立的加载指令。这种场景下，手写 CUDA 可以通过 shared memory 的显式管理来优化（例如将 gather 转换为 shared memory 的一次连续加载 + 后续的 shuffle），但 Triton 编译器目前还没有足够的信息来做出这种重构。

**自动调度（Automatic Scheduling）** 。Triton 编译器自动决定如何将 block 内的计算映射到 warp 上，并管理寄存器分配、指令流水线等底层细节。这意味着同一个 Triton kernel 在不同 GPU 架构上可以获得不同的 warp 映射策略——编译器为 Ampere 生成的代码可能将一个 block 映射为 4 个 warp，为 Hopper 生成的可 能映射为 8 个 warp，开发者无需修改源码。

这种自动调度的核心约束是寄存器和共享内存的预算。GPU 的寄存器文件是有限的（A100 上每个 SM 有 65536 个寄存器，分配到最多 2048 个线程），编译器必须在"更多线程 = 更好延迟隐藏"和"每个线程更多寄存器 = 减少溢出"之间做权衡。Triton 编译器通过分析 kernel 中的临时变量数量和生存期来估算寄存器需求，然后选择最优的 block size 和 warp 数量组合。这正是 ``num_warps`` 和 ``max_registers`` 这两个 autotune 参数所控制的——它们本质上是寄存器压力与并行度之间的旋钮。

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

Triton 编译器的架构采用经典的三阶段设计：前端解析、中间表示优化、后端代码生成。理解这个架构对使用 Triton 编写高性能 kernel 至关重要。了解 Triton 的编译器架构不仅能帮助你写出更快的 kernel，更能让你理解 Triton 设计者在"易用性与性能"之间做出的一系列精巧权衡。

编译流水线概述
--------------------

从 ``@triton.jit`` 修饰的 Python 函数到 GPU 上执行的 SASS 机器码，Triton 编译器经历以下转换阶段：

.. figure:: /_static/figures/triton_compiler_pipeline.svg
   :align: center
   :alt: Triton 编译器流水线
   :figwidth: 100%

   从 Python AST 到 cubin 的完整编译器流水线，分为前端、设备优化、NVIDIA 后端三阶段。

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
