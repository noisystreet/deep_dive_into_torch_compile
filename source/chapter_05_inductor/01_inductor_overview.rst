.. _inductor-overview:

==================
Inductor 概览
==================

.. tip::

   **Inductor 的名字来自"磁感线圈"（inductor）。**
   PyTorch 团队在命名时遵循了一个传统：用物理学名词命名编译器组件。Dynamo（发电机）、Inductor（电感）、Transformer（变压器）——PyTorch 的编译栈成了一组"电气工程"主题的命名集合。团队曾开玩笑说，如果以后要再做一个模块，应该叫 "Capacitor"（电容）。相比无聊的 "backend_v2"，这样的命名显然更有记忆点。

.. note::

   **Inductor 是三个组件中投入最大的——提交数量是 AOTAutograd 的 6.7 倍。**
   从入仓至今，Inductor 的提交次数约 8,787 次，占编译栈总提交（16,543 次）的 53%。其中约 1,709 次（~19.5%）是 bug fix，1012 次（~11.5%）是 revert。Inductor 的修改量最大并非意外——代码生成是最复杂、最容易出问题的环节。每次 PyTorch 新增一个 ATen 算子，Inductor 就需要为其添加 lowering 函数；每次 Triton 编译器有变动，Inductor 的 codegen 可能也需要跟着适配。相比之下，Dynamo（字节码分析）和 AOTAutograd（图分区）的接口更稳定，变动的频率也低得多。

Inductor 是 torch.compile 的默认编译器后端。它接收 AOTAutograd 分区后的 FX Graph，经过降级、融合、代码生成三个阶段，最终输出高效的 GPU（Triton）或 CPU（C++/OpenMP）代码。

这一节我们从整体上了解 Inductor 的架构和工作流程。

Inductor 在编译流水线中的位置
===================================

回顾前面几章：Dynamo 捕获 FX Graph → AOTAutograd 联合求导并分区 → 得到前向子图 + 反向子图。这两个子图各自独立进入 Inductor：

.. code-block:: text

   compile_fx_inner() 整体编排               ← 在 compile_fx.py 中
       │
       ├─ aot_autograd(
       │       decompositions=...,         ← 见第 4.6 节
       │   )
       │   输出: 前向子图 + 反向子图
       │
       ├─ post_grad_passes(fwd_gm)         ← 见第 4.7 节
       │  post_grad_passes(bwd_gm)
       │
       ├─ Lowering（降级）
       │      FX Graph → Inductor IRNode
       │      （lowering.py）
       │
       ├─ Scheduler（调度 + 融合）
       │      将兼容的节点融合为 FusedSchedulerNode
       │      （scheduler.py）
       │
       ├─ Codegen（代码生成）
       │      GPU → Triton 代码（codegen/triton.py）
       │      CPU → C++/OpenMP 代码（codegen/cpp.py）
       │
       └─ 编译 + 返回 callable
               编译生成的代码，
               返回 CompiledFxGraph

源码结构
============

Inductor 的代码全在 ``pytorch/torch/_inductor/`` 目录中。以下是核心文件的职能：

.. code-block:: text

   torch/_inductor/
   ├── compile_fx.py          # 主入口，编排编译流程（含 decomposition 配置）
   ├── decomposition.py       # 算子分解表（select_decomp_table）
   ├── graph.py               # GraphLowering：整个 Inductor 图的构建
   ├── ir.py                  # IRNode 定义（Pointwise, Reduction 等）
   ├── lowering.py            # 从 FX → IR 的降级函数注册
   ├── scheduler.py           # Scheduler：融合、调度、排序
   ├── fx_passes/             # FX 图优化 pass（pre_grad, post_grad 等）
   ├── codegen/               # 代码生成器
   │   ├── triton.py          # GPU → Triton 代码
   │   ├── cpp.py             # CPU → C++/OpenMP 代码
   │   └── wrapper.py         # Python wrapper 代码生成
   ├── pattern_matcher.py     # 模式匹配框架
   └── config.py              # Inductor 配置参数

主入口：compile_fx
=====================

``compile_fx`` 函数（在 ``compile_fx.py`` 中）是 Inductor 被调用的入口。AOTAutograd 在分区后调用它：

.. code-block:: python

   # pytorch/torch/_dynamo/backends/inductor.py
   @register_backend
   def inductor(*args, **kwargs):
       from torch._inductor.compile_fx import compile_fx
       return compile_fx(*args, **kwargs)

``compile_fx`` 内部调用 ``_compile_fx_inner``，后者是实际的编译核心函数。完整的流程包括 aot_autograd（含 decomposition 见第 4.6 节，FX Passes 见第 4.7 节）、lowering、scheduler、codegen 四个阶段。

.. code-block:: python
   :caption: pytorch/torch/_inductor/compile_fx.py（简化示意）

   def _compile_fx_inner(graph_module, example_inputs):
       # 1. aot_autograd: 联合求导 + decomposition（见第 4.6 节）+ 分区
       fw_module, bw_module = aot_autograd(
           graph_module,
           decompositions=select_decomp_table(),
           ...
       )
       
       # 2. post_grad: 在 lowering 之前优化（见第 4.7 节）
       fw_module = post_grad_passes(fw_module)
       bw_module = post_grad_passes(bw_module)
       
       # 3. Lowering: 创建 GraphLowering，逐步降级
       inductor_graph = GraphLowering(fw_module)
       inductor_graph.compile(fw_module, example_inputs)
       
       # 4. Scheduler: 融合和调度
       scheduler = Scheduler(inductor_graph.operations)
       scheduler.codegen()
       
       # 5. 返回编译后的函数
       return CompiledFxGraph(...)

这个流程的关键在于 ``GraphLowering`` 类（在 ``graph.py`` 中），它负责管理整个降级过程。当你遍历 ``graph_module`` 的节点时，``GraphLowering.register_operation`` 或 ``GraphLowering.call_function`` 会被每个 FX 节点调用，触发对应的 lowering 函数。

从 Define-by-Run IR 的角度看
===================================

Inductor 最重要的设计哲学是 **Define-by-Run IR**。这是 Inductor 设计文档（dev-discuss）中明确提出的理念：Inductor 的 IR 节点就是像 PyTorch 操作一样被逐级构建的，而不是像传统编译器那样从一个静态 IR 推导而来。

.. code-block:: text

   传统编译器:
       前端 → 静态 IR 构建 → IR 优化 → 代码生成
              ↑ IR 在优化过程中被分析和变换

   Inductor:
       FX Graph → 逐级构建 IRNode → scheduler 融合 → codegen
                   ↑ IRNode 是"构建"出来的，不是"推导"出来的

这意味着：

1. **每个 IRNode 的构造就是 lower 的过程**——没有单独的"IR 构建阶段"
2. **IRNode 直接捕获语义**——``Pointwise`` 节点知道它是逐元素操作，``Reduction`` 节点知道它是归约操作
3. **Codegen 直接关联到 IR 类型**——codegen 时只需要遍历 IRNode 列表，根据类型生成代码

关于 lowering、scheduler、codegen 的细节，我们会在本章接下来的小节中逐一深入。

Inductor 的两种模式
=========================

Inductor 有两种运行模式，通过 ``mode`` 参数控制：

.. code-block:: python

   # 默认模式：平衡编译时间与运行性能
   compiled_fn = torch.compile(fn, mode="default")

   # 最大自动调优模式：编译时间长，运行性能最好
   compiled_fn = torch.compile(fn, mode="max-autotune")

在 ``default`` 模式下，Inductor 使用预配置的 heuristic 做 tiling 和调度。在 ``max-autotune`` 模式下，它会枚举多组配置参数（block size、num warps 等），选择最快的一组。这个自动调优过程由 ``autotune_process.py`` 和 ``select_algorithm.py`` 实现。

小结
======

这一节从整体上了解了 Inductor 的工作流程：

- **Inductor 的核心流程**：aot_autograd（decomposition 见第 4.6 节，FX Passes 见第 4.7 节）→ Lowering → Scheduler → Codegen
- **主入口**：``compile_fx.py`` 中的 ``compile_fx`` / ``compile_fx_inner``
- **架构理念**：Define-by-Run IR，IRNode 是构建出来的而不是推导出来的
- **运行模式**：default（heuristic）和 max-autotune（枚举搜索）

接下来的小节将逐一深入 Inductor 的每个环节。
