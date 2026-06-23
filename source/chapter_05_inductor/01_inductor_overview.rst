.. _inductor-overview:

==================
Inductor 概览
==================

Inductor 是 torch.compile 的默认编译器后端。它接收 AOTAutograd 分区后的 FX Graph，经过降级、融合、代码生成三个阶段，最终输出高效的 GPU（Triton）或 CPU（C++/OpenMP）代码。

这一节我们从整体上了解 Inductor 的架构和工作流程。

Inductor 在编译流水线中的位置
===================================

回顾前面几章：Dynamo 捕获 FX Graph → AOTAutograd 联合求导并分区 → 得到前向子图 + 反向子图。这两个子图各自独立进入 Inductor：

.. code-block:: text

   AOTAutograd 的输出
       │  前向 FX Graph Module    反向 FX Graph Module
       ▼
   Inductor compile_fx_inner()
       │
       ├─ 1. FX Passes（图级别优化）
       │      在 FX Graph 上运行优化 pass
       │      （位于 fx_passes/ 目录）
       │
       ├─ 2. Lowering（降级）
       │      FX Graph 中的 call_function 节点
       │      被映射为 Inductor IRNode
       │      （lowering.py）
       │
       ├─ 3. Scheduler（调度 + 融合）
       │      分析 IRNode 之间的依赖关系，
       │      将兼容的节点融合为 FusedSchedulerNode
       │      （scheduler.py）
       │
       ├─ 4. Codegen（代码生成）
       │      GPU → Triton 代码（codegen/triton.py）
       │      CPU → C++/OpenMP 代码（codegen/cpp.py）
       │
       └─ 5. 编译 + 返回 callable
               编译生成的代码，
               返回 CompiledFxGraph

源码结构
============

Inductor 的代码全在 ``pytorch/torch/_inductor/`` 目录中。以下是核心文件的职能：

.. code-block:: text

   torch/_inductor/
   ├── compile_fx.py          # 主入口，编排编译流程
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

``compile_fx`` 内部调用 ``compile_fx_inner``，后者是实际的编译核心函数：

.. code-block:: python
   :caption: pytorch/torch/_inductor/compile_fx.py（简化示意）

   def compile_fx_inner(graph_module, example_inputs):
       # 1. FX Passes: 在 FX 图上运行优化
       graph_module = post_grad_passes(graph_module)
       
       # 2. Lowering: 创建 GraphLowering，逐步降级
       inductor_graph = GraphLowering(graph_module)
       inductor_graph.compile(gm, example_inputs)
       # 此时所有 FX 节点已被降级为 IRNode
       # IRNode 列表存储在 inductor_graph.operations 中
       
       # 3. Scheduler: 融合和调度
       scheduler = Scheduler(inductor_graph.operations)
       scheduler.codegen()  # 触发代码生成
       
       # 4. 返回编译后的函数
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

在后续的小节中，我们会逐一深入这些组件。

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

- **Inductor 的核心流程**：FX Passes → Lowering → Scheduler → Codegen
- **主入口**：``compile_fx.py`` 中的 ``compile_fx`` / ``compile_fx_inner``
- **架构理念**：Define-by-Run IR，IRNode 是构建出来的而不是推导出来的
- **运行模式**：default（heuristic）和 max-autotune（枚举搜索）

接下来的小节将逐一深入 Inductor 的每个环节。
