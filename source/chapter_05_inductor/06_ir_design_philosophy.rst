.. _ir-design-philosophy:

==========================
Inductor IR 的设计思想
==========================

上一节（:ref:`ir-node`）介绍了 ``Pointwise`` 、 ``Reduction`` 等 IR 类型长什么样。本节回答三个更深层的问题： **Inductor IR 在编译流水线里扮演什么角色？为什么长成现在这样？与 FX、XLA、MLIR 等同类 IR 相比有何取舍？ **

对照 PyTorch v2.12.1 源码时，建议从 `ir.py <https://github.com/pytorch/pytorch/blob/v2.12.1/torch/_inductor/ir.py>`__ 顶部的 ``[Note: Inductor IR]`` （159–198 行）读起——那是官方对整套 IR 设计意图的浓缩说明。

IR 在流水线中的位置
======================

Inductor IR 不是单一「图 IR」，而是**FX Graph（算子级）与 Triton/C++ kernel（指令级）之间的多层中间表示** ：

.. code-block:: text

   FX Graph（aten 算子节点）
       │  GraphLowering：Define-by-Run lowering
       ▼
   Inductor IR（Buffer + Loop + Layout）
       │  Scheduler：依赖分析 + FusedSchedulerNode
       ▼
   LoopBody（inner_fn 的 FX 子图，调度期二次 trace）
       │  Codegen：OpsHandler → Triton / C++
       ▼
   可执行 kernel 源码

与第 5.1 节四层分工的对应关系：

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - 抽象层
     - 核心问题
     - 典型类型 / 机制
   * - 张量句柄
     - 这个值对应哪块存储？有没有 view？
     - ``TensorBox`` → ``View`` → ``StorageBox``
   * - 内存对象
     - Buffer 叫什么？shape/stride 是什么？
     - ``Buffer`` + ``Layout`` （ ``FlexibleLayout`` / ``FixedLayout`` ）
   * - 计算描述
     - 怎么算？能否继续融合？
     - ``Pointwise`` / ``Reduction`` / ``TemplateBuffer`` / ``ExternKernel``

FX 关心 **「调用了哪个 ATen op」 ** ；Inductor IR 关心**「哪块具名 buffer、如何按 index 计算」 ** 。第 5.2 节 FX Pass 改「算什么」，第 5.6 节 Scheduler 在 IR 层改「怎么融、怎么排」——正是因为 IR 已经切换到**内存 + 循环 ** 视角。

六条核心设计原则
======================

源码与注释里可以归纳出六条主线；理解它们，比死记类名更有用。

以内存为中心，而非以算子为中心
----------------------------------

``GraphLowering`` 维护**双 registry** （ ``graph.py`` ）： ``buffers`` 与 ``operations`` 并行增长。Scheduler 用 ``MemoryDep`` / ``ReadWrites`` （ ``dependencies.py`` ）判断融合合法性，依据的是 **buffer 读写关系** ，而不是 FX 节点名。

这与 XLA HLO、MLIR bufferization 之后的风格相近，但 Inductor 更 tightly 绑定 PyTorch 的 view、mutation 与 alias 语义。

Lazy Fusion：先描述计算，后分配内存
--------------------------------------

``Pointwise`` / ``Reduction`` 携带 ``inner_fn`` 与 ``ranges`` ，在 ``StorageBox.realize()`` 之前 **不必落盘** 。 ``IRNode.realize()`` 的语义是： **结束「还能继续 fuse 进来」的可能性** ，物化为 ``ComputedBuffer`` 并注册到 ``operations`` 。

逐元素链因此在 IR 层长期保持为 **未物化的表达式** ；遇到多个 consumer 或必须写内存时再 ``realize()``——这是 Inductor 与「先建完整静态 IR、再跑全局 pass」路线的重要分歧。

Functionalized Mutation
---------------------------

in-place 算子（ ``add_`` 等）在 IR 里 **不原地改 Buffer** ，而是新建 Buffer、用 ``mutation_renames`` 跟踪别名。Scheduler 据此维护依赖，融合分析可近似为 **纯函数数据流** 。

代价是 IR 与用户心智中的「同一块物理 tensor」不完全一一对应，调试 alias 问题需要多看一层 indirection。

Layout 是一等公民
--------------------

- ``FlexibleLayout`` ：Scheduler/Codegen 仍可选 stride（例如 channels_last）
- ``FixedLayout`` ： ``freeze_layout()`` 后冻结

View（ ``BaseView.make_loader`` ）通过 **reindexer 组合 loader** 访问底层存储，不复制数据。第 5.4 节提到：早期若跳过 virtualization、让 lowering 直接处理 view，Scheduler 融合率可从 80% 以上跌到 40% 以下——**IR 对 view 的建模质量直接决定优化上限 ** 。

LoopBody：循环体的二次 FX trace
---------------------------------

调度时 ``ComputedBuffer.simplify_and_reorder()`` 会把 ``inner_fn`` trace 成 ``LoopBody`` （`loop_body.py` 92 行起）——IR 内部的**micro-FX** ，便于索引化简与依赖提取。

取舍：复用 FX 基础设施，但 IR 表示分裂为「Python closure（lowering 期）」+「FX 子图（scheduling 期）」两层，阅读曲线更陡。

Scheduler 图 ≠ IR 图
-----------------------

Lowering 产出 ``operations: list[Operation]`` ；Scheduler 包装为 ``SchedulerNode`` ，融合时构造 **虚拟** 的 ``FusedSchedulerNode`` （ ``scheduler.py`` 1924 行起），依赖取各子节点并集， **不 rewrite IR 本身** 。

Codegen 对 Fused 节点 **重放 LoopBody** ，生成一个 kernel——与 MLIR dialect pass 重写、或 TVM 显式 schedule 变换是不同范式。

Define-by-Run Lowering
-------------------------

``GraphLowering`` 继承 ``torch.fx.Interpreter`` ，按 FX 拓扑 **执行** lowering 函数即时构造 IR（ ``make_pointwise`` 的 docstring 写明 *define-by-run IR*）。IR 生长顺序与 eager 执行顺序一致， **语义对齐成本低** ；全局重排则更多交给 FX pass 与 Scheduler。

设计取舍：优点与局限
======================

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - 维度
     - 优势
     - 局限
   * - PyTorch 语义
     - ATen schema 直接映射；broadcast/type promotion 在 lowering 封装
     - 深度绑定 ATen，难以作为通用 ML IR 导出
   * - 融合
     - Lazy Pointwise + loader 内联 + Scheduler 贪心融合，elementwise 链极强
     - Template 与 Loop 双轨， ``can_fuse`` 规则复杂
   * - 大算子
     - ``TemplateBuffer`` 挂 GEMM/Conv 模板与 epilogue fusion
     - 无 lowering 时 ``ExternKernel`` 回退 eager，性能不可预测
   * - 动态形状
     - ``sympy.Expr`` 贯穿 layout/dep，与 Dynamo symbolic shapes 一致
     - 符号化简与 codegen 成本高，首编译延迟大
   * - 后端
     - ``OpsHandler`` + virtualized ``ops.*`` ，Triton/C++/Halide 可插拔
     - IR 是 Python 对象图，缺少 MLIR/XLA 式可 dump 文本 IR
   * - 工程迭代
     - ``@register_lowering`` 加算子成本低
     - ``ir.py`` 体量大、概念多，新人上手难

一句话概括设计目标：

   **用 Define-by-Run 换与 eager 的对齐和工程速度；用 Lazy Fusion + Scheduler 分层换后端可插拔；代价是表示分裂、全局优化弱于 declarative 编译器、调试门槛高。**

与同类 IR 的比较
======================

总览
----

.. list-table::
   :header-rows: 1
   :widths: 16 14 22 22 26

   * - IR
     - 抽象层级
     - 核心单元
     - 融合主要发生在
     - 与 Inductor IR 关系
   * - FX Graph
     - 算子级
     - ``call_function(aten.*)``
     - pre/joint/post_grad FX pass
     - **上游输入**
   * - Inductor IR
     - 内存 + 循环级
     - Buffer + Pointwise/Reduction/Template
     - Scheduler + lazy inline
     - 本文主角
   * - LoopBody
     - 循环体级
     - FX 子图
     - ``simplify_and_reorder``
     - IR 内部二次表示
   * - Triton 生成代码
     - kernel 级
     - ``@triton.jit`` 块
     - SIMDKernel 扁平化 tiling
     - **下游输出**
   * - XLA HLO
     - 函数式算子级
     - ``HloInstruction``
     - HLO fusion pass
     - 更 declarative；算子集更封闭
   * - MLIR Linalg/MemRef
     - 多层 dialect
     - ``linalg.generic`` 等
     - dialect lowering + pass
     - 模块化强；PyTorch 默认不走此路径
   * - TVM TIR
     - 循环 + 存储级
     - ``BufferLoad/Store`` + loop
     - 显式 schedule
     - 最像 Inductor loop 层；schedule 由用户/AutoScheduler 控制

对比 FX Graph
-------------

.. code-block:: text

   维度          FX Graph                 Inductor IR
   ─────────────────────────────────────────────────────────
   关注点        调用了哪个 ATen op        哪块 buffer、如何按 index 算
   内存          隐式（FakeTensor meta）   显式 Layout + Buffer
   融合          代数替换（conv+relu）     循环 / kernel 合并
   动态性        节点 + meta               sympy 符号贯穿 shape/stride

FX 适合 **语义变换 ** （decomposition、pattern match）；Inductor IR 适合**内存与并行 ** （fusion、layout）。二者互补，不是谁替代谁。

对比 XLA HLO / StableHLO
--------------------------

**相似 ** ：函数式数据流（Inductor 通过 functionalize mutation 逼近）；融合是核心；大算子走特殊路径（XLA custom call ≈ ``TemplateBuffer`` ）。

**不同 ** ：

- HLO 是**可序列化 ** 的 declarative IR；Inductor 是**运行时构建 ** 的 Python 对象图
- XLA 在固定 IR 上跑**全局 pass pipeline** ；Inductor**先 lower 再 schedule** ，全局重排能力较弱
- StableHLO 偏 **export / 跨框架** ；Inductor 偏 **进程内 torch.compile 默认后端**

对比 MLIR
---------

MLIR 用 ** 多层 dialect + pass manager**组合优化；Inductor 用 ** 单一 ``ir.py`` + Python lowering 表 + Scheduler DAG**，没有通用 textual IR。

PyTorch 生态中的 **torch-mlir** 与 Inductor 是 **并行战略** ：Export/AOT 部署可走 MLIR； ``torch.compile`` 默认仍走 Inductor 快路径。选型取决于要 **与 PyTorch 同进程迭代** 还是 **可交换的中间产物** 。

对比 TVM TIR / Halide
-----------------------

- **TVM TIR** ：schedule 显式（ ``split`` / ``reorder`` / ``bind`` ）；Inductor 由 Scheduler**隐式 ** 贪心 + autotune
- **Halide** ：算法（ ``Func`` ）与 schedule**分离 ** ；Inductor**合并 ** 在一次 compile 里，普通用户不可单独重写 tile 顺序

Inductor 对**自动融合 elementwise 链 ** 更「开箱即用」；对手工极致 GEMM schedule 的工作流，深度调过的 TVM/CUTLASS 仍可能占优。

对比 Triton（生成产物）
-------------------------

Inductor IR**故意不 ** 表达 block size、mask、shared memory——这些在 `codegen/triton.py <https://github.com/pytorch/pytorch/blob/v2.12.1/torch/_inductor/codegen/triton.py>`__ 的 ``TritonKernel`` 里生成：

.. code-block:: text

   Inductor IR:  命名维度 loops + sympy 索引 + ops 原语
         │  SIMDKernel 扁平化 + tiling
         ▼
   Triton 代码:  program_id, block_ptr, tl.load/mask

因此：**Inductor IR 是 device-agnostic 的 loop IR；Triton 是 GPU 专用的低级 IR** 。Inductor 的价值在于自动从前者生成后者。

与本书其他章节的衔接
======================

.. code-block:: text

   第 2.1 节   Define-by-Run、四层分工
   第 5.2 节   FX pass（算子级） vs Scheduler（IR 级）
   第 5.3 节   GraphLowering 如何构造 IR
   第 5.4 节   TensorBox / StorageBox / realize
   第 5.5 节   Pointwise / Reduction / Template 类型 ← :ref:`ir-node`
   第 5.6 节   FusedSchedulerNode 与融合算法
   第 6 章     LoopBody → Triton/C++ codegen

小结
======

- Inductor IR 是 **内存 + 循环 + layout** 的多层表示，不是 FX 的简单降级副本
- **Lazy Fusion** 、 **functionalized mutation** 、 **FlexibleLayout** 是理解融合与 layout 优化的三根支柱
- **Scheduler 图与 IR 图分离 ** ：融合是合并调度节点，不是 rewrite IR
- 相对 XLA/MLIR：**更贴 PyTorch、更快迭代 ** ；相对 TVM/Halide：**更自动、更少 schedule 控制 **
- 相对 FX：** 算子名消失、buffer 名与依赖浮现**——这是进入 Inductor 后端后「看世界的方式」的根本变化
