.. _fx-passes:

=================
FX Passes：图优化
=================

第 4 章完成了 AOTAutograd 侧的图变换（联合求导、分区、decomposition）。从本节开始，我们进入 Inductor 后端——而 **FX Passes** 正是 Inductor 编译流程中、在 lowering 之前对 FX Graph 做优化的第一道工序。

虽然 AOTAutograd 和 FX Passes 在 ``compile_fx_inner`` 中交替出现，但 FX Passes 的代码全部位于 ``torch/_inductor/fx_passes/``，属于 Inductor 职责。因此本书将其放在第 5 章而非第 4 章，避免读者在 AOTAutograd 章节中遇到 Inductor 专有逻辑。

为什么要在多个阶段执行 Pass？
======================================

读者第一次看到 ``compile_fx_inner`` 的流水线，常会问：**既然都是改 FX Graph，为什么不把所有 pass 攒到最后、对 lowering 前的那一张图统一跑一遍？**

答案是：**编译过程中间的图「形状」和「语义」在变**——同一张图在 Dynamo 出口、joint graph 内部、分区后的前向/反向子图上，能安全做的优化完全不同。FX Passes 的分阶段设计，本质是 **在正确的抽象层、正确的时机做正确的变换**。

编译过程中图的三次「变形」
----------------------------------

.. code-block:: text

   阶段 A：Dynamo 出口
       一张前向 FX Graph
       节点可能是 aten.layer_norm、aten._scaled_dot_product_attention 等高层算子
       │
       ▼  pre_grad_passes（本节）
       │
   阶段 B：AOTAutograd 内部
       joint graph → 分区 → decomposition（第 4 章）
       图膨胀：出现反向节点；高层算子可能被展开为基本算子
       │
       ▼  post_grad_passes × 2（前向子图 + 反向子图）
       │
   阶段 C：Lowering 入口
       两张（或一张推理）基本算子为主的 FX Graph
       │
       ▼  Scheduler 融合（第 5.6 节，IR 层）

**关键 invariant**：pass 只能作用于 **当前已经存在** 的图结构。在阶段 A 还没有反向子图，谈不上对 backward 做 Flash Attention 替换；在阶段 C 已经完成 decomposition，``layer_norm`` 可能已拆成十几个 ``mean``/``mul``，再在 FX 层匹配 ``layer_norm`` 模式为时已晚。

分阶段的设计逻辑
--------------------

.. list-table::
   :header-rows: 1
   :widths: 18 22 30 30

   * - 阶段
     - 时机
     - 此时图长什么样
     - 为什么在这里做
   * - ``pre_grad_passes``
     - AOTAutograd **之前**
     - 单张前向图，Dynamo 刚捕获
     - 减轻 joint trace 负担；去掉 ``x*1`` 等 autograd 不必追踪的冗余
   * - decomposition
     - AOTAutograd **内部**
     - joint graph 追踪时展开高层算子
     - 不是 FX pass 文件，但会 **改变节点集合**，是 pre/post 的分界事件
   * - ``post_grad_passes``
     - 分区 + decomposition **之后**，lowering **之前**
     - 前向/反向 **各一张** 基本算子图
     - 模式匹配（conv+relu、SDPA）在 decomp 后才稳定；前反向可 **分别** 优化

用第 2.1 节的话说：这是 **阶段专精** 在图优化层的体现——**autograd 负责造图，Inductor FX pass 负责在造图前后把图收拾干净**。

为什么不合并成「一次 pass 跑到底」？
------------------------------------------

假设只在 lowering 前跑一次大 pass，会遇到三类硬问题：

**1. Joint trace 成本**

AOTAutograd 要对前向代码做一次 **假反向** 追踪，生成 joint graph。Dynamo 捕获的图若充满 ``x + 0``、重复子表达式，joint graph 会 **同比膨胀**。``pre_grad_passes`` 在 trace 前做 CSE、常量折叠、恒等替换，是在 **降低 autograd 追踪的输入规模**——这是编译时间优化，不是运行时优化。

**2. 模式匹配的可见性**

许多 ``post_grad`` 规则匹配 **decomposition 之后** 的基本算子组合。例如 ``fuse_attention.py`` 匹配的是 SDPA 展开后的子图形态；若在 decomposition 之前跑，模式对不上。反之，``pre_grad`` 里的某些简化（如 BN folding 的早期形态）需要在 **高层算子还在** 时识别。

**3. 前向与反向的不同优化空间**

分区之后，``post_grad_passes(fwd_gm)`` 与 ``post_grad_passes(bwd_gm)`` **各跑一遍**。反向图常有 distinct 模式（重计算节点、梯度累积），与前向共享同一套 pass **函数**，但应用在 **不同图** 上。若在 joint graph 上统一优化再分区，要么规则无法区分前/反向语境，要么在 joint 上做 partition-aware 优化，复杂度爆炸。

因此流水线是 deliberate 的 **「pre → autograd 变形 → post × N」**，而不是疏忽导致的重复劳动。

与 IR 层融合的关系
--------------------

FX Passes 和 Scheduler 融合（第 5.6–5.7 节）是 **互补的两层**，不是重复：

.. code-block:: text

   FX Passes（图级代数）     Scheduler（IR 级内存/并行）
   ─────────────────────     ───────────────────────────
   conv + relu → 一个算子     两个 Pointwise → 一个 kernel
   SDPA → flash_attention    逐元素链 → 融合读写
   pad_mm 对齐 Tensor Core   决定 tile 与 launch 顺序

Pattern Matcher（第 5.8 节）大多挂在 ``post_grad_passes`` 里，因为它需要 **FX 节点的语义名字**（``aten.convolution``）。Scheduler 看不到这些名字，只看到 IRNode 类型。第 5.1 节的四层分工在此体现：**FX pass 改「算什么」，Scheduler 改「怎么算、怎么融」**。

FX Passes 分为两个阶段：``pre_grad_passes`` 在 AOTAutograd 之前运行，``post_grad_passes`` 在 lowering 之前、对分区后的前向/反向子图 **分别** 运行。

.. code-block:: text

   compile_fx_inner(gm, ...)
       │
       ├─ pre_grad_passes()          ← 在 AOTAutograd 之前
       │   简化图结构
       │
       ├─ aot_autograd(
       │       gm,
       │       decompositions=...,   ← 见第 4.6 节
       │   )
       │   输出: 前向子图 + 反向子图
       │
       ├─ post_grad_passes(fwd_gm)   ← 本节的优化 pass
       │  post_grad_passes(bwd_gm)   模式匹配、attention 融合
       │
       └─ Lowering → Scheduler → Codegen

pre_grad_passes
====================

``pre_grad_passes``（在 ``pytorch/torch/_inductor/fx_passes/pre_grad.py`` 中）在 AOTAutograd 之前运行。它的输入是 Dynamo 捕获的原始 FX Graph，尚未进行自动微分。

**设计目标**：在 joint trace 发生前 **瘦身**——让 AOTAutograd 追踪更少的节点，生成的 joint graph 更小，后续分区和 lowering 都更便宜。这里的优化偏 **结构性、与梯度无关**：

.. code-block:: text

   pre_grad_passes(gm)
       │
       ├─ 模式匹配替换：x * 1 → x, x + 0 → x
       ├─ 常量折叠：全常量子图在编译时求值
       ├─ 公共子表达式消除（CSE）：重复计算合并为一个节点
       └─ 死代码消除（DCE）

典型场景：Dynamo 捕获的图里常有 Python 语义遗留的恒等操作；若不提前消掉，autograd 会为每个 ``+ 0`` 多追踪一条 backward 边。

post_grad_passes
=====================

``post_grad_passes``（在 ``pytorch/torch/_inductor/fx_passes/post_grad.py`` 中）在 AOTAutograd 分区与 decomposition **之后**、lowering **之前**运行。``compile_fx_inner`` 对 **前向子图** 和 **反向子图** 各调用一次——同一套 pass 序列，两份输入。

**设计目标**：在 **基本算子粒度** 上做 **语义级** 替换与布局类优化。此时：

- decomposition 已展开高层算子，pattern 的 **匹配目标稳定**
- 前向/反向已分离，可对 backward 做专门规则（如重计算相关）
- 尚未 lowering，改 FX 节点仍比改 IRNode 便宜

.. code-block:: text

   post_grad_passes(gm)
       │
       ├─ 模式匹配替换
       │      conv + relu → conv_relu
       │      add + mul   → fma
       │
       ├─ Attention 模式匹配
       │      SDPA 子图 → Flash Attention kernel
       │
       ├─ 矩阵乘法 padding
       │      非对齐 mm → 对齐 mm（利用 Tensor Core）
       │
       ├─ 公共子表达式消除
       └─ 死代码消除

``joint_graph.py`` 等文件中的 pass 则在 **尚未分区** 的 joint 图上做少量变换——属于更特殊的插入点，数量远少于 pre/post 两主阶段。日常阅读源码时，**抓住 pre → autograd → post(fwd) + post(bwd) 这条主线即可**。

关键 FX Pass 文件
======================

这些 pass 的实现分布在 ``pytorch/torch/_inductor/fx_passes/`` 目录中：

.. code-block:: text

   fx_passes/
   ├── pre_grad.py            # autograd 之前：常量折叠、模式替换
   ├── post_grad.py           # lowering 之前：综合优化、CSE、DCE
   ├── fuse_attention.py      # 将 SDPA 匹配为 Flash Attention
   ├── pad_mm.py              # 将非对齐 mm padding 到对齐尺寸
   ├── binary_folding.py      # batchnorm + 后续操作的融合
   ├── joint_graph.py         # joint graph 级别的 pass
   ├── fusion_regions.py      # FX 级别的融合区域规划
   ├── group_batch_fusion.py  # 分组批处理融合
   ├── decompositions         # 分解相关 pass
   └── ...

关于 pattern matching 的具体机制（``@register_graph_pattern``），我们会在第 5.8 节 Pattern Matcher 中详细讨论。

小结
======

- **分阶段原因**：编译过程中图经历「前向 → joint/分区/decomp → 前向+反向子图」三次变形，pass 必须在 **对应形态** 上运行
- **pre_grad_passes**：autograd **之前** 瘦身，降低 joint trace 成本，做与梯度无关的结构性化简
- **decomposition**：在 AOTAutograd **内部**，改变节点集合，是 pre/post 的分界事件（第 4.6 节）
- **post_grad_passes**：lowering **之前** 对前向/反向 **分别** 做语义级模式匹配与布局优化
- **与 Scheduler 互补**：FX pass 改图结构，Scheduler 在 IR 层做 kernel 融合（第 5.6–5.8 节）
