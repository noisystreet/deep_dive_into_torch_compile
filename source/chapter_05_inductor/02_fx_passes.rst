.. _fx-passes:

=================
FX Passes：图优化
=================

第 4 章完成了 AOTAutograd 侧的图变换（联合求导、分区、decomposition）。从本节开始，我们进入 Inductor 后端——而 **FX Passes** 正是 Inductor 编译流程中、在 lowering 之前对 FX Graph 做优化的第一道工序。

虽然 AOTAutograd 和 FX Passes 在 ``compile_fx_inner`` 中交替出现，但 FX Passes 的代码全部位于 ``torch/_inductor/fx_passes/``，属于 Inductor 职责。因此本书将其放在第 5 章而非第 4 章，避免读者在 AOTAutograd 章节中遇到 Inductor 专有逻辑。

FX Passes 分为两个阶段：``pre_grad_passes`` 在 AOTAutograd 之前运行，``post_grad_passes`` 在 lowering 之前运行。

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

``pre_grad_passes``（在 ``pytorch/torch/_inductor/fx_passes/pre_grad.py`` 中）在 AOTAutograd 之前运行。它的输入是 Dynamo 捕获的原始 FX Graph，尚未进行自动微分。这个阶段的目标是简化图结构，减少 autograd 需要追踪的冗余节点：

.. code-block:: text

   pre_grad_passes(gm)
       │
       ├─ 模式匹配替换：x * 1 → x, x + 0 → x
       ├─ 常量折叠：全常量子图在编译时求值
       ├─ 公共子表达式消除（CSE）：重复计算合并为一个节点
       └─ 死代码消除（DCE）

post_grad_passes
=====================

``post_grad_passes``（在 ``pytorch/torch/_inductor/fx_passes/post_grad.py`` 中）在 AOTAutograd 分区之后、lowering 之前运行。前向子图和反向子图各自独立运行相同的 pass 序列。

此时图已经分离，可以针对前向和反向各自做专门优化。这个阶段相比 pre_grad 有更多的优化机会，因为经过了 decomposition 后，图中的算子都是基本算子，可以被 pattern matcher 精确匹配：

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

- **pre_grad_passes**：在 AOTAutograd 之前简化图结构，减少冗余节点
- **post_grad_passes**：在 lowering 之前进行数值优化和模式匹配
- 所有 FX Passes 的实现都在 Inductor 的 ``fx_passes/`` 目录中
