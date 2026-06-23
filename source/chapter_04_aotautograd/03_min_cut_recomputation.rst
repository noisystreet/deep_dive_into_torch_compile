.. _min-cut-recomputation:

======================
Min-Cut 重计算
======================

上一节的朴素分区将所有被反向引用的前向中间结果都保存下来——这就是"最大化保存，最小化重计算"的策略。它的优点是实现简单，缺点是需要大量显存来保存中间结果。

AOTAutograd 提供了另一种分区策略：**min-cut 重计算分区** （min-cut rematerialization partition），它通过"用计算换内存"的方式减少显存占用。

核心思路
==============

Min-cut 重计算的核心思路是：**不在前向中保存所有中间结果，而是在反向中重新计算其中一部分**。

.. code-block:: text

   朴素分区（无重计算）:
       前向: x → sin(x) → cos(sin) → ... → output
               ↓保存  ↓保存  ↓保存         ↓保存
       反向: ... ← grad_sin ← grad_cos ← grad_output
            所有前向中间结果都保存

   Min-cut 分区（部分重计算）:
       前向: x → sin(x) → cos(sin) → ... → output
                     ↓保存  ↓丢弃         ↓保存
       反向: ... ← grad_sin ← cos(sin) ← grad_output
                               ↑
                          重计算，不保存

重计算在这里指：反向中需要的 ``cos(sin(x))`` 不在前向保存，而是在反向中从 ``sin(x)`` 重新计算一遍。

Min-cut 重计算是在"保存内存"和"增加计算"之间做 tradeoff：

.. code-block:: text

   保存中间结果:
       优点：反向直接使用，不需要额外计算
       缺点：消耗显存

   重计算中间结果:
       优点：节省显存
       缺点：增加反向中的计算量

min_cut_rematerialization_partition 的实现
=================================================

``min_cut_rematerialization_partition`` 函数实现在 ``pytorch/torch/_functorch/partitioners.py`` 第 3550 行。它的算法可以分为四个步骤。

**步骤 1：节点分类（classify_nodes）**

首先将联合图中的每个节点分为三类：

.. code-block:: text

   classify_nodes(joint_module)
       │
       ├─ forward_only: 只在前向中使用的节点（必须保存在前向）
       ├─ bwd_only: 只在反向中使用的节点（不存在保存问题）
       └─ share: 前向和反向都使用的节点（候选——保存还是重计算？）

候选节点（share）是 min-cut 算法的决策对象。分类的规则基于：

- 节点的使用者（users）分布：被前向节点使用、被反向节点使用、还是两者都使用？
- 节点的可重计算性：不是所有操作都可以重计算（例如随机数生成操作通常不可重计算）

**步骤 2：构建 min-cut 图**

对于候选节点集合，构建一个最大流最小割图。这个图中：

.. code-block:: text

   源点（source） = "保存"（不重计算）
   汇点（sink）   = "重计算"

   节点是联合图中的候选操作
   边上的容量表示"保存这个节点需要消耗的显存"

   如果一个节点在源点一侧 → 保存（不重计算）
   如果一个节点在汇点一侧 → 丢弃（反向中重计算）

**步骤 3：执行 max-flow 算法**

在构建好的图上执行最大流算法，找到最小割。最小割将节点划分为两组：保存 vs 重计算。

.. code-block:: text

                     min cut（最小割）
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   保存（不重计算）              丢弃（反向重计算）
   ┌────────────────┐       ┌────────────────┐
   │ sin(x)         │       │ cos(sin(x))    │
   │ x * sin(x)     │       │ x + sin(x)     │
   │ ...            │       │ ...            │
   └────────────────┘       └────────────────┘
   占用显存                    反向中重新计算

算法的核心目标是：**尽可能少保存中间结果，同时确保反向的额外计算开销不超过收益**。

**步骤 4：生成前向和反向子图**

根据最小割的结果，生成最终的前向和反向子图：

1. 前向图：保留原始前向节点 + 保存被割到"保存"侧的节点
2. 反向图：在反向的开头插入被割到"重计算"侧的节点的重新计算

完整算法流程（简化）：

.. code-block:: python
   :caption: pytorch/torch/_functorch/partitioners.py（简化示意）

   def min_cut_rematerialization_partition(joint_module, ...):
       # 1. 节点分类
       node_info = classify_nodes(joint_module)
       
       # 2. 构建 min-cut 图并求解
       #    核心是 max-flow min-cut 算法
       cut = solve_min_cut(joint_module, node_info)
       
       # 3. 根据 cut 结果生成前向子图
       fwd_module = create_forward(
           joint_module, 
           saved_nodes=cut.saved,      # 需要保存的节点
           recomputed_nodes=cut.remat,  # 需要重计算的节点
       )
       
       # 4. 根据 cut 结果生成反向子图
       bwd_module = create_backward(
           joint_module,
           saved_nodes=cut.saved,
           recomputed_nodes=cut.remat,  # 在反向中插入重计算
       )
       
       return fwd_module, bwd_module

什么操作可以重计算？
=========================

不是所有操作都可以安全地重计算。``partitioners.py`` 中维护了可重计算操作的集合：

.. code-block:: text

   可重计算:
   - 纯数学运算: sin, cos, add, mul, div, exp, log...
   - 形状变换: view, reshape, permute, transpose...
   - 逐元素操作: relu, sigmoid, tanh...
   
   不可重计算:
   - 随机数生成: dropout, rand, randn...
   - 有副作用的操作: in-place 修改（在功能化后除外）
   - I/O 操作: print, save...

可重计算性的判断逻辑在 ``classify_nodes`` 内部，基于操作的类型标签。

重计算和随机数生成
=========================

对于包含随机数生成的操作（如 dropout），重计算需要特别处理。因为重新执行 ``torch.dropout`` 会产生不同的随机掩码，导致结果不一致。

AOTAutograd 通过 ``PhiloxStateTracker`` 和 ``rng_decompositions`` 来解决这个问题：

.. code-block:: python

   from torch._decomp.decompositions_for_rng import PhiloxStateTracker, rng_decompositions

在联合追踪时，随机数生成操作会被分解为"种子 + 偏移 + 确定性的随机生成"三部分。重计算时使用相同的种子和偏移，确保结果一致。

内存与计算的权衡
====================

min-cut 重计算的效果取决于模型的具体结构和可重计算操作的比例。

.. list-table::
   :header-rows: 1

   * - 模型类型
     - 可重计算比例
     - 重计算效果
   * - ResNet 等 CNN
     - 高（大部分是 conv + relu）
     - 显著节省显存
   * - Transformer
     - 中（attention 中 softmax 可重计算）
     - 适度节省
   * - 含大量随机操作
     - 低
     - 效果有限

需要注意的是，重计算节省显存的效果在大 batch size 场景下最明显（因为中间结果的尺寸与 batch size 成正比）。

小结
======

这一节介绍了 min-cut 重计算分区策略：

- **核心思路**：用反向中的额外计算换取前向中的显存节省
- **四个步骤**：节点分类 → 构建 min-cut 图 → 执行 max-flow 算法 → 生成子图
- **可重计算性**：纯数学运算可重计算，随机数/副作用操作不可重计算
- **随机数处理**：通过 PhiloxStateTracker 分解种子和偏移，保证重计算结果一致

朴素分区和 min-cut 分区都是 AOTAutograd 内置的选择。实际使用中，min-cut 分区是默认行为——AOTAutograd 默认使用 ``min_cut_rematerialization_partition`` 作为分区函数。

下一节我们来看 Functionalization——AOTAutograd 如何处理 PyTorch 中的 in-place 操作和 view 操作。
