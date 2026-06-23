.. _scheduler:

===========
Scheduler
===========

当 lowering 将 FX Graph 中的所有节点转换为 IRNode 后，Inductor 得到了一个扁平的 IRNode 列表。每个 IRNode 都是一个独立可执行的 kernel——但逐个 launch 它们是非常低效的。

Scheduler 的工作就是**分析这些 IRNode 之间的依赖关系，将兼容的节点融合成更大的 kernel，并规划最优的执行顺序**。

Scheduler 的职责
====================

Scheduler（定义在 ``pytorch/torch/_inductor/scheduler.py``）的核心职责有三：

1. **依赖分析**：分析 IRNode 之间的数据依赖，构建依赖图
2. **融合（Fusion）**：将兼容的 IRNode 合并为 ``FusedSchedulerNode``
3. **调度（Scheduling）**：确定最终 kernel 的执行顺序

.. code-block:: text

   Lowering 输出: [IRNode1, IRNode2, IRNode3, IRNode4, ...]
       │
       ▼
   Scheduler.__init__
       │
       ├─ 1. 创建 SchedulerNode
       │      将每个 IRNode 包装为 SchedulerNode
       │
       ├─ 2. 依赖分析
       │      构建 SchedulerNode 之间的依赖边
       │      （读/写依赖 + 别名解析）
       │
       ├─ 3. 融合循环
       │      基于启发式算法，将兼容的节点融合
       │      SchedulerNode + SchedulerNode → FusedSchedulerNode
       │
       └─ 4. codegen() 调用
              对每个 FusedSchedulerNode 调用对应后端的 codegen

SchedulerNode 与 FusedSchedulerNode
============================================

``SchedulerNode`` 是单个 IRNode 的包装器。它记录了这个节点读写了哪些 buffer、依赖哪些前置节点。

.. code-block:: text

   SchedulerNode(name="buf0")
       read_deps: {input_buffer_x}    # 读取的 buffer
       write_deps: {buf0}             # 写入的 buffer
       ir_node: Pointwise(...)        # 实际的 IR 节点
       group: (device='cuda:0', ...)  # 分组标识

``FusedSchedulerNode`` 是多个 ``SchedulerNode`` 融合的结果。它包含一组节点，它们将在同一个 kernel 中执行：

.. code-block:: python
   :caption: pytorch/torch/_inductor/scheduler.py（简化示意）

   class FusedSchedulerNode(BaseSchedulerNode):
       """一组被融合的 SchedulerNode"""
       nodes: list[BaseSchedulerNode]  # 融合前各个节点
       read_deps: set[str]             # 融合后的总读依赖
       write_deps: set[str]            # 融合后的总写依赖

       def get_name(self):
           # 融合后的名字
           return "_".join(n.get_name() for n in self.nodes)

融合算法
=============

Scheduler 的融合算法是一个迭代过程，核心逻辑在 ``decide_fusion`` 和 ``fuse_nodes_once`` 中。

.. code-block:: text

   融合循环（简化）:
       while True:
           fused = False
           for node in scheduler_nodes:
               # 尝试将 node 与它的每个邻居融合
               for neighbor in node.neighbors:
                   if can_fuse(node, neighbor):
                       fused_node = FusedSchedulerNode.fuse(node, neighbor)
                       scheduler_nodes.replace(node, fused_node)
                       fused = True
                       break
               if fused:
                   break
           if not fused:
               break

融合的决策（``can_fuse`` 或 ``fuse_if_speedup``）基于以下条件：

1. **设备相同**：两个节点必须在同一个设备上
2. **类型兼容**：Pointwise + Pointwise 总是可融合；Pointwise + Reduction 有条件融合（reduction 必须是外层）
3. **依赖无环**：融合后不能产生循环依赖
4. **性能收益**：``fuse_if_speedup`` 会估算融合后的性能提升，融合后收益为正才执行

其中条件 4 是可选的——在 ``max-autotune`` 模式下会做更激进的融合评估，在 ``default`` 模式下则使用启发式规则。

依赖分析
=============

Scheduler 在初始化时对每个 ``SchedulerNode`` 分析其读/写依赖。依赖信息被存储为 ``set[str]`` （buffer 名称的集合）：

.. code-block:: text

   节点 A: 写入 buf1, buf2  读取 buf0
   节点 B: 写入 buf3        读取 buf1, buf2
   节点 C: 写入 buf4        读取 buf3

   依赖图:
       buf0 → A → buf1, buf2 → B → buf3 → C → buf4
                ↓
             (A 和 B 可以融合，因为 A 写出的 buf1/buf2 被 B 读取)

当两个节点形成"生产者-消费者"链（一个写出的 buffer 被另一个读取），且两个节点的操作类型和范围兼容时，它们就是融合的候选。

``prune_deps()`` 方法在依赖图上运行死代码消除，移除不必要的依赖边。

Codegen 触发
=================

融合完成后，Scheduler 调用 ``codegen()`` 方法。这个方法遍历所有 ``SchedulerNode``（包括融合后的 ``FusedSchedulerNode``），对每个节点调用其目标后端（GPU/CPU）的 codegen 函数。

.. code-block:: text

   scheduler.codegen()
       │
       for node in self.nodes:
           │
           ├─ 根据 node 的设备选择后端
           │      backend = get_scheduling_for_device(node.device)
           │
           └─ backend.codegen(node)
                  GPU → TritonScheduling.codegen (codegen/triton.py)
                  CPU → CPPScheduling.codegen (codegen/cpp.py)

每个后端（Scheduling）负责将 IRNode 翻译为具体的代码。一个 FusedSchedulerNode 包含多个 IRNode，后端会将这些 IRNode 的 ``inner_fn`` 组合到同一个 kernel 的循环体内。

更多关于 codegen 的细节将在第 6 章讨论。

小结
======

这一节介绍了 Inductor 的 Scheduler：

- **核心职责**：依赖分析 → 融合 → 调度
- **SchedulerNode**：单个 IRNode 的包装器，记录读/写依赖
- **FusedSchedulerNode**：多个 SchedulerNode 融合的结果，共享同一个 kernel launch
- **融合条件**：设备一致、类型兼容、无循环依赖、性能收益为正
- **Codegen 触发**：融合后遍历节点，分派给对应后端的 codegen 函数
