.. _joint-forward-backward:

=========================
前向与反向联合求导
=========================

.. tip::

   **AOTAutograd 的名字为什么叫 "AOT"？**
   "AOT" 代表 Ahead-of-Time（提前）。这里的"提前"是相对于 eager 模式而言的——在 eager 模式下，autograd 的 tape 是在前向传播过程中实时构建的；而 AOTAutograd 在模型执行**之前**（编译阶段）就完成了联合图的创建和分析。这有点像"先画好地图再出发"而不是"边走边记路"。这种提前分析使 AOTAutograd 能够看到前向和反向的全局联系，从而做出更好的优化决策。

.. note::

   **AOTAutograd 是三个组件中提交最少但最稳定的。**
   在 PyTorch 编译栈的三个核心模块中，AOTAutograd 的提交次数只有 1,317 次，约为 Inductor（8,787 次）的 15%、Dynamo（6,439 次）的 20%。这并非因为它不重要——而是因为 AOTAutograd 是一个**中间层**，接口相对固定。它的核心逻辑（joint graph 创建、functionalization、分区）在 2023 年初就已经基本定型，后续的提交主要是 bug fix 和对新算子的支持适配。与此对比，Inductor 需要持续迭代代码生成策略以覆盖新算子，Dynamo 需要持续适配 CPython 新版本的字节码变化——AOTAutograd 夹在两者之间，反而是最不需要频繁改动的一层。

AOTAutograd 的全称是 **Ahead-of-Time Autograd** ——"提前"的自动微分。它在模型实际运行之前，通过追踪 autograd 的计算过程，生成一张包含**前向（forward）和反向（backward）的联合计算图**。

AOTAutograd 的源码位于 ``pytorch/torch/_functorch/`` 目录，核心代码在 ``aot_autograd.py`` 和 ``_aot_autograd/`` 子目录中：

.. code-block:: text

   torch/_functorch/
   ├── aot_autograd.py              # 主入口：aot_function, aot_export_module
   └── _aot_autograd/
       ├── graph_capture.py         # 图捕获：两条 dispatch 路径
       ├── graph_capture_wrappers.py # create_joint 联合图创建
       ├── graph_compile.py         # 编译阶段编排
       ├── functional_utils.py      # to_fun/from_fun 功能化
       ├── collect_metadata_analysis.py # 收集元数据
       └── runtime_wrappers.py      # 运行时包装器

AOTAutograd 在编译流水线中的位置
========================================

回顾第 2 章的流水线：Dynamo 输出 FX Graph 后，传递给 AOTAutograd：

.. code-block:: text

   Dynamo 输出
       │  FX Graph + Guards
       ▼
   AOTAutograd 主入口
       │
       ├─ 是否需要求导?
       │   ├─ 是（训练）→ aot_dispatch_autograd_graph
       │   │     1. 功能化输入
       │   │     2. 创建联合前向/反向图
       │   │     3. 图分区 → 前向子图 + 反向子图
       │   │     4. 分别编译前向和反向
       │   │
       │   └─ 否（推理）→ aot_dispatch_base_graph
       │         1. 功能化输入
       │         2. 只编译前向图
       │
       ▼
   Inductor 后端
       前向 kernel + 反向 kernel

两条路径在 ``pytorch/torch/_functorch/_aot_autograd/graph_capture.py`` 中实现：

- ``aot_dispatch_autograd_graph``：训练路径，涉及联合求导和图分区
- ``aot_dispatch_base_graph``：推理路径，只需编译前向

我们用 ``TORCH_LOGS`` 来观察 AOTAutograd 的痕迹。运行以下代码：

.. code-block:: bash

   TORCH_LOGS="+aot" python -c "
   import torch
   def fn(x, y):
       return (x * y).sum()
   
   compiled_fn = torch.compile(fn, fullgraph=True)
   x = torch.randn(3, requires_grad=True)
   y = torch.randn(3)
   result = compiled_fn(x, y)
   result.backward()
   "

日志中会看到 ``aot_graphs`` 输出的前向图和反向图信息。

为什么需要"联合"求导？
=========================

传统 PyTorch 的 eager 模式在反向传播时是逐操作执行的：每个 ``autograd.Function`` 对象的 ``backward`` 方法被依次调用。这种方式的局限在于：

1. **看不到全局图**：每个 backward 调用只知道自己的输入和输出，不知道上下游的优化机会
2. **无法做跨操作融合**：反向中的连续操作也无法融合
3. **无法做重计算规划**：无法决定"哪些中间结果保存、哪些重计算"

AOTAutograd 解决这些问题的办法是：**在执行自动微分之前，先通过 tracing 生成一张包含前向和反向的联合图**，在这张图上可以做全局分析和优化。

create_joint：联合图的创建
====================================

联合图的创建由 ``graph_capture_wrappers.py`` 中的 ``create_joint`` 函数完成。它的核心逻辑是：

.. code-block:: python
   :caption: pytorch/torch/_functorch/_aot_autograd/graph_capture_wrappers.py（简化示意）

   def create_joint(fn, primals_descs, *, aot_config):
       def inner_fn(primals, tangents):
           # 1. 执行前向传播（使用 FakeTensor）
           outs, tangent_mask = fn(*primals)
           
           # 2. 筛选需要梯度的输出
           outs_to_grad = [
               o for needs_tangent, o in zip(tangent_mask, outs) 
               if needs_tangent
           ]
           
           # 3. 执行反向传播追踪
           #    用 autograd.grad 计算梯度，
           #    同时用 proxy tensor 记录所有操作
           grad_outs = torch.autograd.grad(
               outs_to_grad, grad_primals, 
               grad_outputs=tangents,
           )
           
           # 4. 返回前向结果 + 梯度
           return outs, grad_outs
       
       return inner_fn

流程分解如下：

.. code-block:: text

   输入: primals（前向输入）+ tangents（反向起始梯度）
       │
       ▼
   第 1 步: 执行 fn(*primals)
       │  用 FakeTensor 执行前向函数
       │  记录所有前向操作到 FX Graph
       ▼
   输出: outs + tangent_mask
       │  tangent_mask 标记哪些输出需要梯度
       ▼
   第 2 步: 筛选需要梯度的输出
       │  outs_to_grad = [o for o, m in zip(outs, mask) if m]
       ▼
   第 3 步: autograd.grad(outs_to_grad, primals, tangents)
       │  用 proxy tensor 走 autograd
       │  记录所有反向操作到同一个 FX Graph
       ▼
   输出: gradient w.r.t. primals
       │
       ▼
   最终: Joint FX Graph
       ┌──────────────────────────────────────┐
       │  Forward 节点 (被标记 is_forward)      │
       │  Backward 节点 (被标记 is_backward)    │
       │  所有节点在同一个 FX Graph 中           │
       └──────────────────────────────────────┘

关键细节：``tangent_mask`` 的作用。

不是所有前向输出都需要梯度。例如，如果前向函数返回了中间结果用于反向复用，这些中间结果本身不需要梯度，只是被保存下来。``tangent_mask`` 区分了这两类输出：

.. code-block:: python

   def fn(x):
       sin_x = torch.sin(x)
       cos_x = torch.cos(x)
       return sin_x + cos_x, sin_x, cos_x  
       #       ^^^^^^^^^^^^^  ^^^^  ^^^^
       #       需要梯度        不需要  不需要（但反向需要）

auto_functionalize 和 proxy tensor
==========================================

这里有一个关键的技术细节：AOTAutograd 使用 ``proxy tensor`` 来追踪联合图的构建过程。Proxy tensor 是 ``torch.fx`` 中的机制——每个 proxy tensor 包装一个 ``Proxy`` 对象，所有在 proxy tensor 上的操作都会自动在 FX Graph 中创建一个新节点。

AOTAutograd 用 ``make_fx``（来自 ``torch.fx.experimental.proxy_tensor``）来执行联合追踪：

.. code-block:: python

   from torch.fx.experimental.proxy_tensor import make_fx

   # AOTAutograd 内部使用 make_fx 来捕获 autograd.grad 的完整轨迹
   joint_graph = make_fx(inner_fn)(primals, tangents)

当 ``make_fx(inner_fn)`` 被调用时，``inner_fn`` 内的所有 Tensor 操作都会被 proxy tensor 拦截，从而在 FX Graph 中创建节点。这包括：

- 前向的节点（来自 ``fn(*primals)``）
- 反向的节点（来自 ``autograd.grad``）

所有节点都位于同一个 ``fx.Graph`` 中，通过 ``node.meta["partitioner_tag"]`` 来区分属于前向还是反向：

.. code-block:: python

   # 在 create_joint 中
   for node in mode.tracer.graph.nodes:
       if _is_tangent(node):
           node.meta["partitioner_tag"] = "is_backward"
       else:
           node.meta["partitioner_tag"] = "is_forward"

AOTAutograd vs Eager Autograd
======================================

AOTAutograd 和 PyTorch 原有的 eager autograd 的核心差异在于"什么时候做微分"：

.. list-table::
   :header-rows: 1

   * -
     - Eager Autograd
     - AOTAutograd
   * - 微分时机
     - 运行时，逐操作反向
     - 编译时，提前联合追踪
   * - 图信息
     - 无全局图
     - 有完整联合图
   * - 优化空间
     - 无融合
     - 可融合/分区/重计算
   * - 内存规划
     - 无法预知
     - 可预先规划 saved tensors
   * - 实现方式
     - C++ autograd 引擎
     - Python level tracing + make_fx

这也解释了 AOTAutograd 名字中的 "AOT"（Ahead-of-Time）：它在训练开始之前（编译阶段）而不是训练过程中（运行时阶段）完成自动微分的处理。

小结
======

这一节介绍了 AOTAutograd 的联合求导机制：

- AOTAutograd 在编译时用 **proxy tensor** 追踪 autograd 的执行过程
- ``create_joint`` 生成一张**包含前向和反向的联合计算图**
- 联合图上的节点通过 ``partitioner_tag`` 标记属于前向还是反向
- 相比 eager autograd，AOTAutograd 能**全局分析**前向和反向的优化机会

下一节我们来看图分区——如何将这张巨大的联合图切分为前向子图和反向子图。
