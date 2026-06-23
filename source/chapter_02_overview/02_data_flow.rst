.. _data-flow:

=================================
数据流：图在各种表示之间的变换
=================================

上一节我们从"调用路径"的角度理解了编译流水线。这一节我们换个视角：**看数据**——看同一个计算 ``torch.sin(x) + torch.cos(y)`` 在不同的编译阶段长什么样子。

从 Python 函数到最终 Triton kernel，计算图经历了四种不同的表示形式：

.. code-block:: text

   Python 函数
       │
       ▼  Dynamo 字节码捕获
   ┌─────────────────────┐
   │  FX Graph           │  节点 = call_function(target=torch.sin, ...)
   │  (torch.fx.Graph)   │  边 = 节点之间的 args 引用
   └─────────┬───────────┘
             │
             ▼  AOTAutograd 自动微分
   ┌─────────────────────┐
   │  Joint Graph        │  前向节点 + 反向节点 在同一张图内
   │  (还是一个 FX Graph) │  grad_output → ... → grad_input
   └─────────┬───────────┘
             │
             ▼  图分区
   ┌─────────────────────┐
   │  Inductor IRNode    │  Pointwise<sin> → Reduction<add>
   │  (torch._inductor    │  循环级 IR，保留维度信息
   │   .ir)              │
   └─────────┬───────────┘
             │
             ▼  Scheduler 融合
   ┌─────────────────────┐
   │  SchedulerNode      │  被融合为一组：{sin, cos, add}
   │  (FusedSchedulerNode)│  单 kernel launch
   └─────────┬───────────┘
             │
             ▼  Codegen
   ┌─────────────────────┐
   │  Triton/C++ 代码    │  tl.load → tl.sin → tl.store
   └─────────────────────┘

下面我们用一个具体的例子来追踪这个变换过程。

示例：追踪一个计算
====================

假设我们有这样一个函数：

.. code-block:: python

   def fn(x, y):
       return torch.sin(x) + torch.cos(y)

我们用 ``TORCH_LOGS`` 来观察每个阶段的输出。

阶段 1：Dynamo 输出 FX Graph
---------------------------------

当 Dynamo 捕获 ``fn`` 时，它输出一张 FX Graph。我们可以在代码中拿到它：

.. code-block:: python

   import torch

   def fn(x, y):
       return torch.sin(x) + torch.cos(y)

   # 使用 Dynamo 的捕获能力，获取 FX Graph
   from torch._dynamo.testing import CompileCounter

   counter = CompileCounter()
   compiled_fn = torch.compile(fn, backend=counter)
   compiled_fn(torch.randn(3), torch.randn(3))

   # counter 中保存了捕获到的 FX Graph
   fx_graph = counter.graphs[0]
   print(fx_graph)

输出：

.. code-block:: text

   graph():
       %x : [num_users=2] = placeholder[target=x]
       %y : [num_users=1] = placeholder[target=y]
       %sin : [num_users=1] = call_function[target=torch.sin](args = (%x,), kwargs = {})
       %cos : [num_users=1] = call_function[target=torch.cos](args = (%y,), kwargs = {})
       %add : [num_users=1] = call_function[target=torch.add](args = (%sin, %cos), kwargs = {})
       return add

这是最高层的表示。每个节点直接对应一个 PyTorch API 调用。关于 FX Graph 的节点结构我们在 2.3 节已经详细讲过，这里不再重复。

关键观察：这张图里有 **3 个计算节点** （sin、cos、add），理论上可以 fusion 成一个 kernel。但在 FX Graph 层面，我们只看到了"做什么"，没有看到"怎么做"。

阶段 2：AOTAutograd 生成 Joint Graph
-----------------------------------------------

如果 ``fn`` 参与了梯度计算（比如它是模型 forward 的一部分），AOTAutograd 会在 FX Graph 的基础上，通过自动微分生成一张包含前向和反向的联合图。

对 ``fn`` 应用 AOTAutograd 后（假设 loss 是 ``output.sum()``），joint graph 大致长这样：

.. code-block:: text

   # Joint Graph（简化）
   #
   # ┌─ Forward ──────────────────────────┐
   # │  %x    : placeholder               │
   # │  %y    : placeholder               │
   # │  %sin  = torch.sin(%x)              │
   # │  %cos  = torch.cos(%y)              │
   # │  %add  = torch.add(%sin, %cos)      │
   # │  %sum  = torch.sum(%add)            │
   # │  return %sum, %sin, %cos            │
   # └────────────────────────────────────┘
   #
   # ┌─ Backward ─────────────────────────┐
   # │  %grad_output : placeholder         │
   # │  %grad_cos   = torch.mul(...)       │  ← cos 的梯度
   # │  %grad_sin   = torch.mul(...)       │  ← sin 的梯度
   # │  %grad_x     = torch.cos(%grad_sin)  │
   # │  %grad_y     = torch.neg(%grad_cos)  │
   # │  return %grad_x, %grad_y            │
   # └────────────────────────────────────┘

注意：joint graph 中，前向不仅返回了最终结果 ``%add``，还返回了 ``%sin`` 和 ``%cos``——这些是反向计算梯度的过程中需要复用的中间结果。

Joint graph 中还有一类特殊节点：**tangent** —— 反向传播的起始梯度。对于 ``output.sum()`` 来说，grad_output 就是形状为 ``()`` 的标量 1.0。

阶段 3：图分区（Forward / Backward）
------------------------------------------

Joint graph 太大了——它包含前向和反向的所有操作。但编译器最终需要生成两个独立的 kernel 集：一个用于前向，一个用于反向。

``partitioners.py`` 中的 ``min_cut_rematerialization_partition`` 将 joint graph 切分为两个子图：

.. code-block:: text

                    Joint Graph
           ┌────────────────────────┐
           │  Forward: x, y → sin,  │
           │       cos, add, sum    │
           │  Backward: grad →      │
           │       grad_x, grad_y   │
           └───────────┬────────────┘
                       │  partition
                       ▼
   ┌──────────────────┐  ┌──────────────────┐
   │ Forward Subgraph  │  │ Backward Subgraph │
   │ x → sin → add     │  │ grad → grad_x     │
   │ y → cos ↗         │  │       → grad_y    │
   │                    │  │                   │
   │ 保存 sin, cos     │  │ 复用 sin, cos     │
   └──────────────────┘  └──────────────────┘

前向子图额外输出 ``sin`` 和 ``cos`` 作为" saved tensors "，反向子图将它们视为输入。

阶段 4：Inductor 降级为 IRNode
------------------------------------------

现在 Inductor 拿到了一个前向子图（一个 FX Graph）。它需要把这个"Python 函数调用图"降级为更低级的、面向循环的中间表示——**IRNode**。

降级过程由 ``torch/_inductor/lowering.py`` 完成。对于 ``torch.sin(x)``，lowering 会找到对应的降级函数，生成一个 ``Pointwise`` 节点：

.. code-block:: text

   # 降级前的 FX Graph
   %sin = call_function[target=torch.sin](args = (%x,))

   # 降级后的 IRNode
   Pointwise(
       device='cuda:0',
       dtype=torch.float32,
       inner_fn=lambda index: tl.sin(load(x, index)),
       ranges=[1024],  # 假设 x 是 1024 个元素
   )

``Pointwise`` 节点描述的是：对 ``[0, 1024)`` 范围内的每个索引，计算 ``tl.sin(load(x, index))``。这已经非常接近最终生成的 Triton 代码了。

类似地，``torch.add(sin, cos)`` 会被降级为另一个 ``Pointwise``，它的 ``inner_fn`` 是：对每个索引，加载 sin 在 index 处的值、加载 cos 在 index 处的值，然后相加。

降级后的 IRNode 有三种主要类型：

.. list-table::
   :header-rows: 1

   * - IR 类型
     - 语义
     - 示例
   * - ``Pointwise``
     - 逐元素操作，输出形状 = 输入形状
     - sin, cos, add, relu, mul
   * - ``Reduction``
     - 归约操作，输出形状 < 输入形状
     - sum, mean, max
   * - ``TemplateBuffer``
     - 预定义的 kernel 模板（如 GEMM）
     - mm, bmm, conv

阶段 5：Scheduler 融合
--------------------------------

IRNode 层面的每个节点都是一个独立的 kernel。对于小张量的逐元素操作，每个 kernel 都单独 launch 是非常低效的。Scheduler 负责把这些节点融合成更大的组。

Scheduler 遍历 IRNode 之间的数据依赖，将具有相同 ``ranges`` 和 ``device`` 的 ``Pointwise`` 节点合并为一个：

.. code-block:: text

   融合前 (3 个独立 kernel):
       [kernel 1] Pointwise: sin(x)
       [kernel 2] Pointwise: cos(y)
       [kernel 3] Pointwise: add(sin, cos)

   融合后 (1 个 fusion kernel):
       [kernel 1] FusedSchedulerNode:
           sin(x) → load x, compute sin
           cos(y) → load y, compute cos
           add    → add sin and cos, store to output

融合后的 kernel 只需要一次 global memory load（加载 x 和 y）和一次 global memory store（写回结果），中间结果 ``sin`` 和 ``cos`` 完全在寄存器中传递，不需要写回显存。

阶段 6：代码生成
---------------------------

最后，代码生成器接收融合后的 ``SchedulerNode``，生成实际的 GPU 或 CPU 代码。

对于 GPU，``torch/_inductor/codegen/triton.py`` 中的 ``TritonScheduling`` 会生成以下 Triton 代码：

.. code-block:: python

   @triton.jit
   def fused_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(axis=0)
       block_start = pid * BLOCK_SIZE
       offsets = block_start + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements

       # 加载输入
       x = tl.load(x_ptr + offsets, mask=mask)
       y = tl.load(y_ptr + offsets, mask=mask)

       # 融合的计算：sin(x) + cos(y)
       sin_x = tl.sin(x)
       cos_y = tl.cos(y)
       result = sin_x + cos_y

       # 写回
       tl.store(output_ptr + offsets, result, mask=mask)

注意：``sin_x`` 和 ``cos_y`` 是 Triton 寄存器变量，不是 Tensor。它们从未被写回显存，只在寄存器中传递给下一行代码。

一个实际的例子
====================

我们用上一节的前向子图来完整追踪数据流的变换。假设输入是 ``torch.randn(4)``：

.. code-block:: python

   import torch

   @torch.compile(fullgraph=True)
   def fn(x, y):
       return torch.sin(x) + torch.cos(y)

   # 验证编译无 graph break
   out = fn(torch.randn(4), torch.randn(4))

开启完整日志，可以看到每个阶段的具体信息：

.. code-block:: bash

   TORCH_LOGS="+dynamo,+inductor" python -c "
   import torch
   @torch.compile(fullgraph=True)
   def fn(x, y):
       return torch.sin(x) + torch.cos(y)
   out = fn(torch.randn(4), torch.randn(4))
   "

输出中的关键信息（整理后）：

.. code-block:: text

   [dynamo] 捕获 FX Graph: 4 个节点 (2 placeholders, 2 call_function, 1 output)
   [inductor] 降级: 3 个 IRNode → 1 个 FusedSchedulerNode
   [inductor] 生成 Triton 代码:
       fused_kernel(  # 1 个 kernel launch
           x_ptr, y_ptr, output_ptr,
           n_elements=4, BLOCK_SIZE=4
       )

从最初的 3 个 PyTorch API 调用（sin、cos、add）→ 3 个 IRNode → 1 个 fusion kernel → 1 次 GPU kernel launch。这就是 torch.compile 做的事情。

组件间接口契约
======================

从数据流的角度来看，三个组件之间有明确的接口契约。

Dynamo → AOTAutograd / Backend 的契约
------------------------------------------

Dynamo 输出：

.. code-block:: text

   1. torch.fx.GraphModule     — 捕获到的计算图
   2. list[torch.Tensor]       — example_inputs（FakeTensor）
   3. Guards                   — 缓存有效性检查条件

AOTAutograd → Inductor 的契约
------------------------------------

AOTAutograd 输出：

.. code-block:: text

   1. list[torch.fx.GraphModule] — 分区后的子图列表
      每个子图对应一个 compiled function
   2. 元信息: 输入/输出规格,
      哪些输入需要梯度,
      哪些输出是 saved tensors

Inductor → Runtime 的契约
-------------------------------

Inductor 输出：

.. code-block:: text

   1. CompiledFxGraph:
      - callable — 可调用的函数
      - 生成的 Triton/C++ 源码
      - kernel 缓存 key
      - 性能指标

这种清晰的接口契约使得三组件可以独立演进。理论上，你可以用 Dynamo + 自定义后端（跳过 AOTAutograd），或者用 Dynamo + AOTAutograd + 自定义后端（替换 Inductor），或者用 FX Graph + 自定义工具链（完全跳过 torch.compile）。

小结
======

这一节我们追踪了同一个计算在不同编译阶段的表示形式：

.. list-table::
   :header-rows: 1

   * - 阶段
     - 表示形式
     - 关键特点
   * - Dynamo 捕获
     - FX Graph
     - Python API 级别的调用图
   * - AOTAutograd
     - Joint Graph
     - 前向 + 反向在同一张图
   * - 图分区
     - Forward / Backward Subgraph
     - 前向反向分离，saved tensors
   * - Inductor 降级
     - IRNode (Pointwise/Reduction)
     - 循环级 IR，保留维度信息
   * - Scheduler 融合
     - FusedSchedulerNode
     - 多节点合并为单 kernel
   * - Codegen
     - Triton / C++ 代码
     - 可编译执行的 GPU/CPU 代码

这就是 torch.compile 的**数据流全景**。理解了这个数据流，后面第 3～6 章深入每个组件时，你就知道每个组件在整条流水线中的位置和职责了。
