.. _lowering-process:

=========================
从 FX Graph 到 IRNode
=========================

在 post_grad_passes 完成图优化后，Inductor 进入 lowering 阶段——将优化后的 FX Graph 逐节点降级为 IRNode。这一过程由 ``GraphLowering`` 类（在 ``pytorch/torch/_inductor/graph.py`` 中）完成。

GraphLowering 的核心机制
==============================

``GraphLowering`` 继承自 ``torch.fx.Interpreter``：

.. code-block:: python
   :caption: pytorch/torch/_inductor/graph.py

   class GraphLowering(torch.fx.Interpreter):
       """将 FX Graph 降级为 Inductor IRNode。"""

       def __init__(self, gm, ...):
           super().__init__(gm)
           self.buffers: list[ir.Buffer] = []      # 所有 buffer
           self.operations: list[ir.Operation] = []  # 所有 IRNode（降级结果）

``torch.fx.Interpreter`` 是 FX 提供的标准图执行框架。它按拓扑序遍历 FX Graph 的每个节点，每遇到一个节点就调用对应的处理方法：

.. code-block:: text

   Interpreter.run(node)
       │
       ├─ node.op == "placeholder"  → run_placeholder(node)  → 注册为输入
       ├─ node.op == "call_function"→ call_function(target, args, kwargs)  → lowering
       ├─ node.op == "output"       → run_output(node)       → 收集输出
       └─ node.op == "get_attr"     → run_get_attr(node)     → 载入常量

``GraphLowering`` 覆写了 ``run_node`` 作为统一入口，然后根据 ``node.op`` 分派：

.. code-block:: python

   def run_node(self, n: torch.fx.Node) -> object:
       """将单个 FX Node 降级为 IRNode"""
       ...
       if n.op == "call_function":
           args, kwargs = self.fetch_args_kwargs_from_env(n)
           result = self.call_function(n.target, args, kwargs)
           # result 是一个 IRNode（如 Pointwise、Reduction、ExternKernel）
           ...
       return result

Lowering 的完整流程
=========================

以 ``aten.sin(x)`` 为例，看一个 FX 节点如何变成 IRNode：

.. code-block:: text

   FX Graph 中的节点:
       %sin = call_function[target=aten.sin](args = (%x,))

   Step 1: GraphLowering.run_node(%sin)
       │  n.op == "call_function"
       │
       ▼
   Step 2: fetch_args_kwargs_from_env(n)
       │  从环境中获取 %x 对应的 IRNode（TensorBox）
       │  args = (TensorBox(StorageBox(InputBuffer(x))),)
       │
       ▼
   Step 3: call_function(target=aten.sin, args=(TensorBox(x),))
       │
       ├─ target 在 lowerings 字典中吗？
       │   └─ 是的，aten.sin 已注册
       │
       ▼
   Step 4: lowerings[aten.sin](TensorBox(x))
       │  调用注册的 lowering 函数（来自 lowering.py）
       │
       ▼
   Step 5: 返回 IRNode
       Pointwise(
           device='cuda:0',
           dtype=torch.float32,
           inner_fn=lambda idx: ops.sin(ops.load(x, idx)),
           ranges=[1024],
       )
       │
       ▼
   Step 6: register_operation(ir_node)
       │  将 IRNode 加入 self.operations
       │  供后续 Scheduler 使用

整个 lowering 过程本质上就是一个**查表 + 调用**的过程：

.. code-block:: text

   FX call_function 节点
       │
       ├─ target 在 lowerings 表中?
       │   ├─ 是 → 调用对应的 lowering 函数
       │   │        返回 IRNode（Pointwise/Reduction/TemplateBuffer）
       │   │
       │   └─ 否 → target 有 decomposition 吗？
       │              ├─ 是 → 展开后重新追踪
       │              └─ 否 → 创建 ExternKernel 回退到 eager
       │
       ▼
   收集到 operations 列表

Fallback 机制
==================

并不是 all 算子都有 lowering 函数。当 ``GraphLowering.call_function`` 遇到一个不在 ``lowerings`` 字典中的算子时，会触发 fallback：

``call_function`` 的方法签名在 ``pytorch/torch/_inductor/graph.py`` 第 1319 行：

.. code-block:: python

   def call_function(self, target, args, kwargs):
       if target not in lowerings:
           # 检查是否可以通过 decomposition 展开
           if has_decomposition(target):
               decomps = get_decompositions([target])
               # 展开后重新追踪
               ...
           else:
               # 创建 ExternKernel 回退
               # 编译时调用 eager 实现
               return ExternKernel.create(target, *args, **kwargs)
       
       return lowerings[target](*args, **kwargs)

``ExternKernel`` 是一种特殊的 IRNode——它不生成 Triton 或 C++ 代码，而是在编译时调用 PyTorch 的 eager 实现。这意味着即使某个算子没有 lowering 函数，Inductor 也不会崩溃，而是默默回退到 eager 执行。对于没有完全覆盖的算子集合，这个机制保证了编译的健壮性。

输入和输出的处理
======================

除了 ``call_function`` 节点外，``GraphLowering`` 还处理另外两种关键节点：

**placeholder（输入）**：每个 ``placeholder`` 节点对应一个函数输入。``GraphLowering`` 将其包装为 ``InputBuffer``（是 ``ComputedBuffer`` 的子类）：

.. code-block:: python

   # run_placeholder 内部
   input_buffer = InputBuffer(name=arg_name, layout=FixedLayout(
       device=device, dtype=dtype, size=size, stride=stride,
   ))
   self.graph_inputs[arg_name] = TensorBox(input_buffer)

**output（输出）**：``output`` 节点收集所有返回值对应的 IRNode，存入 ``self.graph_outputs`` 列表，供后续的 wrapper 代码生成使用。

降级结果的形态
====================

lowering 完成后，``GraphLowering`` 中积累了以下数据：

.. code-block:: text

   GraphLowering 实例
       │
       ├─ operations: [IRNode1, IRNode2, ...]   ← Scheduler 的输入
       │    每个 entry 是一个 ir.Operation 子类
       │    （Pointwise, Reduction, ExternKernel...）
       │
       ├─ buffers: [Buffer1, Buffer2, ...]       ← 所有 buffer 的列表
       │
       ├─ graph_inputs: {name: TensorBox, ...}   ← 输入映射
       │
       └─ graph_outputs: [IRNode, ...]           ← 输出列表

``operations`` 列表是 Scheduler 的直接输入。Scheduler 遍历这个列表，分析每个 IRNode 之间的依赖关系，然后执行融合和调度。

小结
======

这一节介绍了 FX Graph 到 IRNode 的降级过程：

- **GraphLowering** 继承 ``torch.fx.Interpreter``，按拓扑序遍历 FX 节点
- **call_function** 查 ``lowerings`` 表，找不到则回退到 decomposition 或 ``ExternKernel``
- 降级结果存储在 ``operations`` 列表中，作为 Scheduler 的输入
- 整个流程是**查表 + 调用**的模式，每个 lowering 函数将 FX 操作映射为 IRNode
