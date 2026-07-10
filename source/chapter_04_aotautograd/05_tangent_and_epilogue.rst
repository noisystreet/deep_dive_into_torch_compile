.. _tangent-and-epilogue:

======================
Tangent 与 Epilogue
======================

这一节我们来看 AOTAutograd 中两个与反向传播边界相关的概念： **tangent** （反向传播的起始梯度）和 **epilogue** （前向/反向的收尾操作）。

Tangent：反向传播的起始梯度
===================================

在联合图的创建过程中， ``create_joint`` 除了接收前向输入 ``primals`` 之外，还接收另一组输入： **tangents** 。

.. code-block:: python

   def inner_fn(primals, tangents):
       outs = fn(*primals)
       grad_outs = torch.autograd.grad(outs_to_grad, primals, 
                                        grad_outputs=tangents)
       return outs, grad_outs

tangent 的概念在联合图中扮演了两个角色：

1. **联合图中的占位符 ** ： ``tangent`` 节点在联合图中作为反向子图的输入占位符出现。它们标记了反向传播从哪里开始。

2.**反向子图的第一个输入** ：在分区之后，反向子图将 tangents 作为第一个输入（紧随 saved tensors 之后）。

在一个典型的训练场景中，如果 loss 是 ``output.sum()`` ，tangent 就是一个形状为 ``()`` 的标量张量，值为 1.0。

tangent_mask：哪些输出需要 tangent？
========================================

不是所有前向输出都需要 tangent。区分哪些前向输出参与反向传播是必要的：

.. code-block:: python

   def fn(x, w):
       y = w @ x          # 输出 1：需要梯度（参与 loss 计算）
       y_saved = y.detach()  # 输出 2：不需要梯度（反向不需要）
       return y, y_saved

``create_joint`` 在追踪前向时维护了一个 ``tangent_mask`` ：

.. code-block:: python

   outs, tangent_mask = fn(*primals)
   # tangent_mask = [True, False]
   #                  ↑ y 需要梯度，y_saved 不需要

只有 ``tangent_mask == True`` 的输出才会参与 ``autograd.grad`` 调用。在联合图中，这些输出的梯度被计算，而其他输出被忽略。

在 ``schemas.py`` 的 ``ViewAndMutationMeta`` 中， ``traced_tangents`` 字段保存了这些信息：

.. code-block:: python

   @dataclass
   class ViewAndMutationMeta:
       traced_tangents: list[Any]          # 哪些前向输出需要梯度
       traced_tangents_descs: list[AOTInput]  # tangents 的描述

在运行时，tangent 的维度信息被序列化为 ``traced_tangent_metas`` 并在运行时反序列化，用于构建实际的 tangent 输入。

Epilogue：收尾操作
=========================

Epilogue 指的是 **在完整训练迭代中、反向传播完成后需要执行的额外操作 ** 。AOTAutograd 的 epilogue 机制处理三类场景：

1. 将梯度更新写回参数
2. 应用梯度裁剪
3. 处理 optimizer step

但在 AOTAutograd 的上下文中，"epilogue" 更具体地指**前向图末尾的一些额外操作** 。

前向 epilogue
-------------------

考虑一个场景：前向函数中包含了 in-place 修改输入的语义（通过 functionalization 转换为输出）。

.. code-block:: text

   功能化前:
       def fn(x):
           x.add_(1)      ← 修改输入
           return x * 2

   功能化后:
       def fn(x):
           x = x.add(1)   ← out-of-place
           return x * 2, x  ← 额外输出修改后的 x

   分区后前向图:
       %x   = placeholder
       %x_1 = add(x, 1)
       %out = mul(x_1, 2)
       return %out, %x_1  ← 额外返回修改后的 x

这个额外返回的 ``x_1`` 就是 epilogue 的一部分——它在前向图中被产生，在反向结束后被用于"将修改写回原始输入"。

运行时包装器负责在反向完成后执行这些 epilogue 操作。

AOTDispatchAutograd 中的 epilogue 处理
=============================================

``AOTDispatchAutograd`` （在 ``runtime_wrappers.py`` 中）是实现 epilogue 的关键类。它在运行时管理前向和反向的执行：

.. code-block:: python
   :caption: pytorch/torch/_functorch/_aot_autograd/runtime_wrappers.py（简化示意）

   class AOTDispatchAutograd:
       def forward(self, *args):
           # 1. 调用编译后的前向函数
           fwd_result = self.compiled_fwd(*args)
           
           # 2. 从结果中分离前向输出和 saved tensors
           fwd_outputs = fwd_result[:self.num_fwd_outputs]
           saved_tensors = fwd_result[self.num_fwd_outputs:]
           
           # 3. 返回前向结果（用户只看到这部分）
           #    内部保存 saved_tensors 用于反向
           return fwd_outputs

       def backward(self, *grad_outputs):
           # 1. 组合 grad_outputs + saved_tensors
           bwd_inputs = grad_outputs + saved_tensors
           
           # 2. 调用编译后的反向函数
           grads = self.compiled_bwd(*bwd_inputs)
           
           # 3. 执行 epilogue（如将梯度写回参数）
           self.apply_epilogue(grads)
           
           return grads

整个流程中，saved tensors 是前向图和反向图之间的桥梁，而 tangents 是用户代码和反向图之间的桥梁。

完整的 AOTAutograd 工作流
======================================

把所有概念串起来，AOTAutograd 处理一个训练函数的完整流程如下：

.. mermaid::

   sequenceDiagram
       participant User as 用户训练函数
       participant Func as 功能化
       participant Prep as 准备 autograd
       participant Joint as 创建联合图
       participant FX as make_fx 追踪
       participant Partition as 图分区
       participant Inductor as Inductor 编译
       participant Runtime as 运行时包装

       User->>Func: 原始函数 fn
       Note over Func: 将 in-place 操作<br/>转换为纯函数式
       Func->>Prep: fn_func（功能化后的函数）
       Note over Prep: 准备前向输出的<br/>tangent_mask
       Prep->>Joint: fn_prepped
       Note over Joint: 用 autograd.grad<br/>追踪前向和反向
       Joint->>FX: joint_fn
       Note over FX: 用 proxy tensor<br/>生成 joint FX Graph
       FX->>Partition: fx_g（joint graph）
       Note over Partition: min-cut 分区<br/>分割成前向/反向子图
       Partition->>Inductor: fwd_module + bwd_module
       Note over Inductor: 分别编译前向和反向
       Inductor->>Runtime: compiled_fwd + compiled_bwd
       Note over Runtime: AOTDispatchAutograd<br/>管理执行流和 saved tensors

AOTAutograd 与其他编译栈组件的交互边界
============================================

AOTAutograd 在 PyTorch 编译栈中扮演着"中间层"的角色，它接收 Dynamo 的输出，处理后交给 Inductor。这个交互边界清晰定义了各组件的职责范围：

.. mermaid::

   flowchart LR
       subgraph dynamo["Dynamo"]
           fxgraph["FX Graph（前向）"]
       end

       subgraph aot["AOTAutograd"]
           joint_graph["功能化 + 分解 + 联合求导"]
           partition["图分区<br/>min-cut / 朴素"]
           fwd["前向子图"]
           bwd["反向子图"]
           runtime["运行时包装器<br/>AOTDispatchAutograd"]
       end

       subgraph inductor["Inductor"]
           lowering["Lowering + 融合 + 代码生成"]
       end

       fxgraph -->|"输入"| joint_graph
       joint_graph --> partition
       partition --> fwd
       partition --> bwd
       fwd -->|"编译"| lowering
       bwd -->|"编译"| lowering
       lowering -->|"compiled_fwd"| runtime
       lowering -->|"compiled_bwd"| runtime

三个接口边界的具体含义：

.. list-table::
   :header-rows: 1

   * - 边界
     - 输入
     - 输出
     - 关键职责
   * - Dynamo → AOTAutograd
     - FX Graph（仅前向）
     - Joint FX Graph（前向+反向）
     - 功能化、分解、联合求导追踪
   * - AOTAutograd → Inductor
     - Joint FX Graph
     - 前向子图 + 反向子图（已分解）
     - 图分区、节点标记、saved tensors 规划
   * - AOTAutograd → Runtime
     - 编译后的前向/反向函数
     - 训练时的前向/反向执行
     - saved tensors 传递、epilogue 执行

.. note::

   **AOTAutograd 输出的图已经是"干净的"基本算子图。 ** 经过功能化和分解后，输出给 Inductor 的子图中不再包含 in-place 操作和高层算子（如 ``layer_norm`` 、 ``softmax`` ）。Inductor 可以直接对这些基本算子进行 lowering，不需要再处理功能化或分解的逻辑。

小结
======

这一节介绍了 tangent 和 epilogue 的概念：

- **Tangent** ：反向传播的起始梯度，是联合图中反向子图的输入占位符
- **tangent_mask** ：区分哪些前向输出需要参与反向传播
- **前向 epilogue** ：前向中需要额外返回的、用于反向的值（如修改后的输入）
- **运行时 epilogue** ： ``AOTDispatchAutograd`` 管理前向/反向的执行流和 saved tensors 传递

至此，第 4 章的内容全部完成。我们从联合求导开始，走过了图分区、min-cut 重计算、functionalization、以及 tangent/epilogue，覆盖了 AOTAutograd 的完整工作流程。

下一章我们将进入 Inductor 后端——看看 FX Graph 如何被降级为循环级 IR，然后通过 Scheduler 融合，最终生成高效的 Triton 或 C++ 代码。
