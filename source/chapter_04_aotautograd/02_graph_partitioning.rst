.. _graph-partitioning:

=============
图分区
=============

上一节我们看到了 AOTAutograd 如何生成一张包含前向和反向的联合图。这一节我们来看如何将这张巨大的联合图分割成两个独立的子图——一个用于前向传播，一个用于反向传播。

为什么需要分区？
=====================

联合图是编译时的中间产物，不能直接用于运行时。原因有三：

1. **执行时机不同 ** ：前向传播在训练迭代的前半段执行，反向传播在后半段。它们不能"同时执行"。

2.**输入/输出不同 ** ：前向图的输入是模型的输入张量，输出是 loss + saved tensors。反向图的输入是 grad_output + saved tensors，输出是梯度。

3.**需要不同的 guard** ：前向图和反向图可能对应不同的输入形状约束。

所以，必须将联合图分割为两个独立的 ``fx.GraphModule`` ：

.. mermaid::

   graph TD
       subgraph Joint["联合图（Joint Graph）"]
           JX["%x = placeholder"]
           JSin["%sin = torch.sin(%x)"]
           JCos["%cos = torch.cos(%x)"]
           JAdd["%add = torch.add(%sin, %cos)"]
           JSum["%sum = torch.sum(%add)"]
           JGrad["%grad_output = placeholder"]
           JGradSin["%grad_sin = torch.mul(...)"]
           JGradX["%grad_x = torch.cos(%grad_sin)"]
           JRet["return (%sum, %sin, %cos), (%grad_x,)"]
           
           JX --> JSin --> JCos --> JAdd --> JSum
           JGrad --> JGradSin --> JGradX
           JSin -.->|saved| JGradSin
           JCos -.->|saved| JGradX
       end
       
       Joint ==>|partition| P{"图分区"}
       P ==> Fwd
       P ==> Bwd
       
       subgraph Fwd["前向图（Forward）"]
           FX["%x = placeholder"]
           FSin["%sin = sin(%x)"]
           FCos["%cos = cos(%x)"]
           FAdd["%add = add(...)"]
           FSum["%sum = sum(...)"]
           FRet["return (%sum, %sin, %cos)"]
           
           FX --> FSin --> FCos --> FAdd --> FSum --> FRet
       end
       
       subgraph Bwd["反向图（Backward）"]
           BG["%grad = placeholder"]
           BSin["%sin_saved = placeholder"]
           BCos["%cos_saved = placeholder"]
           BGradSin["%grad_sin = mul(...)"]
           BGradX["%grad_x = cos(...)"]
           BRet["return (%grad_x,)"]
           
           BG --> BGradSin
           BSin --> BGradSin
           BCos --> BGradX
           BGradSin --> BGradX --> BRet
       end

朴素分区策略
===============

最简单的分区策略是： **所有 ``is_forward`` 标记的节点放入前向图，所有 ``is_backward`` 标记的节点放入反向图** 。

回想上一节提到的 ``partitioner_tag`` 标记：

.. code-block:: python

   for node in mode.tracer.graph.nodes:
       if _is_tangent(node):
           node.meta["partitioner_tag"] = "is_backward"
       else:
           node.meta["partitioner_tag"] = "is_forward"

朴素分区算法大致如下：

1. 复制联合图得到两份副本
2. 在前向副本中，删除所有 ``is_backward`` 节点，然后运行死代码消除（DCE）
3. 在反向副本中，删除所有 ``is_forward`` 节点，但保留被反向引用的前向节点作为 placeholder（这些就是 "saved tensors"）

但这个朴素策略有一个问题： **有些前向节点的输出既被前向使用也被反向使用** （例如 ``sin(x)`` 的输出在反向中需要用于计算 ``cos(x) * grad_output`` ）。这些节点会在前向图中保留，但它们的输出需要被"保存"到反向图中。

默认分区器
===============

AOTAutograd 的默认分区器定义在 ``pytorch/torch/_functorch/_aot_autograd/runtime_wrappers.py`` 中，通过 ``AOTDispatchAutograd`` 类实现。它的分区逻辑可以概括为：

.. code-block:: text

   AOTDispatchAutograd.partition(joint_graph)
       │
       ├─ 1. 复制联合图
       │
       ├─ 2. 创建前向图副本
       │      - 保留所有 is_forward 节点
       │      - 删除所有 is_backward 节点
       │      - 添加额外的 saved tensor 输出
       │
       ├─ 3. 创建反向图副本
       │      - 保留所有 is_backward 节点
       │      - 将前向图中被反向引用的节点
       │        替换为 placeholder（saved tensors）
       │
       ├─ 4. 分别在两个副本上运行死代码消除
       │
       └─ 5. 返回 (fwd_module, bwd_module)

前向图的输出由两部分组成：

1. **用户定义的前向返回值 ** （如 loss）
2.**反向所需的中继值 ** （saved tensors）

类似地，反向图的输入由两部分组成：

1.**来自前向的 saved tensors**
2.** 从上游传来的梯度 **（grad_outputs）

ViewAndMutationMeta 的角色
=================================

分区结果的元信息存储在 ``ViewAndMutationMeta`` 对象中（定义在 ``pytorch/torch/_functorch/_aot_autograd/schemas.py`` 中）。它描述了：

.. code-block:: python

   @dataclass
   class ViewAndMutationMeta:
       traced_tangents: list[Any]          # 反向输入（tangents）
       traced_tangents_descs: list[AOTInput]  # tangents 的描述
       num_mutated_inp_runtime_indices: int  # 有多少输入被原地修改
       ...

这个对象在分区过程前后被传递，确保下游组件（Inductor）知道哪些输出是 saved tensors、哪些是真正的梯度。

运行时执行流程
====================

经过分区后，前向图和反向图在训练循环中的执行顺序如下：

.. code-block:: text

   训练 step:
       │
       ├─ 执行 Forward Subgraph
       │      output, saved_tensors = compiled_fwd(x, y)
       │
       ├─ loss = output
       │
       ├─ 触发 backward
       │
       ├─ 执行 Backward Subgraph
       │      grad_x, grad_y = compiled_bwd(grad_output, saved_tensors)
       │
       └─ 更新参数

在运行时， ``AOTDispatchAutograd`` （在 ``runtime_wrappers.py`` 中）封装了这个流程。它负责：

1. 调用编译后的前向函数
2. 保存前向返回的 saved tensors
3. 当 ``.backward()`` 被触发时，调用编译后的反向函数，传入 saved tensors

这个封装对用户是完全透明的——用户看到的仍然是一个普通的 PyTorch 函数调用。

小结
======

这一节介绍了图分区的基本概念：

- ** 朴素分区 **：根据 ``partitioner_tag`` 将联合图切分为前向和反向两张子图
- ** 默认分区器 **：在 ``runtime_wrappers.py`` 中，通过复制 + 裁剪 + DCE 实现
- **ViewAndMutationMeta** ：保存分区后前向/反向接口的元信息
- **运行时封装 ** ： ``AOTDispatchAutograd`` 透明地管理前向/反向的执行和 saved tensors 的传递

默认分区器对所有前向中间结果一视同仁：只要反向用到就保存。但有些中间结果可以通过**重计算** （recomputation）的方式在反向中重新算出来，而不是保存它们。下一节的 min-cut 重计算分区器会讨论这个话题。
