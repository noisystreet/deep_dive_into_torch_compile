.. _tangent-and-epilogue:

======================
Tangent 与 Epilogue
======================

这一节我们来看 AOTAutograd 中两个与反向传播边界相关的概念：**tangent** （反向传播的起始梯度）和 **epilogue** （前向/反向的收尾操作）。

Tangent：反向传播的起始梯度
===================================

在联合图的创建过程中，``create_joint`` 除了接收前向输入 ``primals`` 之外，还接收另一组输入：**tangents**。

.. code-block:: python

   def inner_fn(primals, tangents):
       outs = fn(*primals)
       grad_outs = torch.autograd.grad(outs_to_grad, primals, 
                                        grad_outputs=tangents)
       return outs, grad_outs

tangent 的概念在联合图中扮演了两个角色：

1. **联合图中的占位符**：``tangent`` 节点在联合图中作为反向子图的输入占位符出现。它们标记了反向传播从哪里开始。

2. **反向子图的第一个输入**：在分区之后，反向子图将 tangents 作为第一个输入（紧随 saved tensors 之后）。

在一个典型的训练场景中，如果 loss 是 ``output.sum()``，tangent 就是一个形状为 ``()`` 的标量张量，值为 1.0。

tangent_mask：哪些输出需要 tangent？
========================================

不是所有前向输出都需要 tangent。区分哪些前向输出参与反向传播是必要的：

.. code-block:: python

   def fn(x, w):
       y = w @ x          # 输出 1：需要梯度（参与 loss 计算）
       y_saved = y.detach()  # 输出 2：不需要梯度（反向不需要）
       return y, y_saved

``create_joint`` 在追踪前向时维护了一个 ``tangent_mask``：

.. code-block:: python

   outs, tangent_mask = fn(*primals)
   # tangent_mask = [True, False]
   #                  ↑ y 需要梯度，y_saved 不需要

只有 ``tangent_mask == True`` 的输出才会参与 ``autograd.grad`` 调用。在联合图中，这些输出的梯度被计算，而其他输出被忽略。

在 ``schemas.py`` 的 ``ViewAndMutationMeta`` 中，``traced_tangents`` 字段保存了这些信息：

.. code-block:: python

   @dataclass
   class ViewAndMutationMeta:
       traced_tangents: list[Any]          # 哪些前向输出需要梯度
       traced_tangents_descs: list[AOTInput]  # tangents 的描述

在运行时，tangent 的维度信息被序列化为 ``traced_tangent_metas`` 并在运行时反序列化，用于构建实际的 tangent 输入。

Epilogue：收尾操作
=========================

Epilogue 指的是**在完整训练迭代中、反向传播完成后需要执行的额外操作**。AOTAutograd 的 epilogue 机制处理三类场景：

1. 将梯度更新写回参数
2. 应用梯度裁剪
3. 处理 optimizer step

但在 AOTAutograd 的上下文中，"epilogue" 更具体地指**前向图末尾的一些额外操作**。

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

``AOTDispatchAutograd``（在 ``runtime_wrappers.py`` 中）是实现 epilogue 的关键类。它在运行时管理前向和反向的执行：

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

.. code-block:: text

   用户训练函数
       │
       ▼
   1. 功能化 (functionalization)
      将 in-place 操作转换为纯函数式
      输出: fn_func
       │
       ▼
   2. 准备 autograd (fn_prepped_for_autograd)
      准备前向输出的 tangent_mask
      输出: fn_prepped
       │
       ▼
   3. 创建联合图 (create_joint)
      用 autograd.grad 追踪前向和反向
      输出: joint_fn
       │
       ▼
   4. 追踪联合图 (make_fx)
      用 proxy tensor 生成 joint FX Graph
      输出: fx_g (joint graph)
       │
       ▼
   5. 图分区 (min_cut_rematerialization_partition)
      分割成前向子图和反向子图
      输出: (fwd_module, bwd_module)
       │
       ▼
   6. 编译 (Inductor compile_fx)
      分别编译前向和反向
      输出: (compiled_fwd, compiled_bwd)
       │
       ▼
   7. 运行时包装 (AOTDispatchAutograd)
      管理前向/反向的执行流和 saved tensors
      输出: AOTDispatchAutograd.forward()

小结
======

这一节介绍了 tangent 和 epilogue 的概念：

- **Tangent**：反向传播的起始梯度，是联合图中反向子图的输入占位符
- **tangent_mask**：区分哪些前向输出需要参与反向传播
- **前向 epilogue**：前向中需要额外返回的、用于反向的值（如修改后的输入）
- **运行时 epilogue**：``AOTDispatchAutograd`` 管理前向/反向的执行流和 saved tensors 传递

至此，第 4 章的内容全部完成。我们从联合求导开始，走过了图分区、min-cut 重计算、functionalization、以及 tangent/epilogue，覆盖了 AOTAutograd 的完整工作流程。

下一章我们将进入 Inductor 后端——看看 FX Graph 如何被降级为循环级 IR，然后通过 Scheduler 融合，最终生成高效的 Triton 或 C++ 代码。
