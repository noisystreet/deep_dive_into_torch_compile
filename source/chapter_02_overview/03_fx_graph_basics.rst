.. _fx-graph-basics:

========================
FX Graph 基础
========================

第 1 章我们已经看到，torch.compile 的三大组件（Dynamo、AOTAutograd、Inductor）都在操作一种叫做"计算图"的东西。在深入每个组件之前，我们需要先搞清楚这个"图"到底是什么。

从 trace 开始
=================

打开 Python，运行下面这段代码：

.. synced-code-start::

   .. code-block:: python
      :linenos:

      import torch


      class MyModel(torch.nn.Module):
          def forward(self, x):
              return torch.sin(x) + torch.cos(x)


      model = MyModel()
      fx_model = torch.fx.symbolic_trace(model)
      print(fx_model.graph)
      fx_model.graph.print_tabular()

.. synced-code-end::

输出会是这样：

.. code-block:: text

   graph():
       %x : [num_users=2] = placeholder[target=x]
       %sin : [num_users=1] = call_function[target=torch.sin](args = (%x,), kwargs = {})
       %cos : [num_users=1] = call_function[target=torch.cos](args = (%x,), kwargs = {})
       %add : [num_users=1] = call_function[target=torch.add](args = (%sin, %cos), kwargs = {})
       return add

以及表格形式：

.. code-block:: text

   opcode         name    target      args         kwargs
   ─────────────  ──────  ──────────  ───────────  ───────
   placeholder    x       x           ()           {}
   call_function  sin     torch.sin   (x,)         {}
   call_function  cos     torch.cos   (x,)         {}
   call_function  add     torch.add   (sin, cos)   {}
   output         output  output      (add,)       {}

这就是一张 **FX Graph**——PyTorch 的计算图中间表示。

图的结构
============

FX Graph 是一个 **有向无环图（DAG）** ，由两种元素构成：

节点（Node）
----------------

图中每一行就是一个节点。每个节点有四个核心属性：

.. list-table::
   :header-rows: 1

   * - 属性
     - 含义
     - 例子
   * - ``op``
     - 操作类型
     - ``placeholder``, ``call_function``, ``output``
   * - ``target``
     - 具体操作
     - ``torch.sin``, ``torch.add``
   * - ``args``
     - 位置参数
     - ``(x,)``, ``(sin, cos)``
   * - ``kwargs``
     - 关键字参数
     - ``{}``

五种操作类型：

.. list-table::
   :header-rows: 1

   * - ``op``
     - 含义
   * - ``placeholder``
     - 函数输入参数
   * - ``get_attr``
     - 获取模块参数 / buffer
   * - ``call_function``
     - 调用普通函数
   * - ``call_module``
     - 调用子模块
   * - ``call_method``
     - 调用张量方法（如 ``x.view()`` ）
   * - ``output``
     - 返回值

边（Edge）
----------------

一个节点在 ``args`` 中引用另一个节点，就形成了一条边。在上面的例子中， ``sin`` 节点引用 ``x`` ， ``add`` 节点引用 ``sin`` 和 ``cos``——这就构成了 ``x → sin → add`` 的数据流路径。

.. code-block:: text

   x ──→ sin ──┐
    │           ├──→ add ──→ return
    └──→ cos ──┘

从图的角度看， **"编译"本质上就是对这个图进行变换** ：

1.**优化** ：合并 ``sin`` 和 ``cos`` 的 kernel launch、消除中间变量
2.**降级** ：把 ``torch.sin`` 映射到底层的 CUDA 或 Triton 实现
3.**分区** ：将一张大图切成几块，分别编译，中间的边界就是 graph break

为什么要用图？
=================

你可能会问：为什么不能直接在 Python 函数上做优化？非要搞个图出来？

原因有两点：

**第一，图是语言无关的中间表示** 。一张 FX Graph 里的 ``torch.sin`` 节点，可以翻译成 Triton 代码（GPU）、C++ 代码（CPU），或者甚至对接给 XLA 后端。**一次捕获，多种后端** ，这是编译器的核心抽象。

**第二，图让你能看到整体的计算模式** 。Python 的视角是局部的——它只看到当前这一行代码。图的视角是全局的——它能看到哪些操作可以融合、哪些中间结果可以复用、哪些计算可以移除。比如上面 ``sin(x) + cos(x)`` ，编译器一看就知道是两个逐元素操作可以融合成一个 kernel。

图的变换：一个简单的例子
================================

FX Graph 不仅可以看，还可以直接修改。下面这段代码在图中插入一个 ``relu`` ：

.. code-block:: python

   import torch.fx as fx

   class MyModel(torch.nn.Module):
       def forward(self, x):
           return torch.sin(x) + torch.cos(x)

   model = MyModel()
   traced = fx.symbolic_trace(model)

   # 在 sin 后面插入一个 relu
   for node in traced.graph.nodes:
       if node.target == torch.sin:
           with traced.graph.inserting_after(node):
               new_node = traced.graph.call_function(
                   torch.nn.functional.relu, args=(node,))
               # 把后续引用 sin 的地方替换为 relu(sin)
               node.replace_all_uses_with(new_node)
               # 但 add 的第二个参数还是原来的 sin，恢复它
               new_node.args = (node,)
           break

   traced.recompile()
   print(traced.graph)

输出：

.. code-block:: text

   graph():
       %x : [num_users=1] = placeholder[target=x]
       %sin : [num_users=1] = call_function[target=torch.sin](args = (%x,), kwargs = {})
       %relu : [num_users=1] = call_function[target=torch.nn.functional.relu](args = (%sin,), kwargs = {})
       %cos : [num_users=1] = call_function[target=torch.cos](args = (%x,), kwargs = {})
       %add : [num_users=1] = call_function[target=torch.add](args = (%relu, %cos), kwargs = {})
       return add

Graph 变成了 ``x → sin → relu ─┐`` 加 ``x → cos ───┐ → add`` 。

当然，在实际的 torch.compile 内部，图的变换比这复杂得多。但原理是一样的：**读图 → 分析 → 变换 → 重新编译** 。

FX Graph 与 torch.compile
======================================

在 torch.compile 的编译流水线中，FX Graph 是贯穿三大组件的核心数据结构：

- **Dynamo** 的输出是一张 FX Graph
- **AOTAutograd** 消费这张图，输出分区后的 FX Graph
- **Inductor** 消费分区后的 FX Graph，输出 IRNode（更低级的中间表示）

理解 FX Graph 是理解 torch.compile 的起点。下一节我们来看编译缓存的架构——这是 torch.compile 运行时性能的关键支撑。然后第 3 章我们会深入 Dynamo，看看它是如何通过字节码分析一步一步构建出这张 FX Graph 的。
