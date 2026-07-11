.. _virtualization:

==============================
虚拟化（Virtualization）
==============================

.. note::

   **Virtualization 是 Inductor 中最容易被忽略的优化。 **
   它没有复杂的算法，也没有大幅的性能提升——它的作用更像"扫清障碍"：没有 virtualization，Scheduler 就无法正确识别两个操作是否可以融合，因为视图操作会引入虚假的数据依赖。有趣的是，Inductor 团队在早期版本中曾尝试跳过 virtualization，让 lowering 直接处理所有视图操作，结果发现 Scheduler 的融合率从 80% 降到了 40% 以下。这印证了一个编译器设计的常识：** 一个好的 IR 比一个好的优化更重要 **。

虚拟化（Virtualization）是 Inductor 降级过程中的第一步。它解决的是 FX Graph 中张量别名和视图操作（view/reshape/transpose/slice）导致的"存储歧义"问题。

别名为什么是问题
====================

考虑一个简单的例子：

.. code-block:: python

   def fn(x):
       y = x[:, 1:]   # y 是 x 的 view
       z = y + 1      # z 是 y + 1 的结果
       return z

在 FX Graph 中， ``x[:, 1:]`` 会被表示为一个 ``call_function[target=torch.slice]`` 节点，输出的 ``y`` 在语义上是 "x 的一个视图"——它和 ``x`` 共享存储，只是偏移和步长不同。

但在 Inductor 的 IR 层面，如果我们将 ``y`` 和 ``x`` 视为两个独立的 buffer，会丢失 "y 是 x 的一部分" 这个关键信息，导致后续无法正确融合或内存规划。

虚拟化要解决的问题就是：**将视图操作（view/reshape/transpose/slice）"虚拟化"为对 base tensor 的索引计算，而不是创建新的 buffer**。

TensorBox 和 StorageBox
==============================

Inductor 中使用 ``TensorBox`` 和 ``StorageBox`` 两层包装来实现虚拟化。

.. code-block:: text

   TensorBox  ——  对外的"张量"视图
       │
       │  持有对底层存储的引用 + 布局信息（偏移/步长）
       │
       ▼
   StorageBox ——  实际的存储（可能被多个 TensorBox 共享）
       │
       ▼
   ComputedBuffer / InputBuffer —— 实际的 IR 数据节点

.. code-block:: text

   x = InputBuffer(name="x", ...)

   y = TensorBox(                           # y 虚拟化
       StorageBox(x),                       # 共享 x 的存储
       layout=AsStridedLayout(              # 布局描述
           offset=1,                        # 从第 1 列开始
           strides=(N, 1),                  # 步长调整
           shape=(M, N-1),
       )
   )

关键设计： ``TensorBox`` 不分配新的存储，它只是在现有的 ``StorageBox`` 上增加了一个 **布局描述**。后续的 ``lowering`` 函数在访问 ``y`` 中的元素时，自动计算其在 ``x`` 中的真实索引。

虚拟化与实现（Realize）
============================

虚拟化并非在所有场景下都适用。有些操作必须创建实际的存储：

- **写入操作**：对 view 进行 in-place 修改时，必须确保存储是独立且连续的
- **跨设备/数据类型操作**：复制到不同设备或转换数据类型需要实际分配
- **代码生成阶段**：最终生成代码时，buffer 必须是"已实现"的

当虚拟化不再适用时，Inductor 会调用 ``realize()`` 方法将虚拟化的 IRNode 转换为实际的 ``ComputedBuffer`` ：

.. code-block:: python
   :caption: pytorch/torch/_inductor/ir.py（简化示意）

   class TensorBox:
       def realize(self):
           """将虚拟化的 view 转换为实际的 buffer"""
           if self.is_realized():
               return self.data
           # 创建新的 ComputedBuffer，分配实际存储
           realized = ComputedBuffer(
               name=new_name,
               layout=FixedLayout(self.get_device(), self.get_dtype(), self.get_size()),
               data=self.data,
           )
           return realized

``realize_hint()`` 方法则在 lowering 过程中触发，提示当前操作应该被实现——但并不强制立即执行，而是将"实现"推迟到最终的代码生成阶段。

虚拟化的好处
===============

虚拟化为 Inductor 带来了几个关键优势：

1.**减少内存分配**：视图操作不需要分配新的 buffer，只需记录索引变换
2.**更好的融合机会**：虚拟化的 TensorBox 可以被反向传播到其 base storage，使依赖图上原本分离的节点变得相邻，从而被 scheduler 融合
3.**简化代码生成**：代码生成器只需要为实际的 storage 生成加载/存储代码，视图访问通过索引表达式自动展开

.. code-block:: text

   无虚拟化（朴素降级）:
       %y = slice(%x)      → 分配 buffer y
       %z = add(%y, 1)     → 分配 buffer z，从 y 读取
       产生额外分配和拷贝

   有虚拟化:
       %y = TensorBox(StorageBox(%x), layout=slice_layout)
       %z = add(索引: StorageBox(%x) + offset, +1)
       不分配，直接基于 x 生成融合后的代码

虚拟化和 StorageBox
=========================

实际实现中， ``StorageBox`` 负责管理真实的数据存储。它内部持有一个 ``IRNode`` 作为实际数据，多个 ``TensorBox`` 可以通过不同的 ``Layout`` 共享同一个 ``StorageBox`` ：

.. code-block:: text

   StorageBox(data=ComputedBuffer)
       ↑           ↑           ↑
       │           │           │
   TensorBox   TensorBox   TensorBox
   (layout=A)  (layout=B)  (layout=C)

当 scheduler 最终为这些 TensorBox 生成代码时，可以通过共享的 StorageBox 推断出数据依赖，从而将原本分散的操作融合到同一个 kernel 中。

小结
======

这一节介绍了 Inductor 的虚拟化机制：

- **问题**：FX Graph 中的视图操作（view/reshape/transpose/slice）在 IR 层面会产生存储歧义
- **方案**： ``TensorBox`` + ``StorageBox`` 双层结构，TensorBox 持布局信息，StorageBox 持实际存储
- **realize（实现）** ：需要实际分配时调用 ``realize()`` ，将虚拟化节点转换为实际 buffer
- **优势** ：减少内存分配、改善融合机会、简化代码生成
