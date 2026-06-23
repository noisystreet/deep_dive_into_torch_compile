.. _functionalization:

=================
Functionalization
=================

在前几节的讨论中，我们一直假设联合图里的操作都是**纯函数式**的——没有 in-place 修改，没有 alias。但在现实 PyTorch 代码中，in-place 操作（如 ``x.add_(y)``、``x.t_()``）和 view 操作（如 ``x.view()``、``x.transpose()``）无处不在。

这让 AOTAutograd 面临一个问题：**如果前向中有 in-place 操作，autograd 的 joint trace 无法正确处理它**。

这一节我们来看 AOTAutograd 如何通过 **Functionalization** （功能化）来解决这个问题。

in-place 操作为什么是问题
==============================

考虑一个简单的例子：

.. code-block:: python

   def fn(x):
       x.add_(1)  # in-place 修改 x
       return x * 2

在 eager 模式下，``x.add_(1)`` 直接修改了 ``x`` 的存储。autograd 知道这个操作是因为 ``add_`` 的 ``backward`` 函数被正确注册了——它输出 ``grad_input = grad_output``。

但在 AOTAutograd 的联合追踪中，问题出现了：

.. code-block:: text

   用 make_fx 追踪 fn:
       x.add_(1) 在 FX Graph 中变成了 torch.add(x, 1)
       → 但这不准确！add_ 修改了 x，而 add 创建了新 tensor

   反向中的梯度计算就偏差了

根本上说，**FX Graph 是一个纯函数式 IR**——每一个节点都创造新的输出，不修改输入。而 ``torch.add_`` 是一个有副作用的操作。两者不兼容。

Functionalization 的解决方案
======================================

``FunctionalTensor`` 是 PyTorch 中解决这个问题的机制。它的实现在 ``pytorch/torch/_subclasses/functional_tensor.py`` 中，而 AOTAutograd 的功能化入口在 ``pytorch/torch/_functorch/_aot_autograd/functional_utils.py``。

核心思路是：**将用户函数中的所有 in-place 操作和 view 操作，在追踪之前转换为纯函数式操作**。

.. code-block:: text

   用户的函数:
       def fn(x):
           x.add_(1)          ← in-place 修改
           return x * 2

   功能化后的函数:
       def fn_functionalized(x):
           x = x.clone()      ← 先拷贝
           x = torch.add(x, 1)  ← 改为 out-of-place
           return x * 2

这样，FX Graph 中就只有纯函数式节点了。in-place 操作的"副作用"被表达为函数返回值的传递。

to_fun 和 from_fun
============================

功能化的核心 API 是两个函数：``to_fun`` 和 ``from_fun``，定义在 ``functional_utils.py`` 中。

.. code-block:: python
   :caption: pytorch/torch/_functorch/_aot_autograd/functional_utils.py

   def to_fun(t):
       """将普通 Tensor 包装为 FunctionalTensor"""
       if isinstance(t, Tensor):
           return FunctionalTensor.to_functional(t)
       return t

   def from_fun(t):
       """从 FunctionalTensor 中提取真实 Tensor"""
       if isinstance(t, FunctionalTensor):
           return t.elem  # 提取内部的真实张量
       return t

FunctionalTensor 是一个包装类（继承自 ``torch.Tensor``），它拦截所有 in-place 操作，将它们转换为 out-of-place 操作加上"版本号更新"：

.. code-block:: text

   x = FunctionalTensor(tensor)

   x.add_(1)  ← 被拦截
       │
       ▼
   FunctionalTensor 内部:
       1. 读取 x 的当前值
       2. 执行 out-of-place 的 torch.add(x, 1)
       3. 将 x 的内部存储替换为计算结果
       4. 递增版本号

从外部看，``x.add_(1)`` 的效果和 eager 模式一致——``x`` 的值被更新了。但从图捕获的角度看，实际记录的 FX 操作是 ``torch.add(x, 1)``（out-of-place），而不是 ``torch.add_(x, 1)``。

FunctionalTensor 与 FunctionalTensorMode
===============================================

``FunctionalTensor`` 是通过 PyTorch 的 dispatch 模式机制工作的。``FunctionalTensorMode``（在 ``pytorch/torch/_subclasses/functional_tensor.py`` 中）是一个 ``TorchDispatchMode``，它在每次 Tensor 操作时被激活：

.. code-block:: python

   class FunctionalTensorMode(TorchDispatchMode):
       def __torch_dispatch__(self, func, types, args, kwargs):
           # 拦截所有操作
           # 如果是 in-place 操作（如 add_），转换为 out-of-place
           # 并触发"同步"机制

在 ``create_functionalized_fn`` 中（在 ``graph_capture_wrappers.py`` 中），AOTAutograd 将用户函数包装为功能化版本：

.. code-block:: text

   create_functionalized_fn(flat_fn, ...)
       │
       ├─ 1. 用 to_fun 将所有输入 Tensor 包装为 FunctionalTensor
       │
       ├─ 2. 在 FunctionalTensorMode 上下文中执行用户函数
       │     所有 in-place 操作被自动转换
       │
       ├─ 3. 用 from_fun 将结果提取为普通 Tensor
       │
       └─ 4. 返回功能化后的函数

View 操作的处理
====================

View 操作（如 ``x.view()``、``x.transpose()``、``x[:, 1:]``）与 in-place 操作同样需要功能化。

考虑一个例子：

.. code-block:: python

   def fn(x):
       y = x[:, 1:]  # y 是 x 的 view
       y.add_(1)     # 通过 view 修改了 x！
       return x

在 eager 模式下，``y.add_(1)`` 会**同时修改 y 和 x**（因为两者共享存储）。功能化必须正确处理这种"通过 view 修改 base tensor"的情况。

Functionalization 的处理方式是：

.. code-block:: text

   功能化后:
       y = x[:, 1:]        ← view 操作保留（纯函数式）
       y_updated = y + 1   ← out-of-place add
       x_updated = x.copy_() 将 y_updated 的值同步回 x
       return x_updated

这个"同步"操作通过 ``sync_functional_tensor`` 实现（在 ``functional_utils.py`` 中）：

.. code-block:: python

   def sync_functional_tensor(t):
       """将 FunctionalTensor 上的修改同步到底层存储"""
       if isinstance(t, FunctionalTensor):
           t.sync()  # 触发版本比较和同步

在功能化过程中，当检测到一个被修改的 view 需要将其值同步回 base tensor 时，AOTAutograd 会插入一个 ``torch.ops.aten.copy_.default`` 操作。这个操作在 FX Graph 中是一个显式的节点，表示"将 view 的值写回 base"。

.. code-block:: text

   功能化后的 FX Graph:
       %x    = placeholder
       %y    = slice(x, dim=1, start=1)    ← y = x[:, 1:]
       %y_1  = add(y, 1)                   ← y + 1
       %x_1  = copy_(x, y_1)               ← 将 y 的修改同步回 x
       return x_1

注意：``copy_`` 即使本身是 in-place 操作，在 FunctionalTensor 内部也会被转换为 out-of-place——它输出一个新的 tensor，而不是修改输入。

功能化后 AOTAutograd 的流程
======================================

加上 functionalization 后，AOTAutograd 的完整流程变为：

.. code-block:: text

   用户函数 fn
       │
       ├─ 1. to_fun: 所有输入 Tensor → FunctionalTensor
       │
       ├─ 2. create_functionalized_fn
       │      在 FunctionalTensorMode 中执行 fn
       │      所有 in-place 操作被自动替换为 out-of-place
       │      view 操作通过 sync_functional_tensor 同步
       │
       ├─ 3. 功能化后的函数 fn_func
       │      不包含任何 in-place 操作
       │      可以被 make_fx 安全追踪
       │
       ├─ 4. create_joint(fn_func)
       │      生成联合图（全是纯函数式操作）
       │
       └─ 5. 图分区 → 编译

功能化之后，联合图中的操作都是纯函数式的，可以被编译器安全地融合、重排、重计算。

小结
======

这一节介绍了 functionalization 机制：

- **问题**：in-place 操作和 FX Graph 的纯函数式假设不兼容
- **方案**：``FunctionalTensor`` 通过 ``TorchDispatchMode`` 拦截 in-place 操作，自动转换为 out-of-place
- **API**：``to_fun``/``from_fun`` 包装/解包 FunctionalTensor；``sync_functional_tensor`` 处理 view 的同步
- **效果**：功能化后的函数不包含任何 in-place 操作，可以被 ``make_fx`` 安全追踪

下一节我们来看 AOTAutograd 的最后一个专题：tangent 和 epilogue——反向传播的起始梯度和收尾操作。
