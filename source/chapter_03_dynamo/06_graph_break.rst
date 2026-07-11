.. _graph-break:

=============
Graph Break
=============

在第 1 章我们提到过：当 Dynamo 遇到无法捕获的操作时，它不会报错退出，而是优雅地"断图"——在无法继续追踪的地方切断，之前的部分形成一个子图并编译，之后的部分留给 eager 执行。

这一节我们深入 graph break 的触发条件和实现机制。

什么会触发 Graph Break？
=============================

graph break 分为两大类： **显式触发** 和 **隐式触发** 。

**显式触发** ：Dynamo 明确知道某些操作无法追踪。

- 调用 ``print()`` 、 ``assert`` 等非 Tensor 操作
- 使用 ``torch.Tensor.item()`` 将 Tensor 转为 Python 数值
- 试图将 Tensor 传给 C 扩展或 numpy 函数
- 使用 ``@torch._dynamo.disable`` 装饰器标注的函数

**隐式触发** ：符号执行过程中遇到了无法处理的异常。

- 遇到了不认识的字节码指令
- 变量类型不在 ``VariableTracker`` 的覆盖范围内
- 图规模过大（超过了配置的阈值）
- 递归调用或有其他 Dynamo 不支持的高级 Python 特性

Dynamo 维护了一张 trace rules 表（在 ``pytorch/torch/_dynamo/trace_rules.py`` 中），记录了哪些函数和模块可以追踪、哪些必须 graph break：

.. code-block:: text

   trace_rules.py
   ├── is_supported():            # 检查函数是否可追踪
   ├── is_numpy():                # numpy 操作 → graph break
   ├── is_forbidden():            # 禁止追踪的操作
   └── ...

``break_graph_if_unsupported`` 装饰器
===========================================

Dynamo 使用 ``break_graph_if_unsupported`` 装饰器来标记那些"可能触发 graph break"的字节码处理方法。这是 ``InstructionTranslatorBase`` 中的关键模式：

.. code-block:: python
   :caption: pytorch/torch/_dynamo/symbolic_convert.py（简化示意）

   from .exc import unimplemented

   def break_graph_if_unsupported(push=True):
       """装饰器：如果操作不支持，触发 graph break"""
       def decorator(handler):
           @functools.wraps(handler)
           def wrapper(self, inst):
               try:
                   handler(self, inst)           # 尝试执行 handler
               except Unsupported:
                   if push:
                       self.push(self.pop())     # 将栈状态恢复
                   self.graph_break(inst)        # 触发 graph break
           return wrapper
       return decorator

   使用方式:
   @break_graph_if_unsupported(push=True)
   def CALL_FUNCTION_EX(self, inst):
       # 处理变参函数调用的逻辑
       ...

当 ``CALL_FUNCTION_EX`` 的执行过程中抛出了 ``Unsupported`` 异常（比如参数类型无法处理），装饰器捕获这个异常，调用 ``self.graph_break()`` 中断当前的子图捕获，将控制权交还给 Dynamo 的调度器。

graph_break() 内部
=========================

``graph_break()`` 是 ``InstructionTranslatorBase`` 中的一个方法，它做的事情是：

.. code-block:: text

   graph_break(inst)
       │
       ├─ 1. 冻结当前 FX Graph
       │      完成当前子图的构建
       │
       ├─ 2. 生成 Restart 指令
       │      在字节码中插入特殊标记，
       │      标记"从这一行继续执行"
       │
       ├─ 3. 重置符号执行状态
       │      清空模拟栈和局部变量缓存，
       │      为下一个子图做准备
       │
       └─ 4. 继续执行
              从断点之后的下一条指令开始，
              重新开始一个新的子图

关键的设计是：**graph break 不是直接返回，而是继续执行** 。Dynamo 会在断点处插入一个特殊的 ``RESUME`` 标记，然后继续追踪后续的字节码，形成第二个子图。

这相当于：

.. code-block:: text

   用户函数:
       x = torch.sin(x)    ← 可追踪
       print(x)            ← 不可追踪 → graph break
       x = torch.cos(x)    ← 可追踪

   Dynamo 处理结果:
       Subgraph 1:  x → sin(x)
           [resume at: print(x)]
       Subgraph 2:  x → cos(x)

   运行时执行顺序:
       执行 Subgraph 1 → 拿到 sin(x) 的结果
       执行 print(x)（eager 模式）
       执行 Subgraph 2 → 拿到 cos(x) 的结果

两个子图的 Runtime 串联
===============================

多个子图之间的串联由 ``resume_execution.py`` 处理。Dynamo 会在生成的字节码中插入 ``TORCH_DYNAMO_RESUME_IN_PREFIX`` 前缀标记，用于标记"这是上一个 graph break 的恢复点"。

运行时的工作流程：

.. code-block:: text

   第一次进入 fn(x):
       │
       ├─ Dynamo 拦截帧
       │
       ├─ 执行 Subgraph 1（编译后的 kernel）
       │      output = compiled_subgraph1(x)
       │
       ├─ Dynamo 将 output 写回帧的局部变量
       │
       ├─ Python 解释器执行剩下的代码
       │      print(output)           ← eager 执行
       │      result = cos(output)    ← 被 Dynamo 再次拦截
       │
       └─ Dynamo 拦截帧（第二次）
              │
              ├─ 检查：这是恢复执行？
              │   是的，co_code 中有 RESUME 标记
              │
              ├─ 执行 Subgraph 2
              │
              └─ 返回最终结果

注意：每个子图都有自己的 guard。如果 ``Subgraph 1`` 的 guard 失败（比如输入形状变化），Dynamo 会重新编译 Subgraph 1，而 Subgraph 2 可能仍然命中缓存。

这就是 torch.compile 被称为"渐进式编译器"的原因——它不会"要么全编译要么全不编译"，而是尽可能编译能编译的部分。

Graph Break 的性能影响
=============================

graph break 是有代价的。每个 graph break 意味着：

1.**子图之间无法融合** ：Subgraph 1 的输出必须写到显存，Subgraph 2 再从显存读入，错失 fusion 机会
2.**额外的 kernel launch** ：原来可以 fusion 成一个 kernel 的操作被拆成多个
3.**额外的 Python 执行** ：graph break 之间的代码在 eager 模式下执行，无法享受编译加速

但 graph break 的代价不是均匀的。如果 graph break 发生在模型的前向函数 **最外层** （比如 ``torch.compile`` 嵌套了一个 ``print`` ），代价相对较小。如果 graph break 发生在 **循环内部** （比如 ``for i in range(100): print("step", i); x = torch.sin(x)`` ），会导致每轮循环都产生两个子图 + 一次 Python 解释器调用——性能损失很大。

一个经验法则是：**graph break 次数越少、位置越靠近函数边界，性能越好。** 第 8 章会介绍如何用 ``TORCH_LOGS`` 定位 graph break。

fullgraph=True 的用途
============================

``torch.compile(fn, fullgraph=True)`` 的作用就是：强制要求零 graph break。如果有 graph break 发生，直接抛出异常而不是默默生成多个子图。

.. code-block:: python

   @torch.compile(fullgraph=True)
   def fn(x):
       x = torch.sin(x)
       print(x)              # ← 这里会报错
       return torch.cos(x)

   fn(torch.randn(3))
   # RuntimeError: ... graph break ...

这在调试时非常有用——如果希望确认某个函数能否被完整编译为单个 kernel，加上 ``fullgraph=True`` ，如果能运行不报错，就说明它是"完全可编译"的。

动手验证：控制流导致的 Graph Break
============================================

下面这个例子展示了控制流（ ``if`` 语句）如何触发 graph break——当 ``x.sum() > 0`` 这个条件依赖于 Tensor 的运行时值时，Dynamo 无法在编译期确定走哪个分支，因此只能在此处断图：

.. synced-code-start::

   .. code-block:: python
      :linenos:

   import torch


   def complex_function(x):
       x = torch.sin(x)
       if x.sum() > 0:
           x = torch.cos(x)
       else:
           x = torch.tanh(x)
       return x


   compiled_fn = torch.compile(complex_function, backend="eager", fullgraph=False)
   x = torch.randn(4)
   print(compiled_fn(x))

.. synced-code-end::

小结
======

这一节我们介绍了 graph break 的触发条件和实现机制：

- **触发条件** ：非 Tensor 操作、不支持的 Python 特性、超出覆盖范围的类型
- **实现机制** ： ``break_graph_if_unsupported`` 装饰器 + ``graph_break()`` 冻结当前子图 + ``resume_execution`` 恢复执行
- **多个子图的运行时串联** ：通过嵌入 ``RESUME`` 标记的字节码实现
- **性能影响** ：graph break 阻止了跨边界融合，但 Dynamo 会尽可能追踪能追踪的部分

下一节我们来看第 3 章的最后一个话题：缓存与重新编译——一个 code object 的缓存链表是怎么维护的，以及重新编译的触发条件。
