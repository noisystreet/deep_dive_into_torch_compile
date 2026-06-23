.. _hello-world:

=============
Hello World
=============

现在让我们跑起来第一个 ``torch.compile`` 示例。环境搭建参见附录 D。

运行示例
=============

新建一个文件 ``hello_compile.py``，输入以下代码：

.. literalinclude:: ../examples/hello_compile.py
   :language: python
   :linenos:

运行它：

.. code-block:: bash

   python hello_compile.py

输出是一个 3x3 的张量：

.. code-block:: text

   tensor([[ 0.6785,  0.1234, -0.4567],
           [ 0.7890, -0.2345,  0.5678],
           [-0.8901,  0.3456,  0.9012]])

值是什么不重要。重要的是：**这段代码的执行路径和普通的 PyTorch 代码完全不同**。

第一次调用：编译
=====================

当你调用 ``compiled_foo(x, y)`` 时，发生的事情是这样的：

.. code-block:: text

   compiled_foo(x, y)
       │
       ▼
   TorchDynamo 拦截调用
       │
       ├─ 执行 foo(x, y) 并同时追踪字节码
       │
       ├─ 捕获到 FX Graph:
       │      x ──→ sin ──┐
       │      y ──→ cos ──┼──→ add ──→ return
       │
       ├─ 调用 backend="eager"
       │   （只是运行 FX Graph，不优化）
       │
       └─ 缓存编译结果 + 生成 guard
       │
       ▼
   ─────────────────────────────────────
   │   返回结果                           │
   │   下次调用相同的 x/y 形状时直接使用缓存 │
   ─────────────────────────────────────

关键细节：

1. **Dynamo 在第一次调用时执行并追踪**。它同时做两件事：正常运行 ``foo`` 的计算，以及通过 PEP 523 接口观察 Python 字节码的执行流，逐条捕获涉及 PyTorch 操作的调用。

2. **backend="eager"** 意味着不进行真正的后端优化，只是将 FX Graph 中的 ``call_function`` 节点逐个映射回 eager 执行。这是调试用的后端。默认的 ``inductor`` 后端则会生成 Triton kernel。

3. **编译结果被缓存**。相同的输入形状下一次调用时直接命中缓存，跳过编译。

Guard 机制
===============

等一等——Dynamo 怎么知道"相同的输入形状"？

它在编译时生成了一个 **guard**：一个检查输入张量的 ``.shape``、``.dtype``、``.device`` 是否与编译时一致的断言。如果 guard 检查通过，直接执行缓存；如果失败，重新编译。

.. code-block:: text

   第二次调用 compiled_foo(z, w):
       │
       ├─ 检查 guard: z.shape == (3,3)?  w.shape == (3,3)?
       │
       ├─ yes → 直接执行缓存的编译结果（无编译开销）
       │
       └─ no  → 重新编译 + 生成新的 guard

我们用实际的代码来验证：

.. code-block:: python

   import torch

   def foo(x, y):
       return torch.sin(x) + torch.cos(y)

   compiled_foo = torch.compile(foo, backend="eager", fullgraph=True)

   # 第一次调用：编译
   result1 = compiled_foo(torch.randn(3, 3), torch.randn(3, 3))

   # 第二次调用：形状相同，命中缓存
   result2 = compiled_foo(torch.randn(3, 3), torch.randn(3, 3))

   # 第三次调用：形状不同，触发重新编译
   result3 = compiled_foo(torch.randn(5, 5), torch.randn(5, 5))

如果开启 ``TORCH_LOGS="+guards"``，可以看到 guard 的生成和检查日志。

.. code-block:: bash

   TORCH_LOGS="+guards" python hello_compile.py

关于 guard 的完整机制，我们会在第 3 章深入。

验证编译确实发生了
========================

你可以通过 ``TORCH_LOGS`` 环境变量观察编译过程：

.. code-block:: bash

   TORCH_LOGS="+dynamo" python hello_compile.py

你会看到类似这样的输出（节选）：

.. code-block:: text

   [__graph_breaks]  没有 graph break，捕获完整图
   [__guards]        生成 guard: ... equal check on (3, 3)
   [__compile]       Compiling function foo...

这意味着 Dynamo 成功地捕获了完整的计算图，没有发生 graph break。

第一次调用为什么慢
=========================

你可能注意到了：第一次调用 ``compiled_foo`` 比普通 PyTorch 调用慢得多。这是正常的，因为它在"编译"而不是"执行"。后面我们会用基准测试来量化：

- 第一次调用：编译开销（几十到几百毫秒，取决于图大小）
- 后续调用：执行开销（与 eager 相当或更快）

如果图足够大（比如整个 ResNet-50），编译开销分摊到多次前向传播后，净收益是显著的。但如果函数只调用一次，编译开销就是净损失。这就是 torch.compile 适合**重复执行的场景**（训练循环、推理服务）的原因。

小结
======

这一节我们跑通了第一个 ``torch.compile`` 示例，看到了它的执行流程：

1. 第一次调用：**编译** （追踪 → 捕获图 → 生成代码 → 缓存 + guard）
2. 后续调用：**检查 guard** （命中缓存 → 直接执行；未命中 → 重新编译）

我们还验证了 guard 机制的存在，并解释了为什么第一次调用慢。

下一节我们来看 ``torch.compile`` 的常见用法和配置选项。
