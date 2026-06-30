.. _symbolic-shapes:

================================
符号形状（Symbolic Shapes）
================================

第 3.5 节的 guard 机制默认把 ``x.shape[0] == 32`` 这样的**具体数值**编进 guard 里。输入形状一变，guard 失败，触发重新编译。对于 batch size 固定、序列长度恒定的训练任务，这完全合理；但对于变长 batch、LLM 推理中的不同 prompt 长度，频繁重编译会让编译时间压过执行收益。

``dynamic=True`` 和符号形状机制，就是 Dynamo 用来**用符号代替具体数值**、一次编译适配多种形状的方案。这一节从 ``ShapeEnv`` 和 ``SymNode`` 出发，解释符号形状在编译栈中如何工作。

从具体形状到符号形状
==========================

静态编译时，Dynamo 看到的输入形状是具体的：

.. code-block:: text

   输入 x.shape = (32, 784)
   guard 表达式: x.shape[0] == 32 AND x.shape[1] == 784

开启动态编译（``dynamic=True`` 或 ``mark_dynamic``）后，Dynamo 不再绑定具体数值，而是创建 **符号变量**：

.. code-block:: text

   输入 x.shape = (s0, 784)     ← s0 是符号，代表"某个未知的第 0 维大小"
   guard 表达式: s0 >= 1        ← 只约束值域，不绑定具体数值

这样 ``(32, 784)``、``(64, 784)``、``(128, 784)`` 都能命中同一份编译结果，只要 ``s0`` 满足已记录的约束。

ShapeEnv 与 SymNode
======================

符号形状的管理集中在 ``torch/fx/experimental/symbolic_shapes.py`` 中的 ``ShapeEnv`` 类。每次 Dynamo 开始编译一个 frame，都会创建或复用一个 ``ShapeEnv`` 实例，负责：

- 创建符号变量（``SymNode``）
- 记录符号之间的约束（如 ``s0 == s1``、``s0 >= 1``）
- 在 guard 失败时给出约束冲突的原因
- 与 FakeTensor 协作，让符号执行阶段就能"看到"符号化的 shape

``SymNode`` 是单个符号的载体。它包装了一个 SymPy 表达式，并携带 dtype、值域等元数据：

.. code-block:: python
   :caption: SymNode 的简化示意

   class SymNode:
       def __init__(self, expr, hint=None):
           self._expr = expr          # SymPy 表达式，如 Symbol('s0')
           self._hint = hint        # 编译时的具体 hint 值（如 32），用于 codegen 优化

       @property
       def shape(self):
           return self  # SymInt 包装

当你访问 ``x.shape[0]`` 且 ``x`` 是 FakeTensor 时，如果第 0 维被标记为动态，返回的是 ``SymInt(SymNode('s0'))`` 而不是整数 ``32``。

符号形状在编译流水线中的流转
==================================

.. code-block:: text

   Dynamo 符号执行
       │
       ├─ ShapeEnv 创建 s0, s1, ...
       ├─ FakeTensor.shape → SymInt
       ├─ 算子 trace 时 shape 运算也符号化（s0 + 1, s0 * 2 等）
       │
       ▼
   FX Graph（带 SymInt 的 meta）
       │
       ├─ AOTAutograd joint trace（shape 约束传递）
       │
       ▼
   Inductor Lowering
       │
       ├─ 静态 shape → 常量折叠、循环展开
       ├─ 符号 shape → 生成泛化 kernel（runtime 读取实际 shape）
       │
       ▼
   Guard 检查（运行时）
       │
       └─ 验证 s0 是否仍满足编译时记录的约束

关键点是：**符号形状信息从 Dynamo 一直传递到 Inductor**。Inductor 在 lowering 时看到 ``s0`` 而不是 ``32``，会生成读取 runtime shape 的 kernel，而不是把 ``32`` 硬编码进 Triton 代码。

如何启用符号形状
====================

**全局开启** （最简单）：

.. code-block:: python

   @torch.compile(dynamic=True)
   def fn(x):
       return x * 2

**按维度标记** （更精细）：

.. code-block:: python

   import torch._dynamo as dynamo

   x = torch.randn(100, 784)
   dynamo.mark_dynamic(x, 0)       # 仅第 0 维动态
   # dynamo.mark_static(x, 1)     # 显式标记第 1 维静态

   @torch.compile
   def fn(x):
       return x @ weight

   fn(x)

**Export 路径声明** （编译期固定约束）：

.. code-block:: python

   batch = torch.export.Dim("batch", min=1, max=512)
   torch.export.export(model, (x,), dynamic_shapes={"x": {0: batch}})

三种方式底层都走 ``ShapeEnv``，但约束的**声明时机**不同：``mark_dynamic`` 在运行时、``export Dim`` 在 trace 时、``dynamic=True`` 则默认将所有维度视为可能变化。

符号 guard vs 数值 guard
==============================

Guard 树（第 3.5 节）在符号形状模式下会生成不同类型的检查：

.. list-table::
   :header-rows: 1

   * - Guard 类型
     - 静态模式示例
     - 符号模式示例
   * - Shape guard
     - ``x.shape[0] == 32``
     - ``x.shape[0] >= 1`` （值域约束）
   * - 符号关系
     - 无
     - ``x.shape[0] == y.shape[0]`` （两输入 batch 对齐）
   * - 数据依赖
     - 可能 graph break
     - 可能 graph break 或符号化失败

符号 guard 的粒度比数值 guard 粗——这正是它减少重编译的原因。代价是生成的 kernel 无法利用 ``shape[0] == 32`` 这样的常量信息做激进优化（如完全展开循环）。

性能权衡
============

.. code-block:: text

   静态形状编译:
       编译次数: 每种形状一次
       kernel 质量: 高（常量折叠、特化 tiling）
       适用: 固定 batch 训练

   符号形状编译:
       编译次数: 少（一种泛化 kernel）
       kernel 质量: 中（runtime 读 shape，优化保守）
       适用: 变长输入推理、多 batch size 服务

PyTorch 还在探索 **延迟特化** （第 5.10 节）：先用泛化 kernel 运行，后台为常见形状编译特化版本，后续自动切换。这是静态与动态之间的折中。

与 graph break 的交互
==========================

并非所有 Python 代码都能被符号化。以下情况仍可能导致 graph break 或编译失败：

- **数据依赖形状**：``x[x.sum() > 0]``——输出 shape 取决于数据值，符号执行无法确定
- **Python 整数强制转换**：``int(x.shape[0])`` 用于 Python 控制流
- **不支持的 SymPy 运算**：某些 shape 算术无法表达为符号约束

遇到这些情况，Dynamo 要么 graph break（第 3.6 节），要么回退到静态 guard。调试时可启用：

.. code-block:: bash

   TORCH_LOGS="+dynamic,+guards" python train.py

关于 Dynamic Shapes 的日志解读和重编译诊断，见第 8.5 节。

小结
======

- **符号形状** 用 ``SymNode`` / ``SymInt`` 代替具体维度数值，让一份编译结果适配多种输入尺寸
- ``ShapeEnv`` （``symbolic_shapes.py``）是符号变量的创建、约束求解与 guard 生成的核心
- **启用方式**：``dynamic=True``、``mark_dynamic``、``torch.export.Dim`` 三种路径，底层共享同一基础设施
- **权衡**：减少重编译 vs kernel 特化程度；静态训练优先关闭 dynamic，变长推理优先开启
- **调试**：``TORCH_LOGS=+dynamic`` 追踪符号决策；详见第 8.5 节

至此，第 3 章的内容全部完成。我们从字节码基础开始，走过了字节码分析、图捕获、guard 机制、graph break、缓存与重新编译、符号形状，覆盖了 TorchDynamo 的完整工作流程。

下一章我们将进入 AOTAutograd——看看 Dynamo 捕获的 FX Graph 如何被扩展为包含前向和反向的联合计算图。
