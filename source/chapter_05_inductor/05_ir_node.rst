.. _ir-node:

=======
IRNode
=======

IRNode 是 Inductor 内部的核心中间表示，位于 FX Graph（计算图级）和最终代码（循环级）之间。理解 IRNode 是理解 Inductor 如何"思考"的关键。

IRNode 层次结构
==================

Inductor 中的 IR 节点按照功能和抽象层级形成一个分层结构：

.. code-block:: text

   IRNode（抽象基类）
   │
   ├── Operation（可调度到 GPU/CPU 的操作）
   │   ├── ExternKernel（外部 kernel 调用）
   │   ├── TemplateBuffer（预设 kernel 模板）
   │   └── ComputedBuffer（计算产生的 buffer）
   │       └── InputBuffer（输入 buffer）
   │
   ├── Loops（基于循环的 IR）
   │   ├── Pointwise——逐元素操作
   │   ├── Reduction——归约操作
   │   └── Scan——扫描操作
   │
   ├── TensorBox（虚拟化包装）
   │   └── StorageBox（实际存储）
   │
   └── Layout（布局描述）
       ├── FixedLayout（固定形状和步长）
       ├── AsStridedLayout（视图：偏移 + 步长）
       └── ...

从 ``ir.py`` 的类定义来看（以 ``Pointwise`` 为例）：

.. code-block:: python
   :caption: pytorch/torch/_inductor/ir.py

   class Pointwise(Loops):
       """
       逐元素操作 IR。
       inner_fn 是一个接受索引范围、返回 OpsValue 的可调用对象。
       """
       device: torch.device
       dtype: torch.dtype
       inner_fn: Callable[[Sequence[Expr]], OpsValue]
       ranges: Sequence[sympy.Expr]

当一个 ``torch.sin(x)`` 被 lowering 时，它变成：

.. code-block:: text

   Pointwise(
       device='cuda:0',
       dtype=torch.float32,
       inner_fn=lambda index: ops.sin(ops.load(x.buffer, index)),
       ranges=[1024],   # 假设 x 有 1024 个元素
   )

``inner_fn`` 是这个 IR 节点的核心——它是一个 Python 函数，描述了"如何在范围和索引下计算这个操作的值"。代码生成器只需要调用 ``inner_fn`` 就能得到当前索引下的值。

三种主要的 IR 类型
========================

**Pointwise**：描述逐元素操作。输入和输出的形状相同。``inner_fn`` 对每个索引独立产生一个值。

.. code-block:: text

   # torch.sin(x) 的 IR
   Pointwise(
       ranges=[M, N],
       inner_fn=lambda idx: ops.sin(ops.load(x, idx)),
       # 对 (i, j) 范围内的每个元素调用 sin
   )

**Reduction**：描述归约操作。输出形状小于输入形状。除了 ``inner_fn`` 外，还包含归约类型（sum、max、min 等）。

.. code-block:: text

   # x.sum(dim=1) 的 IR
   Reduction(
       ranges=[M, N],        # 输入范围
       reduction_ranges=[N],  # 归约的维度
       reduction_type="sum",  # 归约类型
       inner_fn=lambda idx: ops.load(x, idx),
       # 沿着 reduction_dim 求和
   )

**TemplateBuffer**：描述预定义的 kernel 模板，常用于矩阵乘法和卷积。这些操作有高度优化的实现（如 cuBLAS、Triton GEMM），不适合用 pointwise 或 reduction 逐元素生成代码。

.. code-block:: text

   # torch.mm(x, y) 的 IR
   TemplateBuffer(
       name="mm",
       layout=FixedLayout(device='cuda:0', size=[M, K, N]),
       inputs=[x, y],
       kernel_template="atlas_gemm",
   )

Lowering 映射表
======================

每一个 FX 操作（如 ``torch.sin``、``torch.add``、``torch.mm``）都有对应的 lowering 函数，通过 ``register_lowering`` 装饰器注册到 ``lowerings`` 字典中。

``register_lowering`` 定义在 ``pytorch/torch/_inductor/lowering.py``：

.. code-block:: python
   :caption: pytorch/torch/_inductor/lowering.py（简化示意）

   lowerings: dict[torch._ops.OpOverload, Callable] = {}

   def register_lowering(aten_fn, broadcast=False, type_promotion_kind=...):
       """装饰器：将 FX 操作注册到 lowering 映射表"""
       def decorator(fn):
           lowerings[aten_fn] = fn
           return fn
       return decorator

   # 注册示例
   @register_lowering(aten.sin, type_promotion_kind=None)
   def sin_lower(x):
       # x 是 TensorBox
       # 返回 Pointwise IRNode
       return Pointwise(
           device=x.get_device(),
           dtype=x.get_dtype(),
           inner_fn=lambda idx: ops.sin(x.get_dtype())(ops.load(x, idx)),
           ranges=x.get_size(),
       )

   @register_lowering(aten.add, broadcast=True, type_promotion_kind=...)
   def add_lower(x, y):
       # 处理 broadcast，生成 Pointwise
       ...

   @register_lowering(aten.mm)
   def mm_lower(x, y):
       # 生成 TemplateBuffer（使用 GEMM 模板）
       ...

``broadcast`` 参数告诉 lowering 框架自动处理广播语义——当 ``x`` 和 ``y`` 形状不同时，自动插入广播逻辑。``type_promotion_kind`` 参数控制类型提升规则。

ops 原语
=============

在 ``inner_fn`` 内部，计算通过一组原语操作（``ops.*``）来描述。这些原语是 IR 层面到具体硬件指令的桥梁：

.. list-table::
   :header-rows: 1

   * - 原语
     - 作用
   * - ``ops.load(name, index)``
     - 从 buffer 加载值
   * - ``ops.store(name, index, value)``
     - 将值存入 buffer
   * - ``ops.sin(value)``
     - 计算正弦
   * - ``ops.add(a, b)``
     - 加法
   * - ``ops.constant(val, dtype)``
     - 创建常量
   * - ``ops.index_expr(expr, dtype)``
     - 索引表达式
   * - ``ops.reduction(reduction_type, name, index)``
     - 归约操作

在代码生成阶段，这些 ``ops.*`` 调用被翻译为具体的 Triton 或 C++ 代码。

IR 的设计思想与同类对比见 :ref:`ir-design-philosophy`。

小结
======

这一节介绍了 Inductor 的核心中间表示 IRNode：

- **三种主要 IR 类型** ：``Pointwise`` （逐元素）、``Reduction`` （归约）、``TemplateBuffer`` （预定义模板）
- **Lowering 映射表**：通过 ``register_lowering`` 装饰器将 FX 操作注册到 IR 构造函数
- **ops 原语**：IR 层面的计算描述，在 codegen 时翻译为具体硬件代码
- **inner_fn** ：IR 节点的核心——描述「如何在索引下计算值」
- **设计思想**：内存中心、Lazy Fusion、与 FX/XLA/MLIR 等的定位对比见 :ref:`ir-design-philosophy`
