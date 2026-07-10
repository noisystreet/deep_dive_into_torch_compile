.. _lowering-process:

=========================
从 FX Graph 到 IRNode
=========================

在 post_grad_passes 完成图优化后，Inductor 进入 lowering 阶段——将优化后的 FX Graph 逐节点降级为 IRNode。这一过程由 ``GraphLowering`` 类（在 ``pytorch/torch/_inductor/graph.py`` 中）完成。

GraphLowering 的核心机制
==============================

``GraphLowering`` 继承自 ``torch.fx.Interpreter`` ：

.. code-block:: python
   :caption: pytorch/torch/_inductor/graph.py

   class GraphLowering(torch.fx.Interpreter):
       """将 FX Graph 降级为 Inductor IRNode。"""

       def __init__(self, gm, ...):
           super().__init__(gm)
           self.buffers: list[ir.Buffer] = []      # 所有 buffer
           self.operations: list[ir.Operation] = []  # 所有 IRNode（降级结果）

``torch.fx.Interpreter`` 是 FX 提供的标准图执行框架。它按拓扑序遍历 FX Graph 的每个节点，每遇到一个节点就调用对应的处理方法：

.. code-block:: text

   Interpreter.run(node)
       │
       ├─ node.op == "placeholder"  → run_placeholder(node)  → 注册为输入
       ├─ node.op == "call_function"→ call_function(target, args, kwargs)  → lowering
       ├─ node.op == "output"       → run_output(node)       → 收集输出
       └─ node.op == "get_attr"     → run_get_attr(node)     → 载入常量

``GraphLowering`` 覆写了 ``run_node`` 作为统一入口，然后根据 ``node.op`` 分派：

.. code-block:: python

   def run_node(self, n: torch.fx.Node) -> object:
       """将单个 FX Node 降级为 IRNode"""
       ...
       if n.op == "call_function":
           args, kwargs = self.fetch_args_kwargs_from_env(n)
           result = self.call_function(n.target, args, kwargs)
           # result 是一个 IRNode（如 Pointwise、Reduction、ExternKernel）
           ...
       return result

从 FX 节点到 IRNode：完整流程
=====================================

以 ``aten.sin(x)`` 为例，追踪一个 FX 节点如何变成 IRNode：

.. code-block:: text

   FX Graph 中的节点:
       %sin = call_function[target=aten.sin](args = (%x,))

   Step 1: GraphLowering.run_node(%sin)
       │  n.op == "call_function"
       │
       ▼
   Step 2: fetch_args_kwargs_from_env(n)
       │  从环境中获取 %x 对应的 IRNode（TensorBox）
       │  args = (TensorBox(StorageBox(InputBuffer(x))),)
       │
       ▼
   Step 3: call_function(target=aten.sin, args=(TensorBox(x),))
       │
       ├─ target 在 lowerings 字典中吗？
       │   └─ 是的，aten.sin 已注册
       │
       ▼
   Step 4: lowerings[aten.sin](TensorBox(x))
       │  调用注册的 lowering 函数（来自 lowering.py）
       │
       ▼
   Step 5: 返回 IRNode
       Pointwise(
           device='cuda:0',
           dtype=torch.float32,
           inner_fn=lambda idx: ops.sin(ops.load(x, idx)),
           ranges=[1024],
       )
       │
       ▼
   Step 6: register_operation(ir_node)
       │  将 IRNode 加入 self.operations
       │  供后续 Scheduler 使用

call_function 的完整调度逻辑
-----------------------------------

``GraphLowering.call_function`` 是 lower 的中央枢纽。它的实际调度逻辑比"查表"更复杂：

.. code-block:: python
   :caption: pytorch/torch/_inductor/graph.py（简化）

   def call_function(self, target, args, kwargs):
       # 1. 解包 TensorBox：从 args/kwargs 中提取真实 IRNode
       args = self.unpack_tensorbox_args(args)
       kwargs = self.unpack_tensorbox_kwargs(kwargs)

       # 2. 在 lowerings 表中查找 target
       if target in lowerings:
           # 最常见的路径：直接调用注册的 lowering 函数
           result = lowerings[target](*args, **kwargs)

       elif needs_alignment_check(target):
           # 某些操作需要对输入做对齐检查（如 torch.nn.functional.scaled_dot_product_attention）
           result = self.fallback_with_alignment(target, args, kwargs)

       elif has_decomposition_in_aot(target):
           # 已由 AOTAutograd 分解过的算子，直接回退到 ExternKernel
           result = ExternKernel.create(target, *args,**kwargs)

       else:
           # 尝试通过 Inductor 自身的 decomposition 展开
           if target in inductor_decompositions:
               result = self.decompose_and_retrace(target, args, kwargs)
           else:
               result = ExternKernel.create(target, *args, **kwargs)
       
       # 3. 将结果包装为 TensorBox（如果还没被包装）
       if not isinstance(result, TensorBox):
           result = TensorBox(result)

       # 4. 将结果注册到 operations 列表
       self.register_operation(result)
       return result

这里的关键动作是 **TensorBox 解包**——在调用 lowering 函数之前，将参数中的 TensorBox 层层剥开，获取底层真实的 IRNode（如 StorageBox、InputBuffer）。低频操作才会进入 ExternKernel 路径。

整个 lowering 过程本质上就是一个 ** 查表 + 调用**的过程：

.. code-block:: text

   FX call_function 节点
       │
       ├─ target 在 lowerings 表中?
       │   ├─ 是 → 调用对应的 lowering 函数
       │   │        返回 IRNode（Pointwise/Reduction/TemplateBuffer）
       │   │
       │   └─ 否 → target 有 decomposition 吗？
       │              ├─ 是 → 展开后重新追踪
       │              └─ 否 → 创建 ExternKernel 回退到 eager
       │
       ▼
   收集到 operations 列表

lowerings 表的内部结构
-------------------------------

``lowerings`` 实际上是一个字典的字典：

.. code-block:: python
   :caption: pytorch/torch/_inductor/lowering.py

   # 主映射表：ATen 操作 → lowering 函数
   lowerings: dict[torch._ops.OpOverload, Callable] = {}

   # 补充映射表：用于处理变体
   # 同一个 ATen 操作可能有多个重载（如 aten.add.Tensor、aten.add.Scalar）
   # lowerings 表中为每个重载分别注册了对应的 lowering 函数
   
   # 另一个重要映射：用于将高频操作直接映射到 Triton 语义
   # 绕过 ATen 层面的间接调用
   fallbacks: dict[torch._ops.OpOverload, Callable] = {}

这个字典通过 ``register_lowering`` 装饰器增量构建。每次 PyTorch 导入时，``lowering.py`` 中所有用该装饰器修饰的函数都会被注册。对于一个典型模型（如 ResNet-50），lowering 阶段只会命中这个字典中的几十个条目，但字典的总条目数通常在 600+（覆盖了 Inductor 支持的所有 ATen 算子）。

四种 lowering 模式
------------------------

不同的 ATen 操作映射为不同类型的 IRNode，每种类型对应一种 lowering 模式：

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 模式
     - 生成的 IRNode
     - 示例操作
   * - **逐元素模式**
     - ``Pointwise``
     - ``aten.sin``、``aten.add``、``aten.mul``、``aten.relu``
   * - **归约模式**
     - ``Reduction``
     - ``aten.sum``、``aten.mean``、``aten.max``、``aten.argmax``
   * - **模板模式**
     - ``TemplateBuffer``
     - ``aten.mm``、``aten.convolution``、``aten.bmm``
   * - **外部模式 **
     - ``ExternKernel``
     - ``aten.sort``、``aten.unique``、自定义 C++ 扩展

** 逐元素模式** 最直接——对每个输入索引独立计算一个输出值。lowering 函数构造一个 ``Pointwise``，其 ``inner_fn`` 通过 ``ops.*`` 原语描述计算逻辑。

**归约模式 ** 需要额外指定归约的维度和类型。``Reduction`` IRNode 包含 ``reduction_ranges`` 和 ``reduction_type`` 两个关键字段。

**模板模式 ** 用于矩阵乘法和卷积等高密度计算。这些操作有高度优化的手写模板（Triton GEMM 或 cuBLAS），编译器不会逐元素生成代码。

**外部模式 ** 是 fallback——对于 Inductor 无法 lowering 的操作，直接调用 PyTorch 的 eager 实现。

具体例子：四种模式逐一分析
---------------------------------------

**例 1：Pointwise 模式——aten.sin**

.. code-block:: python

   @register_lowering(aten.sin, type_promotion_kind=None)
   def sin_lower(x):
       # x 是 TensorBox 类型
       # 返回 Pointwise IRNode
       return Pointwise(
           device=x.get_device(),
           dtype=x.get_dtype(),
           inner_fn=lambda idx: ops.sin(ops.load(x, idx)),
           ranges=x.get_size(),
       )

``inner_fn`` 是一个闭包——它捕获了 ``x`` 对应的 IRNode，当被 codegen 调用时，``ops.load(x, idx)`` 生成加载代码，``ops.sin(...)`` 生成计算代码。整个内联函数的"编译"被推迟到了 codegen 阶段。

** 例 2：Reduction 模式——aten.sum**

.. code-block:: python

   @register_lowering(aten.sum)
   def sum_lower(x, dim=None, keepdim=False):
       ...
       return Reduction(
           device=x.get_device(),
           dtype=x.get_dtype(),
           inner_fn=lambda idx: ops.load(x, idx),
           ranges=x.get_size(),           # 输入范围
           reduction_ranges=[x.get_size()[dim]],  # 归约维度
           reduction_type="sum",
       )

``reduction_ranges`` 告诉 Scheduler：「这个节点需要对某个维度做归约」。在 codegen 阶段，TritonKernel 会将归约维度的索引映射为 ``tl.sum(value, axis)``。

** 例 3：TemplateBuffer 模式——aten.mm**

.. code-block:: python

   @register_lowering(aten.mm)
   def mm_lower(x, y):
       # 矩阵乘法使用预定义的 Triton GEMM 模板
       # 不会生成 Pointwise 逐元素代码
       return TemplateBuffer(
           layout=FixedLayout(
               device=x.get_device(),
               size=[M, K, N],
           ),
           inputs=[x, y],
           kernel_template="atlas_gemm",
       )

``TemplateBuffer`` 不会被 Scheduler 进一步融合。在 codegen 阶段，它直接使用手写的 Triton GEMM 模板（如 ``atlas_gemm``），而不是通过 ``inner_fn`` 逐元素生成。

** 例 4：ExternKernel 模式——aten.sort**

.. code-block:: python

   # aten.sort 没有 register_lowering 装饰器
   # 调用 ExternKernel.create(aten.sort, x, dim=-1)
   # ExternKernel.create 内部：
   #   1. 分配输出 buffer
   #   2. 注册一个"在运行时调用 aitemplate.sort"的操作
   #   3. 返回 ExternKernel 节点

``ExternKernel`` 不生成 Triton 或 C++ 代码。它在编译后的 wrapper 中插入对 PyTorch eager 函数的调用：``buf0 = torch.sort(x, dim=-1)``。

TensorBox 解包与包装
-----------------------------

整个 lowering 过程中， **TensorBox 的解包和重新包装** 是贯穿始终的机械性工作：

.. code-block:: text

   FX 节点输入
       │  TensorBox → StorageBox → InputBuffer(x)
       │
       ▼
   fetch_args_kwargs_from_env
       │  解包：只传递 TensorBox 给 call_function
       │  call_function 内部再解包到 InputBuffer
       │
       ▼
   lowering 函数内部
       │  通过 TensorBox.get_size() 获取形状
       │  通过 ops.load(x, idx) 加载数据
       │  idx 是 TensorBox 内部布局的索引表达式
       │
       ▼
   返回新的 IRNode
       │  Pointwise 被重新包装为 TensorBox
       │  TensorBox(Pointwise(...))
       │
       ▼
   下一级 lowering 函数消费时再次解包

这种"解包 → 计算 → 重新包装"的模式贯穿整个 lowering 过程。正是这种模式使得 Inductor 的 IR 具有 **惰性求值** 的特性——``inner_fn`` 在被 codegen 调用之前不会实际执行任何计算。

TensorBox 的作用不只是"包装"。它还负责 **视图布局管理**——当一个操作产生视图（如 ``x.T``、``x[1:]``）时，TensorBox 会记录布局信息而不是实际创建新的 buffer：

.. code-block:: python

   # x.T 的 lowering:
   # 不创建新的 StorageBox，只创建新的 TensorBox
   # 包裹相同的 StorageBox，但布局变为 AsStridedLayout(transposed=True)
   TensorBox(
       data=StorageBox(InputBuffer(x)),
       layout=AsStridedLayout(transposed=True),
   )

这种"视图虚拟化"（详见第 5.4 节）使得 Scheduler 在融合时能正确处理别名关系，不会因为视图操作而错误地阻止融合。

Broadcast 与 Type Promotion
------------------------------------

Lowering 框架自动处理两个常见语义：

**Broadcast** ：当 ``register_lowering`` 的 ``broadcast=True`` 时，lowering 框架在调用函数前自动对参数做广播展开：

.. code-block:: text

   aten.add(x, y)  # x.shape=[3,1], y.shape=[1,4]
       │
       ▼
   自动广播后：x.shape=[3,4], y.shape=[3,4]
       │
       ▼
   Pointwise(inner_fn=lambda idx: ops.load(x, idx) + ops.load(y, idx))

**Type Promotion** ：当操作数类型不同时（如 float32 + float16），``type_promotion_kind`` 参数控制提升规则。Inductor 在 lowering 阶段预先确定输出类型，避免在 codegen 阶段再做类型判断。这符合第 2.1 节的"编译重、运行轻"原则。

Fallback 机制
==================

并不是 all 算子都有 lowering 函数。当 ``GraphLowering.call_function`` 遇到一个不在 ``lowerings`` 字典中的算子时，会触发 fallback：

``call_function`` 的方法签名在 ``pytorch/torch/_inductor/graph.py`` 第 1319 行：

.. code-block:: python

   def call_function(self, target, args, kwargs):
       if target not in lowerings:
           # 检查是否可以通过 decomposition 展开
           if has_decomposition(target):
               decomps = get_decompositions([target])
               # 展开后重新追踪
               ...
           else:
               # 创建 ExternKernel 回退
               # 编译时调用 eager 实现
               return ExternKernel.create(target, *args,**kwargs)
       
       return lowerings[target](*args, **kwargs)

``ExternKernel`` 是一种特殊的 IRNode——它不生成 Triton 或 C++ 代码，而是在编译时调用 PyTorch 的 eager 实现。这意味着即使某个算子没有 lowering 函数，Inductor 也不会崩溃，而是默默回退到 eager 执行。对于没有完全覆盖的算子集合，这个机制保证了编译的健壮性。

输入和输出的处理
======================

除了 ``call_function`` 节点外，``GraphLowering`` 还处理另外两种关键节点：

**placeholder（输入） ** ：每个 ``placeholder`` 节点对应一个函数输入。``GraphLowering`` 将其包装为 ``InputBuffer``（是 ``ComputedBuffer`` 的子类）：

.. code-block:: python

   # run_placeholder 内部
   input_buffer = InputBuffer(name=arg_name, layout=FixedLayout(
       device=device, dtype=dtype, size=size, stride=stride,
   ))
   self.graph_inputs[arg_name] = TensorBox(input_buffer)

**output（输出） ** ：``output`` 节点收集所有返回值对应的 IRNode，存入 ``self.graph_outputs`` 列表，供后续的 wrapper 代码生成使用。

降级结果的形态
====================

lowering 完成后，``GraphLowering`` 中积累了以下数据：

.. code-block:: text

   GraphLowering 实例
       │
       ├─ operations: [IRNode1, IRNode2, ...]   ← Scheduler 的输入
       │    每个 entry 是一个 ir.Operation 子类
       │    （Pointwise, Reduction, ExternKernel...）
       │
       ├─ buffers: [Buffer1, Buffer2, ...]       ← 所有 buffer 的列表
       │
       ├─ graph_inputs: {name: TensorBox, ...}   ← 输入映射
       │
       └─ graph_outputs: [IRNode, ...]           ← 输出列表

``operations`` 列表是 Scheduler 的直接输入。Scheduler 遍历这个列表，分析每个 IRNode 之间的依赖关系，然后执行融合和调度。

小结
======

这一节介绍了 FX Graph 到 IRNode 的降级过程：

- **GraphLowering** 继承 ``torch.fx.Interpreter``，按拓扑序遍历 FX 节点
- **call_function** 的完整调度逻辑包括：TensorBox 解包、lowerings 查表、alignment check、decomposition 回退、ExternKernel 降级，最后重新包装为 TensorBox 并注册到 operations 列表
- **四种 lowering 模式 ** ：逐元素（Pointwise）、归约（Reduction）、模板（TemplateBuffer）、外部（ExternKernel），每种模式对应不同的 IRNode 类型和生成策略
- **lowerings 表 ** 通过 ``register_lowering`` 装饰器增量构建，覆盖 600+ ATen 算子。Broadcast 和 type promotion 由 lowering 框架自动处理
- **TensorBox 解包与包装** 贯穿始终，使 IR 具有惰性求值的特性——``inner_fn`` 在被 codegen 调用之前不会执行实际计算
- 降级结果存储在 ``operations`` 列表中，作为 Scheduler 的输入
