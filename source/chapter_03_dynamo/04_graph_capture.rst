.. _graph-capture:

=============
图捕获
=============

.. note::

   **Dynamo 的提交流中，超过 12% 是 Revert。 **
   在 Dynamo 的 6,439 次历史提交中，revert 占了 815 次（约 12.7%）。这反映了团队的一个开发哲学：** 先 merge，错了再 revert**，而不是花很长时间逐行 review。在高速迭代的编译器项目中，这比"完美 review 再 merge"更高效——因为有些编译器的 bug 只在特定模型、特定 GPU 上才会出现，review 阶段很难发现。PyTorch 的 CI 虽然严格，但不会覆盖所有模型。所以团队选择用"revert 安全网"代替"review 放大镜"。

第 3.1 节从设计目标出发，说明了 FakeTensor、Guard、Graph Break 如何组成一条链。第 3.3 节讲了 **InstructionTranslator 如何驱动字节码、如何通过 ``call_function`` 派发** 。本节站在 **值语义** 一侧：VariableTracker、Proxy、FakeTensor 如何把一次派发变成 FX Graph 里的节点。

这一节是第 3 章的核心之一。我们追踪 ``torch.sin(x)`` 在 ``call_function`` 进入 ``BuiltinVariable`` 之后，如何创建 Proxy、插入 ``call_function`` 节点。

VariableTracker：一切皆变量
===============================

在 Dynamo 的符号执行环境中， **每个 Python 对象都被包装成一个 ``VariableTracker``** 。无论是 Tensor、int、函数、还是 Module，它们在 InstructionTranslator 的模拟栈上都是以 ``VariableTracker`` 子类的形式存在的。

.. code-block:: text

   真实 Python 值           Dynamo 中的 VariableTracker
   ──────────────────       ────────────────────────────────
   torch.Tensor             TensorVariable
   int / float              ConstantVariable
   function / method        UserDefinedFunctionVariable
   nn.Module                NNModuleVariable
   list / tuple             ListVariable / TupleVariable
   torch.dtype              ConstantVariable
   ...                      ...

这个抽象层的作用是：InstructionTranslator 不需要区分"这是一个 Tensor 还是一个 int"——它看到的所有值都是 ``VariableTracker`` ，只需要调用统一的接口（如 ``call_function`` 、 ``var_getattr`` ），具体的行为由各个子类实现。

核心实现位于 ``pytorch/torch/_dynamo/variables/`` 目录：

.. code-block:: text

   torch/_dynamo/variables/
   ├── tensor.py       # TensorVariable：张量操作追踪
   ├── torch.py        # TorchVariable：torch.* API 调用
   ├── nn_module.py    # NNModuleVariable：nn.Module 调用
   ├── builtin.py      # BuiltinVariable：内置函数（add, mul 等）
   ├── functions.py    # 用户函数和闭包
   ├── lists.py        # 列表/元组操作
   └── ...

追踪一条 ``torch.sin(x)``
==============================

第 3.3 节已说明 ``CALL_FUNCTION`` handler 会调用 ``self.call_function(fn, args, {})`` 。下面从 **VariableTracker 派发** 侧，补全 ``torch.sin(x)`` 如何变成 FX 节点（字节码逐步弹栈过程见第 3.3 节分发表）。

.. code-block:: text

   CALL_FUNCTION
       → call_function(BuiltinVariable(torch.sin), [TensorVariable(x)], {})
           → BuiltinVariable.call_function
               → SubgraphTracer.create_proxy("call_function", torch.sin, ...)
                   → FX Graph 插入 %sin 节点
                       → 返回 TensorVariable(sin_proxy)

执行完 ``return torch.sin(x) + torch.cos(x)`` 对应的所有字节码后，FX Graph 类似：

.. code-block:: text

   graph():
       %x : [num_users=1] = placeholder[target=x]
       %sin : [num_users=2] = call_function[target=torch.sin](args = (%x,), kwargs = {})
       %cos : [num_users=1] = call_function[target=torch.cos](args = (%x,), kwargs = {})
       %add : [num_users=1] = call_function[target=torch.add](args = (%sin, %cos), kwargs = {})
       return add

创建 FX 节点：SubgraphTracer 的角色
==========================================

核心的 FX 节点创建发生在 ``output_graph.py`` 中的 ``SubgraphTracer`` 类，它继承自 ``torch.fx.Tracer`` ：

.. code-block:: python
   :caption: pytorch/torch/_dynamo/output_graph.py（简化示意）

   class SubgraphTracer(fx.Tracer):
       def create_proxy(self, kind, target, args, kwargs, ...):
           """创建并返回一个 torch.fx.Proxy"""
           # 标准的 fx.Tracer.create_proxy 逻辑
           # 会在图中插入一个新的节点
           return super().create_proxy(kind, target, args, kwargs, ...)

       def create_node(self, kind, target, args, kwargs, ...):
           """直接创建 FX Node"""
           # 在 SubgraphTracer 的 graph 中创建节点
           return super().create_node(kind, target, args, kwargs, ...)

当 ``BuiltinVariable`` 需要为 ``torch.sin(x)`` 创建 FX 节点时，调用链如下：

.. code-block:: text

   BuiltinVariable.call_function(fn=torch.sin, args=[x])
       │
       ├─ 1. 识别操作类型
       │      torch.sin 是一个 torch.* 函数
       │
       ├─ 2. 创建 Proxy
       │      proxy = output_graph.tracer.create_proxy(
       │          "call_function",
       │          torch.sin,
       │          args=(x_proxy,),
       │          kwargs={},
       │      )
       │      # 此时 FX Graph 中已经插入了一个新节点
       │
       ├─ 3. 包装为 TensorVariable
       │      result = TensorVariable(proxy)
       │      # TensorVariable 持有这个 proxy，后续操作通过它引用
       │
       └─ 4. 返回给调用者
              压入栈顶

Proxy 的作用很重要：它既是一个 FX 节点（在图中有位置），又是一个"假张量"（可以像 Tensor 一样传递）。后续的操作（比如将 sin 的结果传给 add）通过引用 proxy 而不是引用具体数值，这样就自动建立了图的数据依赖关系。

**Proxy 是连接符号执行和 FX Graph 的桥梁。 **InstructionTranslator 看到的是 ``TensorVariable(proxy)`` ，而 ``proxy`` 内部持有对 FX Graph 中某个节点的引用。

FakeTensor：符号执行中的"假"张量
=========================================

这里有一个关键问题：当 Dynamo 执行 ``torch.sin(x)`` 时，图捕获是成功了，但 ``torch.sin`` 的实际计算并没有发生。那如果后续代码依赖 ``x.sin()`` 的结果（比如检查它的形状、dtype 等），Dynamo 怎么处理？

答案是**FakeTensor** 。Dynamo 在开始符号执行之前，会将所有输入张量替换为 ``FakeTensor`` 。FakeTensor 具有真实张量的所有元数据（形状、dtype、device），但不包含实际数据。

.. code-block:: python
   :caption: FakeTensor 的简化示意

   class FakeTensor:
       def __init__(self, shape, dtype, device):
           self.shape = shape
           self.dtype = dtype
           self.device = device
           # 没有 .data 或实际存储

       def sin(self):
           # 返回一个新的 FakeTensor，形状相同
           return FakeTensor(self.shape, self.dtype, self.device)

FakeTensor 让 Dynamo 可以在不实际计算的情况下"假装"执行代码。当代码检查 ``x.shape`` 时，FakeTensor 可以正确返回。当代码检查 ``x.sum() > 0`` 时，FakeTensor 无法提供真实数值——这时就需要 graph break 或者使用 symbolic shapes（见第 3.8 节）。

Proxy 和 FakeTensor 的关系
===============================

在实际实现中，Proxy 和 FakeTensor 是结合使用的。当一个 Tensor 操作被追踪时：

.. code-block:: text

   真实张量
       │
       ├─ 元数据 → 用于创建 FakeTensor（形状、dtype）
       │
       └─ 用于创建 Proxy（在 FX Graph 中占位）
           │
           ▼
   TensorVariable {
       proxy: Proxy(%x)      # 在 FX Graph 中的位置
       fake_tensor: FakeTensor(shape=(3,3), dtype=float32)  # 元数据
   }

当 Dynamo 后续需要访问这个张量的形状时（如 ``x.shape`` ），它使用 FakeTensor。当它需要引用这个张量在图中的位置时（如作为 ``torch.sin`` 的参数），它使用 Proxy。

OutputGraph：图的最终构建
==================================

符号执行完成后， ``InstructionTranslator.output`` 中包含了一个 ``OutputGraph`` 对象，它持有最终的 FX Graph：

.. code-block:: python
   :caption: pytorch/torch/_dynamo/output_graph.py 关键类

   class OutputGraph:
       def __init__(self, ...):
           self.tracer = SubgraphTracer()  # 持有 FX Graph
           self.guards = []                 # 累积的 guard 条件

       def add_guard(self, guard):
           """在编译过程中添加一个 guard"""
           self.guards.append(guard)

       @property
       def graph(self):
           """返回构建完成的 FX Graph"""
           return self.tracer.graph

``OutputGraph`` 在符号执行过程中积累了两样东西：

1.**FX Graph** ：通过 ``tracer.create_proxy`` 调用逐步构建
2.**Guards** ：每次对张量形状/属性做检查时，InstructionTranslator 会调用 ``output.add_guard`` 记录检查条件

这两样东西最终被一起传给后端编译器。

小结
======

这一节我们追踪了 ``torch.sin(x)`` 从字节码到 FX Graph 节点的完整路径：

1.**VariableTracker** 统一包装层：所有 Python 值被包装为同态对象
2.**SubgraphTracer**/**Proxy** ：通过 ``fx.Tracer.create_proxy`` 在图中插入节点
3.**FakeTensor** ：提供元数据但不执行实际计算
4.**OutputGraph** ：持有最终的 FX Graph 和累积的 Guards

这些机制协同工作，使得 Dynamo 可以在完全不修改用户代码的前提下，将任意 Python 函数的 Tensor 操作部分提取为一张可编译的计算图。

下一节我们来看 guard 机制——它是如何保证缓存的编译结果对新的输入仍然有效的。
