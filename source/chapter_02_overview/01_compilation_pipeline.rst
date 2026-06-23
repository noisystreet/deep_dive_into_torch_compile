.. _compilation-pipeline:

======================
编译流水线
======================

第 1 章我们从"编译 vs 解释"的角度了解了 torch.compile 的基本概念。这一节我们打开引擎盖，看看一次完整的编译过程到底是怎么走完的。我们会追踪一次 ``torch.compile(fn)(x)`` 调用，从 Python 函数入口一直跟踪到生成的 Triton kernel。

源码结构总览
==========================

在动手之前，先看一眼 PyTorch 源码中与 torch.compile 相关的核心模块布局。这些目录就是本章要讲解的三组件在磁盘上的真实位置。

.. code-block:: text

   pytorch/torch/
   ├── _dynamo/                 # TorchDynamo：图捕获前端
   │   ├── eval_frame.py        #   帧拦截入口（PEP 523 回调）
   │   ├── convert_frame.py     #   帧 → FX Graph 转换
   │   ├── symbolic_convert.py  #   符号执行引擎
   │   ├── guards.py            #   Guard 生成与检查
   │   ├── bytecode_analysis.py #   字节码分析
   │   └── backends/
   │       ├── registry.py      #   后端注册与查找
   │       └── inductor.py      #   Inductor 后端入口
   │
   ├── _functorch/              # AOTAutograd：自动微分 + 图分区
   │   ├── aot_autograd.py      #   联合前向/反向求导
   │   ├── _aot_autograd/
   │   │   ├── partitioners.py  #   图分区策略（min-cut 等）
   │   │   └── functional_utils.py  # 功能化变换
   │   └── ...
   │
   └── _inductor/               # Inductor：代码生成后端
       ├── compile_fx.py        #   主入口，编排编译流程
       ├── graph.py             #   Inductor 图表示
       ├── ir.py                #   IRNode 定义
       ├── lowering.py          #   FX Graph → IRNode 降级
       ├── scheduler.py         #   融合与调度
       ├── fx_passes/           #   FX 图优化 pass
       └── codegen/
           ├── triton.py        #   GPU: Triton 代码生成
           ├── cpp.py           #   CPU: C++/OpenMP 代码生成
           └── wrapper.py       #   调用包装器

这个布局本身就是架构的反映：**三个目录对应三个组件，目录之间的调用链就是编译流水线**。

一次完整的编译调用链
============================

从用户调用 ``compiled_fn = torch.compile(fn)`` 开始，到真正执行 ``compiled_fn(x)``，完整的调用路径是这样的：

第一步：torch.compile 注册帧拦截
---------------------------------------

.. code-block:: python

   compiled_fn = torch.compile(fn)

这行代码实际上做了什么？我们查看源码入口。

``torch.compile`` （位于 ``torch/compiler/__init__.py``）最终调用到 ``torch._dynamo.optimize()``，它的核心作用是**注册一个帧拦截回调**到 CPython 解释器。

关键实现在 ``torch/_dynamo/eval_frame.py`` 中。Dynamo 通过 ``set_eval_frame`` 这个 C++ 扩展函数（定义在 ``torch/_C/_dynamo/eval_frame.cpp``），将自己挂入 CPython 的帧执行回调：

.. code-block:: python
   :caption: pytorch/torch/_dynamo/eval_frame.py（简化示意）

   def optimize(backend, *, ...):
       def decorator(fn):
           @functools.wraps(fn)
           def compiled_fn(*args, **kwargs):
               # 注册帧拦截回调
               old_callback = set_eval_frame(_fn_to_frame_eval(backend))
               try:
                   return fn(*args, **kwargs)
               finally:
                   set_eval_frame(old_callback)
           return compiled_fn
       return decorator

这里的关键是 ``set_eval_frame`` ——这个 C 函数通过 PEP 523 提供的 ``PyInterpreterState.eval_frame`` 钩子，让 Dynamo 在每个 Python 帧开始执行时获得一次拦截机会。

PEP 523 是在 CPython 3.6 中引入的（`peps.python.org/pep-0523 <https://peps.python.org/pep-0523/>`__），它允许外部框架注册一个回调函数，该函数在每次 Python 帧开始执行时被调用。Dynamo 是 PEP 523 在深度学习框架中最具影响力的应用。

第二步：第一次调用触发帧捕获
----------------------------------------

当用户真正调用 ``compiled_fn(x)`` 时，CPython 解释器开始执行 ``fn`` 的帧。Dynamo 的回调被触发，进入 ``torch/_dynamo/convert_frame.py`` 的核心逻辑：

.. code-block:: text

   compiled_fn(x, y)
       │
       ▼
   set_eval_frame 拦截帧
       │
       ▼
   convert_frame 开始处理：
       │
       ├─ 1. 检查是否需要跳过（trace_rules）
       │
       ├─ 2. 创建符号执行环境（symbolic_convert）
       │     用 FakeTensor 替换真实张量
       │
       ├─ 3. 逐条执行字节码，同时记录所有
       │     Tensor 操作的调用
       │
       ├─ 4. 输出 FX Graph + Guards
       │
       └─ 5. 查找后端，发送 FX Graph

``torch/_dynamo/symbolic_convert.py`` 中的 ``InstructionTranslator`` 类是符号执行引擎的核心。它模拟 CPython 解释器的栈和局部变量，但将所有 Tensor 操作记录下来而不是真正执行：

.. code-block:: python
   :caption: pytorch/torch/_dynamo/symbolic_convert.py（简化示意）

   class InstructionTranslator:
       def __init__(self, code, ...):
           self.stack = []        # 模拟 Python 值栈
           self.locals = {}       # 模拟局部变量
           self.graph = ...       # 正在构建的 FX Graph

       def CALL_FUNCTION(self, n):
           args = [self.stack.pop() for _ in range(n)]
           fn = self.stack.pop()
           # 检查是否是 Tensor 操作
           if isinstance(fn, torch.Tensor.__class__):
               # 在 FX Graph 中插入一个 call_function 节点
               self.graph.call_function(fn, args)
           else:
               # 非 Tensor 操作 → graph break
               self.graph_break()

这种逐字节码捕获的方式是 Dynamo 区别于 TorchScript 和 FX Graph 的关键创新。

第三步：查找后端并传递 FX Graph
----------------------------------------

``convert_frame`` 完成后，产生了三样东西：

1. **FX Graph** —— 捕获到的计算图
2. **Guards** —— 未来验证缓存是否有效的条件
3. **Example inputs** —— 输入的 FakeTensor 样本

这三样东西被传递给 Dynamo 的后端调度器。调度器在 ``torch/_dynamo/backends/registry.py`` 中根据用户指定的 ``backend`` 名称查找对应的编译器函数：

.. code-block:: text

   lookup_backend("inductor")
       │
       ▼
   找到 entry point: torch._inductor.compile_fx.compile_fx
       │
       ▼
   调用 compile_fx(graph_module, example_inputs)
       │
       ▼
   进入 Inductor 主循环

对于默认的 ``inductor`` 后端（定义在 ``torch/_dynamo/backends/inductor.py``），它会懒加载 ``torch._inductor.compile_fx``：

.. code-block:: python
   :caption: pytorch/torch/_dynamo/backends/inductor.py

   @register_backend
   def inductor(*args, **kwargs):
       from torch._inductor.compile_fx import compile_fx
       return compile_fx(*args, **kwargs)

第四步：AOTAutograd 处理自动微分
----------------------------------------

.. code-block:: text

   compile_fx(graph_module, example_inputs)
       │
       ▼
   aot_autograd(graph_module, example_inputs)
       │
       ├─ 1. 对 FX Graph 运行 Autograd
       │      ┌─────────────────────────────────┐
       │      │  Forward Graph                  │
       │      │  x → sin → cos → add → output   │
       │      └──────────┬──────────────────────┘
       │                 │ autograd.Function
       │                 ▼
       │      ┌──────────────────────────────────┐
       │      │  Joint Graph（前向+反向）        │
       │      │  Forward: x → sin → ... → output │
       │      │  Backward: grad → ... → grad_x   │
       │      └──────────────────────────────────┘
       │
       ├─ 2. 图分区（partition）
       │      Forward Subgraph  │  Backward Subgraph
       │
       ├─ 3. 对每个子图分别调用后端编译器
       │      compile_fx_inner(fwd_subgraph)
       │      compile_fx_inner(bwd_subgraph)
       │
       └─ 4. 返回包装后的 forward_fn + backward_fn

AOTAutograd 的核心代码在 ``torch/_functorch/aot_autograd.py`` 中。它使用 ``torch.fx.experimental.proxy_tensor.make_fx`` 对 FX Graph 执行一次"假反向传播"——即用代理张量走一遍 autograd 流程，同时记录所有反向操作，生成一张包含前向和反向的**联合计算图**（joint graph）。

然后 ``partitioners.py`` 中的 ``min_cut_rematerialization_partition`` 对这联合图进行切分：哪些计算在前向做、哪些在反向做、哪些在反向中重计算以节省显存。

第五步：Inductor 接收分片后的子图
----------------------------------------

每个子图（前向图、反向图）会独立进入 Inductor 的编译流程。入口是 ``torch/_inductor/compile_fx.py`` 中的 ``compile_fx_inner``：

.. code-block:: text

   compile_fx_inner(subgraph_module, example_inputs)
       │
       ├─ 1. FX Passes（图级别优化）
       │      在 FX Graph 上运行优化 pass
       │      （fx_passes/ 目录）
       │
       ├─ 2. Lowering（降级）
       │      FX Graph 中的每个 call_function
       │      被映射为 Inductor IRNode
       │      （lowering.py）
       │
       ├─ 3. Scheduler（调度 + 融合）
       │      分析 IRNode 之间的依赖关系，
       │      将可以融合的节点分组
       │      （scheduler.py）
       │
       ├─ 4. Codegen（代码生成）
       │      GPU → Triton 代码（codegen/triton.py）
       │      CPU → C++/OpenMP 代码（codegen/cpp.py）
       │
       └─ 5. 编译 + 返回 callable
              编译生成的 Triton/C++ 代码，
              返回一个可直接调用的函数

``torch/_inductor/lowering.py`` 负责步骤 2：它维护了一张从 FX 操作到 Inductor IR 的映射表，每个 ``torch.sin``、``torch.cos`` 都会被翻译成对应的 ``IRNode``。我们会在第 5 章深入这部分的实现。

``torch/_inductor/scheduler.py`` 负责步骤 3：它分析 IRNode 之间的数据依赖，构建依赖图，然后通过启发式算法将可以融合的操作分组。融合后的组直接被发送给代码生成器。

第六步：返回编译后的函数
----------------------------------------

最终，Inductor 返回一个 ``CompiledFxGraph`` 对象（定义在 ``torch/_inductor/output_code.py``），它包含了：

- 生成的 Triton kernel 代码
- kernel launch 的包装函数
- 缓存信息

这个对象被包装成普通的 Python callable，返回给用户。以后每次调用 ``compiled_fn(x, y)``，都会：

1. 检查 guard 是否通过
2. 命中缓存，直接执行编译好的 kernel
3. 跳过整个编译流水线

时序总览
============

下面这张序列图总结了整个编译过程的时间线：

.. code-block:: text

   用户代码         Dynamo        AOTAutograd     Inductor
      │              │               │              │
      │ torch.compile│               │              │
      │─────────────→│               │              │
      │  注册回调     │               │              │
      │←─────────────│               │              │
      │              │               │              │
      │ compiled_fn()│               │              │
      │─────────────→│               │              │
      │              │               │              │
      │              │convert_frame   │              │
      │              │───→ FX Graph  │              │
      │              │               │              │
      │              │lookup_backend │              │
      │              │──────────────→│              │
      │              │               │ aot_autograd │
      │              │               │───→ Joint    │
      │              │               │───→ Partition│
      │              │               │              │
      │              │               │ compile_fx   │
      │              │               │─────────────→│
      │              │               │              │
      │              │               │              │ lowering
      │              │               │              │ scheduler
      │              │               │              │ codegen
      │              │               │              │
      │              │ cached fn     │              │
      │←─────────────│←──────────────│←─────────────│
      │              │               │              │
      │ 后续调用      │               │              │
      │─────────────→│ guard check   │              │
      │              │ hit cache     │              │
      │←─────────────│               │              │

关键源代码入口速查
==========================

.. list-table::
   :header-rows: 1

   * - 阶段
     - 源码路径
     - 核心函数
   * - 帧拦截注册
     - ``torch/_dynamo/eval_frame.py``
     - ``_fn_to_frame_eval``, ``set_eval_frame``
   * - 帧 → FX Graph
     - ``torch/_dynamo/convert_frame.py``
     - ``ConvertFrame``, ``convert_frame``
   * - 符号执行
     - ``torch/_dynamo/symbolic_convert.py``
     - ``InstructionTranslator``
   * - 后端查找
     - ``torch/_dynamo/backends/registry.py``
     - ``lookup_backend``
   * - AOTAutograd
     - ``torch/_functorch/aot_autograd.py``
     - ``aot_function``, ``aot_export_module``
   * - 图分区
     - ``torch/_functorch/_aot_autograd/partitioners.py``
     - ``min_cut_rematerialization_partition``
   * - Inductor 主入口
     - ``torch/_inductor/compile_fx.py``
     - ``compile_fx``, ``compile_fx_inner``
   * - 降级
     - ``torch/_inductor/lowering.py``
     - ``lower_to_ir``
   * - 调度与融合
     - ``torch/_inductor/scheduler.py``
     - ``Scheduler``
   * - Triton 代码生成
     - ``torch/_inductor/codegen/triton.py``
     - ``TritonKernel``, ``TritonScheduling``
   * - C++ 代码生成
     - ``torch/_inductor/codegen/cpp.py``
     - ``CPPKernel``, ``CPPScheduling``

小结
======

这一节我们追踪了一次完整的 torch.compile 调用链，从 ``torch.compile()`` 注册帧拦截，到 Dynamo 捕获 FX Graph，再到 AOTAutograd 联合求导和图分区，最后 Inductor 降级和代码生成。关键要点：

- Dynamo 通过 **PEP 523** 在字节码层级拦截帧执行
- **符号执行引擎** 用 FakeTensor 模拟 Tensor 操作，同时构建 FX Graph
- **三阶段流水线** （Dynamo → AOTAutograd → Inductor）松耦合，每阶段可独立替换
- **编译结果被缓存**，后续调用直接命中缓存跳过编译

下一节我们换个视角，看看编译过程中数据（张量）和控制流在不同组件之间是如何流转的。后面 2.4 节还会专题讨论贯穿三个组件的编译缓存架构。
