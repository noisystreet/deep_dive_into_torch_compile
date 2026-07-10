.. _compilation-pipeline:

======================
编译流水线
======================

.. seealso::

   **为什么选择三阶段架构？ **
   在 torch.compile 之前，PyTorch 的编译器尝试（TorchScript）试图在一个阶段内完成"图捕获 + 自动微分 + 代码生成"。这种做法耦合度高，任何一个环节出错整个编译就失败了。三阶段架构的设计灵感来自 LLVM——** 每个阶段只做一件事，且通过标准化的中间表示（FX Graph）通信 **。Dynamo 输出 FX Graph，AOTAutograd 消费并变换它，Inductor 消费变换后的图。这种松耦合的设计使得每个组件可以独立测试、独立演进。例如，社区已经开发了 ``torch-xla`` 后端，它直接消费 FX Graph 并生成为 XLA 代码，完全跳过了 Inductor。

编译栈的设计原则
==========================

第 1.1 节我们从历史角度回顾了 TorchScript → torch.compile 的三次路线演变。这里把散落在各章的设计决策收成** 六条原则** ，作为阅读后续章节的「透镜」——遇到具体机制时，可以问：它体现了哪条原则？放弃了什么？代价是什么？

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - 原则
     - 含义
     - 主要体现
   * - **编译器适应 Python**
     - 不强迫用户改写代码；无法捕获时 graph break，而非报错退出
     - Dynamo 帧拦截、graph break（第 3 章）
   * - **阶段专精 + 标准 IR**
     - 捕获、求导、代码生成分离；FX Graph 作为组件间契约
     - 三目录架构（本节下文）、自定义 backend（第 9.1 节）
   * - **策略与机制分离**
     - 中间层只提供机制；谁消费图，谁定策略
     - decomposition 配置在 Inductor、执行在 AOTAutograd（第 4.6 节）
   * - **正确性优先于性能**
     - guard 失败则重编译；缓存超限则 fallback eager，宁可慢也不能错
     - guard（第 3.5 节）、缓存 fallback（第 3.7 节）、三层缓存（第 2.4 节）
   * - **Define-by-Run**
     - IR 随 lower 过程逐级构建，保留 PyTorch「边执行边定义」的语义
     - Inductor IRNode（第 5.1 节）
   * - **编译重、运行轻**
     - 编译可以慢（autotune、重编译）；推理/训练 loop 必须快
     - 磁盘缓存（第 2.4 节）、AOTInductor（第 9.5 节）、 ``max-autotune`` （第 9.2 节）

用一个具体场景串起这些原则。假设用户写了：

.. code-block:: python

   @torch.compile
   def train_step(x, y):
       print("step")          # Python 副作用
       return (x * y).sum()

第一次调用 ``train_step(x, y)`` 时发生了什么？

1. **编译器适应 Python** ：Dynamo 不会因为有 ``print`` 就拒绝编译，而是在 ``print`` 处 graph break，前后各形成一个子图（第 3.6 节）。
2.**阶段专精 ** ：Dynamo 只负责捕获前向 FX Graph；AOTAutograd 在编译期追联合反向图；Inductor 只负责生成 kernel——各干各的。
3.**策略与机制分离** ：Inductor 的 ``select_decomp_table()`` 决定 ``(x*y).sum()`` 要不要分解为基本算子，AOTAutograd 只执行分解，不关心 Triton 怎么写。
4. **正确性优先 ** ：若下次 ``x.shape`` 变了，guard 失败触发重编译；若重编译次数过多，fallback 到 eager 而不是 silent wrong result。
5.**Define-by-Run** ：Inductor 在遍历 FX 节点时逐个 ``lower`` 出 IRNode，而不是先建完整静态 IR 再优化。
6.**编译重、运行轻 ** ：首次调用可能要数秒编译；从第二次起命中缓存，执行接近手写 kernel 的速度。

.. code-block:: text

   设计原则              在本例中的体现
   ─────────────────────────────────────────────
   适应 Python           print → graph break，其余仍编译
   阶段专精              Dynamo 捕获 → AOT 求导 → Inductor codegen
   策略/机制分离         Inductor 定 decomposition，AOT 执行
   正确性优先            shape 变 → guard → 重编译
   Define-by-Run         逐节点 lower 为 IRNode
   编译重、运行轻        首次慢，后续命中缓存

后续第 3–6 章会分别深入三大组件；读每一章时，不妨回到这张表，看看源码里的类名和函数名分别对应哪条原则。

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

这个布局本身就是架构的反映：**三个目录对应三个组件，目录之间的调用链就是编译流水线 ** 。

一次完整的编译调用链
============================

从用户调用 ``compiled_fn = torch.compile(fn)`` 开始，到真正执行 ``compiled_fn(x)`` ，完整的调用路径是这样的：

第一步：torch.compile 注册帧拦截
---------------------------------------

.. code-block:: python

   compiled_fn = torch.compile(fn)

这行代码实际上做了什么？我们查看源码入口。

``torch.compile`` （位于 ``torch/compiler/__init__.py`` ）最终调用到 ``torch._dynamo.optimize()`` ，它的核心作用是**注册一个帧拦截回调** 到 CPython 解释器。

关键实现在 ``torch/_dynamo/eval_frame.py`` 中。Dynamo 通过 ``set_eval_frame`` 这个 C++ 扩展函数（定义在 ``torch/_C/_dynamo/eval_frame.cpp`` ），将自己挂入 CPython 的帧执行回调：

.. code-block:: python
   :caption: pytorch/torch/_dynamo/eval_frame.py（简化示意）

   def optimize(backend, *, ...):
       def decorator(fn):
           @functools.wraps(fn)
           def compiled_fn(*args, **kwargs):
               # 注册帧拦截回调
               old_callback = set_eval_frame(_fn_to_frame_eval(backend))
               try:
                   return fn(*args,**kwargs)
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

1.**FX Graph**—— 捕获到的计算图
2.**Guards**—— 未来验证缓存是否有效的条件
3.**Example inputs** —— 输入的 FakeTensor 样本

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

对于默认的 ``inductor`` 后端（定义在 ``torch/_dynamo/backends/inductor.py`` ），它会懒加载 ``torch._inductor.compile_fx`` ：

.. code-block:: python
   :caption: pytorch/torch/_dynamo/backends/inductor.py

   @register_backend
   def inductor(*args, **kwargs):
       from torch._inductor.compile_fx import compile_fx
       return compile_fx(*args,**kwargs)

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

AOTAutograd 的核心代码在 ``torch/_functorch/aot_autograd.py`` 中。它使用 ``torch.fx.experimental.proxy_tensor.make_fx`` 对 FX Graph 执行一次"假反向传播"——即用代理张量走一遍 autograd 流程，同时记录所有反向操作，生成一张包含前向和反向的 ** 联合计算图**（joint graph）。

然后 ``partitioners.py`` 中的 ``min_cut_rematerialization_partition`` 对这联合图进行切分：哪些计算在前向做、哪些在反向做、哪些在反向中重计算以节省显存。

第五步：Inductor 接收分片后的子图
----------------------------------------

每个子图（前向图、反向图）会独立进入 Inductor 的编译流程。入口是 ``torch/_inductor/compile_fx.py`` 中的 ``compile_fx_inner`` ：

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

``torch/_inductor/lowering.py`` 负责步骤 2：它维护了一张从 FX 操作到 Inductor IR 的映射表，每个 ``torch.sin`` 、 ``torch.cos`` 都会被翻译成对应的 ``IRNode`` 。我们会在第 5 章深入这部分的实现。

``torch/_inductor/scheduler.py`` 负责步骤 3：它分析 IRNode 之间的数据依赖，构建依赖图，然后通过启发式算法将可以融合的操作分组。融合后的组直接被发送给代码生成器。

第六步：返回编译后的函数
----------------------------------------

最终，Inductor 返回一个 ``CompiledFxGraph`` 对象（定义在 ``torch/_inductor/output_code.py`` ），它包含了：

- 生成的 Triton kernel 代码
- kernel launch 的包装函数
- 缓存信息

这个对象被包装成普通的 Python callable，返回给用户。以后每次调用 ``compiled_fn(x, y)`` ，都会：

1. 检查 guard 是否通过
2. 命中缓存，直接执行编译好的 kernel
3. 跳过整个编译流水线

时序总览
============

下面这张序列图总结了整个编译过程的时间线：

.. mermaid::

   sequenceDiagram
       participant 用户代码
       participant Dynamo
       participant AOTAutograd
       participant Inductor

       用户代码->>Dynamo: torch.compile
       Dynamo-->>用户代码: 注册回调

       用户代码->>Dynamo: compiled_fn()
       Dynamo->>Dynamo: convert_frame
       Note over Dynamo: FX Graph

       Dynamo->>AOTAutograd: lookup_backend
       AOTAutograd->>AOTAutograd: aot_autograd
       Note over AOTAutograd: Joint Graph + Partition

       AOTAutograd->>Inductor: compile_fx
       Inductor->>Inductor: lowering
       Inductor->>Inductor: scheduler
       Inductor->>Inductor: codegen

       Inductor-->>AOTAutograd: cached fn
       AOTAutograd-->>Dynamo: cached fn
       Dynamo-->>用户代码: cached fn

       用户代码->>Dynamo: 后续调用
       Dynamo->>Dynamo: guard check → hit cache
       Dynamo-->>用户代码: 结果

对应前面 "设计原则在本例中的体现" 的六条原则：序列图中的每一次箭头跨越，都对应一次阶段间的 IR 传递；而后续调用时 Dynamo 内部的 "guard check → hit cache" 则体现了 ** 编译重、运行轻**和 ** 正确性优先** 。

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
- **符号执行引擎 ** 用 FakeTensor 模拟 Tensor 操作，同时构建 FX Graph
- **三阶段流水线 ** （Dynamo → AOTAutograd → Inductor）松耦合，每阶段可独立替换
- **六条设计原则 ** （见上文）贯穿三大组件，是理解各章实现细节的纲
- **编译结果被缓存** ，后续调用直接命中缓存跳过编译

下一节我们换个视角，看看编译过程中数据（张量）和控制流在不同组件之间是如何流转的。后面 2.4 节还会专题讨论贯穿三个组件的编译缓存架构。
