.. _what-is-torch-compile:

==========================
什么是 torch.compile
==========================

.. note::

   **torch.compile 的代号是 "TorchDynamo" 还是 "PyTorch 2.0"？**
   严格来说， ``torch.compile`` 是 PyTorch 2.0 引入的 **编译 API** ，而 TorchDynamo 是它底层的图捕获引擎。项目在开发阶段的内部代号是 "Dynamo"，团队最初只打算做一个更好的 TorchScript 替代品，但后来发现字节码方案比预期强大得多，最终整个编译栈变成了 PyTorch 2.0 最核心的新特性。PyTorch 2.0 发布会时，核心维护者 Soumith Chintala 说这是 "PyTorch 历史上最大的一次发布"。

从一个实际问题说起
====================

假设你写了一个简单的 PyTorch 函数：

.. code-block:: python
   :caption: 一个简单的计算函数

   def fn(x, y):
       for _ in range(100):
           x = (x.sin() + x.cos()) * y.tanh()
       return x

在原生 PyTorch 中，每一行 Python 代码都被逐行解释执行： ``x.sin()`` 启动一次 CUDA kernel， ``x.cos()`` 再启动一次， ``y.tanh()`` 又一次。循环 100 轮，就是 300 次 kernel launch + 300 次 Python 解释器开销。

如果用 ``torch.compile`` 包装一下：

.. code-block:: python

   compiled_fn = torch.compile(fn)
   result = compiled_fn(x, y)

它的执行路径完全不同了——不再是逐行解释，而是先将整个 ``fn`` 的计算过程捕获成一张计算图，再整体编译成一个高效的 GPU kernel。循环内部的 3 个操作会被 **融合** 成单个 kernel，一次 launch 就能跑完。

这就是 ``torch.compile`` 最核心的价值：**把 Python 级别的灵活表达，转化为编译器级别的极致优化** 。

编译与解释
================

要理解 torch.compile 做了什么，先看 PyTorch 通常是怎么运行代码的。

传统的 PyTorch 是 **解释执行** （eager mode）：

.. code-block:: text

   Python 源码
       │
       ▼
   Python 解释器 —— 逐条执行字节码
       │
       ├─ x.sin()   → launch kernel 1
       ├─ x.cos()   → launch kernel 2
       ├─ y.tanh()  → launch kernel 3
       └─ ...       → ...

每一步都完整走完"Python → C++ → CUDA"的调用链。这种方式的好处是灵活——你可以随时 ``print()`` 、断点调试、动态改变张量形状。代价是性能：**Python 解释器开销 + 频繁的 kernel launch + 错失融合机会** 。

torch.compile 换了一种思路：**先捕获，再编译，最后执行** 。

.. code-block:: text

   Python 源码
       │
       ▼
   TorchDynamo —— 在字节码级别捕获计算图
       │
       ▼
   AOTAutograd —— 生成前向+反向联合图
       │
       ▼
   Inductor —— 降级为循环级 IR → 生成 Triton/C++ 代码
       │
       ▼
   编译后的 kernel（一次 launch 跑完所有操作）

这就是 PyTorch 2.x 引入的 **编译模式** （compiled mode）。

一张图看懂全貌
====================

下面这张图展示了 torch.compile 的整体架构和调用链：

.. code-block:: text

   用户代码 (Python)
        │
        ▼
   ┌─────────────────────────────────────────────┐
   │  TorchDynamo                                 │
   │  • 通过 PEP 523 挂入 CPython 解释器           │
   │  • 在字节码层面捕获计算图 (FX Graph)          │
   │  • 遇到无法捕获的操作 → graph break          │
   └───────────────┬─────────────────────────────┘
                   │ FX Graph
                   ▼
   ┌─────────────────────────────────────────────┐
   │  AOTAutograd                                 │
   │  • 联合前向与反向，生成 joint graph           │
   │  • 图分区：前向子图 vs 反向子图               │
   │  • min-cut 重计算：用计算换内存               │
   └───────────────┬─────────────────────────────┘
                   │ 分片后的 FX Graph
                   ▼
   ┌─────────────────────────────────────────────┐
   │  Inductor                                    │
   │  • FX Graph → 虚拟化 → IRNode               │
   │  • Scheduler 融合、布局优化                  │
   │  • 代码生成：GPU→Triton, CPU→C++/OpenMP     │
   └───────────────┬─────────────────────────────┘
                   │ 编译后的 kernel
                   ▼
               执行

这是 torch.compile 的三大组件。第 2 章会展开它们的协作关系，第 3~5 章逐一深入每个组件，第 6 章聚焦 Inductor 的代码生成实现细节。

为什么要现在做编译器？
=========================

torch.compile 不是 PyTorch 第一次尝试编译器方案。事实上，从 PyTorch 1.0 到 2.x，编译器经历了三次重大的技术路线演变。理解这段历史，才能明白 torch.compile 为什么长成今天这个样子。

第一阶段：TorchScript（PyTorch 1.0, 2018）
--------------------------------------------------------

PyTorch 1.0 发布时，团队就敏锐地意识到：eager 模式灵活但性能不够，需要一个编译器方案。于是推出了 **TorchScript**——一种 Python 的静态子集。

TorchScript 提供了两种捕获方式：

.. code-block:: python

   # 方式一：trace —— 用示例输入执行，记录执行路径
   traced = torch.jit.trace(model, example_input)
   traced.save("model.pt")

   # 方式二：script —— 用 Python 子集重写模型
   @torch.jit.script
   def fn(x):
       if x.sum() > 0:    # TorchScript 支持这种控制流
           return x.sin()
       return x.cos()

``torch.jit.trace`` 的思路简单但致命：它只记录"这次执行走了哪条路"，不记录"有多少种可能的路径"。如果模型里有 ``if x.sum() > 0`` ，trace 只会捕获被执行的这条分支。下游换一个输入走了另一条分支，结果就不对了。

``torch.jit.script`` 试图解决这个问题：它解析 Python 源码（不是字节码），将其翻译为 TorchScript IR——一种受限的 Python 子集。但这套方案有更根本的问题：

- **必须用 TorchScript 语法**：很多 Python 特性（如 ``**kwargs`` 、 ``dataclass`` 、异常处理）不支持或行为不一致
- **语法解析高度脆弱**：Python 是一门极其动态的语言，源码解析无法处理运行时动态构造的函数
- **割裂的体验**：用户需要学会"TorchScript 能做什么、不能做什么"，更像是学了一门新语言

结果是社区反馈高度分化：简单模型（如 ResNet）体验不错，但凡涉及控制流、动态特征、第三方库的模型，经常出现"加上 TorchScript 反而跑不起来"的窘境。大量用户放弃使用，TorchScript 的采用率远低于团队预期。

第二阶段：FX Graph（PyTorch 1.9, 2021）
-----------------------------------------------

吸取了 TorchScript 的教训，团队开始寻找更灵活的方案。**FX Graph** （ ``torch.fx`` ）的思路是：不发明新语言，而是在 Python 内部用符号执行（symbolic tracing）构建计算图。

``torch.fx.symbolic_trace`` 通过"用代理张量替换真实张量"的方式，让模型在假的 Tensor 上执行一遍，同时记录所有操作调用：

.. code-block:: python

   import torch.fx as fx

   class MyModel(torch.nn.Module):
       def forward(self, x):
           return torch.sin(x) + torch.cos(x)

   traced = fx.symbolic_trace(MyModel())
   print(traced.graph)
   # graph():
   #     %x = placeholder
   #     %sin = call_function[target=torch.sin](args=(%x,))
   #     %cos = call_function[target=torch.cos](args=(%x,))
   #     %add = call_function[target=torch.add](args=(%sin, %cos))
   #     return add

注意：FX Graph 仍然受制于动态控制流。如果模型里有 ``if x.sum() > 0`` ，symbolic trace 同样只执行一次，只捕获到被触发的分支。但它提供了一个重要的新能力——**图是可编程、可变换的** （我们在第 2 章会详细演示）。这意味着你可以在捕获后对图做任意操作：插入节点、删除节点、融合操作。

FX Graph 做了两件重要的事情：

1. 给出了一个 **语言无关的中间表示**——图中的每个节点只描述"做什么"，不依赖于 Python 语法
2. 提供了 **程序化图变换** 的 API——后续所有的优化 pass 都可以在这张图上操作

不过 FX Graph 自身不是编译器。它只是"图"，至于"怎么把图变成高效的代码"，留给了下游处理。

第三阶段：torch.compile（PyTorch 2.0, 2022）
----------------------------------------------------

2022 年 PyTorch 2.0 发布时，团队在 TorchScript 和 FX Graph 的基础上，做出了几个关键的技术抉择。

**抉择一：在字节码级别捕获，而不是源码级别**

TorchScript 用 Python 源码解析（AST），但 AST 看不到运行时信息——不知道变量的实际类型、不知道 ``x.sin()`` 是调用了哪个具体函数。

TorchDynamo 换了一条路：通过 CPython 的 `PEP 523 <https://peps.python.org/pep-0523/>`__ 框架，在 **字节码** 级别拦截帧（frame）执行。当 Python 解释器开始执行一个函数时，Dynamo 会观察它生成的每一条字节码指令，识别出所有涉及 Tensor 的操作。

.. code-block:: text

   字节码（CPython 3.10）：
       LOAD_FAST   x
       LOAD_ATTR   sin
       CALL_FUNCTION
       ...

Dynamo 的做法好在哪里？

- **不需要源码**：即使函数是通过 ``exec()`` 或 ``eval()`` 动态创建的，Dynamo 也能捕获
- **天然兼容 Python**：字节码层面的操作不依赖于 Python 语法子集，遇到不能处理的操作（如 ``print()`` ）只需 graph break，不会崩溃
- **精度高**：字节码指令中能准确区分变量是 Tensor 还是普通 Python 对象

**抉择二：编译器来适应 Python，而不是让 Python 去适应编译器**

这是 torch.compile 最核心的设计哲学差异。TorchScript 的思路是"你（用户）给我一份我能编译的代码"，torch.compile 的思路是"你（用户）随便写 Python，我来处理剩下的"。

实现这一点的关键是：**graph break 不是错误，而是设计特性** 。

当 Dynamo 遇到无法捕获的操作时（比如调用外部 C 库、使用 Python 原生容器），它不会报错退出，而是优雅地在这一点切断图——之前捕获的部分形成一个子图并编译，之后的操作留给 eager 执行。用户看到的是结果正确的输出，只是可能性能不如完整图捕获那么高。

.. code-block:: text

   用户代码中有一段：
       x = torch.sin(x)          ← caught
       print(x)                  ← graph break
       x = torch.cos(x)          ← new subgraph

   Dynamo 编译为两个子图：
       Subgraph 1: x → sin(x)
       [Python: print(x)]
       Subgraph 2: x → cos(x)

**抉择三：三阶段流水线替代单阶段编译**

.. note::

   **TorchDynamo 和 Inductor 曾经是独立的 GitHub 仓库。**
   2022 年 9 月之前，TorchDynamo 在独立的 ``pytorch/torchdynamo`` 仓库中开发，与 PyTorch 主仓库完全分离。Inductor（最初叫 TorchInductor）也以独立库的形式存在。2022 年 9 月，它们通过合并请求 ``#86461`` （标题 "Move TorchDynamo into PyTorch core"）同时被搬入 PyTorch 主仓库。入仓后，团队仍然保持了一段时间的"双仓库同步"模式——在独立仓库开发，然后通过 ``Sync changes from pytorch/torchdynamo`` 的提交同步到主仓库。这种模式直到 2022 年底才结束。此后所有开发直接在 PyTorch monorepo 中进行。

TorchScript 试图在一个阶段内完成"图捕获 + 代码生成"。torch.compile 将这个过程拆分为三个松耦合的阶段，每个阶段专精于一个任务：

.. list-table::
   :header-rows: 1

   * - 组件
     - 职责
     - 输入
     - 输出
   * - TorchDynamo
     - 图捕获
     - Python 函数
     - FX Graph
   * - AOTAutograd
     - 自动微分 + 图分区
     - FX Graph
     - 分片后的 FX Graph
   * - Inductor
     - 代码生成
     - FX Graph
     - Triton / C++ 代码

这种分层设计的好处是每个组件可以被独立替换和演进：

- 你可以用 ``torch.compile(fn, backend="eager")`` 跳过 Inductor，只验证图捕获是否正确
- 你可以注册自己的后端替换 Inductor，用同一个图捕获管道

**抉择四：选择 Inductor 作为默认后端，而非对接已有编译器**

当时社区有几个可供选择的"图 → 代码"方案：

- **XLA** （TensorFlow 的编译器）：成熟但对 PyTorch 生态适配不足
- **NVIDIA TensorRT** ：推理优化很强但不支持 PyTorch 训练
- **TVM**(Apache)：灵活的端到端编译栈，但社区整合成本高
- **LLVM** ：底层编译器基础设施，离 PyTorch 太远

PyTorch 团队最终决定自研 **Inductor** ，原因如下：

1.**端到端控制** ：从 FX Graph 到最终代码的整条链路都在 PyTorch 生态内，排查问题和迭代优化的周期最短
2.**Triton 语言** ：Inductor 的 GPU 后端生成 Triton 代码。Triton 提供比 CUDA 更高的抽象层级，让 Inductor 不必直接编写和维护复杂 CUDA kernel。CPU 后端则生成 C++/OpenMP 代码
3.**Define-by-run IR** ：Inductor 内部使用循环级 IR（IRNode），这种 IR 保留了 PyTorch 的 "define-by-run" 哲学——IR 节点本身就是像 PyTorch 操作一样被逐级构建的

不影响这个决策的是：用户可以在 ``mode`` 参数中指定不同的后端。Inductor 是 **默认** 的，但不是 **唯一** 的。

三种设计哲学的根本差异
============================================

.. list-table::
   :header-rows: 1

   * -
     - TorchScript
     - FX Graph
     - torch.compile
   * - 捕获方式
     - 源码解析 (AST)
     - 符号执行
     - 字节码钩子 (PEP 523)
   * - 对用户的要求
     - 学习 TorchScript 子集
     - 无（但不支持动态控制流）
     - 无（graph break 兜底）
   * - 动态控制流
     - 部分支持
     - 不支持
     - 支持（graph break）
   * - 可变换性
     - 低
     - 高（programmatic）
     - 高（基于 FX）
   * - 后端可替换
     - 否
     - 是（FX Graph 可消费）
     - 是（ ``backend`` 参数）
   * - 当前状态
     - 维护模式
     - 基础设施
     - 主线

为什么是现在？
==================

一个问题值得问：既然编译器是深度学习框架的必备能力，为什么 PyTorch 到了 2.x 才算真正意义上做成了？

原因有三：

**第一，基础设施成熟了** 。PEP 523（2016 年合入 CPython 3.6）提供了字节码级别的帧回调机制，但没有好用的 Python 字节码分析工具。Dynamo 团队花了不少精力构建字节码分析器（见 ``torch/_dynamo/bytecode_analysis.py`` ），这是 TorchScript 时代做不到的。

**第二，Triton 语言出现了** 。Triton（由 OpenAI 的 Philippe Tillet 开发）从根本上降低了 GPU 代码生成的门槛。Inductor 的 GPU 后端生成 Triton 代码而不是直接生成 PTX/SASS，大大降低了开发和维护成本。

**第三，社区需求爆发了** 。2021-2022 年，模型规模的增长远超单 GPU 算力增长。用户对性能的需求从"nice to have"变成了"must have"。同时大量用户在 TorchScript 上碰壁后积累了"恨意"，社区对可用性更高的编译器方案有强烈的呼声。

小结
======

这一节从一个性能问题出发，引出了 torch.compile 的核心思路——**先捕获计算图，再编译执行** 。我们还回顾了 PyTorch 编译器的发展历程：

- **TorchScript** （静态子集）：用户体验割裂，采用率低
- **FX Graph** （符号执行）：提供了可编程图表示，但仍是 capture-only
- **torch.compile** （字节码钩子）：Dynamo 负责图捕获、AOTAutograd 负责微分、Inductor 负责代码生成

torch.compile 的设计确保了三点： **用户代码不需要修改** 、 **不支持的操作有 graph break 兜底** 、 **三阶段松耦合可独立演进** 。

下一节我们用 Hello World 跑起来第一个编译示例。

.. note::

   本节提到的"300 次 kernel launch"是一个估算。实际开销取决于张量大小：小张量时 kernel launch 延迟是瓶颈，大张量时计算本身占据主导。我们会在 1.5 节用实际基准测试来验证这一点。


