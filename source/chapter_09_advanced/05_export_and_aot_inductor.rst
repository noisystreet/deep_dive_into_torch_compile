.. _export-and-aot-inductor:

======================================
Export 与 AOTInductor 离线部署
======================================

前几章我们追踪的主要是 **在线 JIT 编译** 路径：每次 ``torch.compile(model)(x)`` 调用时，Dynamo 捕获图、Inductor 生成 kernel，编译结果缓存在内存或磁盘中供后续命中。这条路径适合训练和在 Python 进程内做推理服务。

但在生产部署中，常见需求是：**在部署前一次性编译好所有 kernel，在 C++ 运行时直接加载 ``.so``，完全消除 Python 端的编译延迟**。PyTorch 为此提供了 ``torch.export`` 和 **AOTInductor** 两条互补的 API。这一节梳理它们与 ``torch.compile`` 的关系，以及典型的离线部署流程。

torch.compile vs torch.export
=================================

两者解决不同层面的问题：

.. list-table::
   :header-rows: 1

   * - 维度
     - ``torch.compile``
     - ``torch.export``
   * - 主要目标
     - 运行时加速（JIT）
     - 图提取与序列化（AOT）
   * - 输出
     - 可执行的 Python callable
     - ``ExportedProgram`` （带约束的 FX Graph）
   * - 动态性
     - 依赖 guard + 重编译
     - 显式声明 ``Dim`` 约束
   * - 典型场景
     - 训练、Python 推理服务
     - 跨语言部署、编译器后端输入

``torch.export`` 不会直接生成 Triton kernel，它的产物是一张带形状约束的 FX Graph（``ExportedProgram``）。你可以把它交给 Inductor、TensorRT、XLA 等后端做进一步编译。

最小 export 示例
====================

.. code-block:: python

   import torch

   class M(torch.nn.Module):
       def forward(self, x):
           return torch.relu(x @ self.weight.T)

   model = M()
   model.weight = torch.nn.Parameter(torch.randn(10, 20))

   example_inputs = (torch.randn(4, 20),)

   # 导出为 ExportedProgram
   exported = torch.export.export(model, example_inputs)
   print(exported.graph_module)

``export`` 会在 trace 阶段记录输入张量的形状，并生成对应的 guard 约束。如果某个维度需要支持变化，必须显式声明：

.. code-block:: python

   batch = torch.export.Dim("batch", min=1, max=1024)
   dynamic_shapes = {"x": {0: batch}}

   exported = torch.export.export(
       model,
       example_inputs,
       dynamic_shapes=dynamic_shapes,
   )

这里的 ``Dim`` 与第 3.8 节讨论的符号形状（symbolic shapes）共享同一套 ``ShapeEnv`` 基础设施——``export`` 在编译期就固定了哪些维度是符号化的，而不是像 ``torch.compile`` 那样在运行时通过 guard 失败来发现新形状。

AOTInductor：Inductor 的离线变体
=====================================

**AOTInductor** （Ahead-of-Time Inductor）在 ``ExportedProgram`` 或 ``torch.compile`` 捕获的图上运行完整的 Inductor 流水线，但把产物写成 **可独立加载的共享库**，而不是留在 Python 进程内。

.. code-block:: text

   在线路径（torch.compile）:
       Python 调用 → Dynamo 捕获 → Inductor 编译 → 内存/磁盘缓存 → 同进程执行

   离线路径（AOTInductor）:
       export / aot_compile → Inductor 编译 → 生成 .so + .cpp wrapper
       → C++ 运行时加载 .so → 无 Python 编译开销

AOTInductor 的典型输出包括：

- ``model.so``：编译好的 kernel 与调度逻辑
- ``model.cpp`` / 头文件：C++ 调用接口
- 常量权重文件（若模型含 ``nn.Parameter``）

使用 ``torch._export.aot_compile`` 导出
==========================================

PyTorch 提供了 ``aot_compile`` API，在 export 的同时触发 AOTInductor 编译：

.. code-block:: python

   import torch

   class M(torch.nn.Module):
       def forward(self, x):
           return x.sin() + x.cos()

   model = M()
   example_inputs = (torch.randn(8, 16),)

   # 指定输出目录，Inductor 在此生成 .so 等产物
   so_path = torch._export.aot_compile(
       model,
       example_inputs,
       options={
           "aot_inductor.output_path": "/tmp/aot_model",
       },
   )
   print(so_path)

等价的配置方式是通过 ``torch._inductor.config``：

.. code-block:: python

   import torch._inductor.config as inductor_config

   inductor_config.aot_inductor.output_path = "/tmp/aot_model"

编译完成后，C++ 侧通过 AOTInductor 生成的 runner 加载并执行。具体 API 随 PyTorch 版本演进，核心思路不变：**Python 负责编译一次，C++ 负责反复执行**。

与 torch.compile 共享的编译栈
=================================

AOTInductor 并非独立的编译器，它复用本书前面章节介绍的完整流水线：

.. code-block:: text

   ExportedProgram / FX Graph
       │
       ├─ pre_grad_passes          ← 第 5.2 节
       ├─ aot_autograd + decomp    ← 第 4 章
       ├─ post_grad_passes        ← 第 5.2 节
       ├─ Lowering → Scheduler    ← 第 5 章
       ├─ Codegen（Triton / C++）  ← 第 6 章
       └─ 写入 .so（而非返回 Python callable）

因此，本书关于 Dynamo guard、符号形状、Inductor fusion 的知识同样适用于 AOTInductor 场景——区别主要在 **产物形态** 和 **形状约束的声明方式**。

CppWrapper 与 Python Wrapper
================================

第 6 章介绍了 Inductor 的两种 wrapper 生成方式：

- ``PythonWrapperCodegen``：生成 Python 代码调用 Triton kernel（``torch.compile`` 默认）
- ``CppWrapperCpu`` / ``CppWrapperGpu``：生成 C++ wrapper，减少 Python 解释器开销

AOTInductor 路径几乎总是使用 **C++ wrapper**，因为部署目标就是脱离 Python 运行时。``codegen/cpp_wrapper_gpu.py`` 和 ``codegen/cpp_wrapper_cpu.py`` 负责生成可直接链接的 C++ 调用代码。

何时选择哪条路径
====================

.. list-table::
   :header-rows: 1

   * - 场景
     - 推荐路径
   * - 模型训练、快速迭代
     - ``torch.compile`` （在线 JIT）
   * - Python 推理服务（Triton Server、FastAPI）
     - ``torch.compile`` + 磁盘缓存（``TORCHINDUCTOR_CACHE_DIR``）
   * - C++ 推理引擎、边缘部署
     - ``torch.export`` + AOTInductor
   * - 需要跨框架交换计算图
     - ``torch.export`` → 第三方后端（TensorRT 等）
   * - 输入形状高度固定、追求最低延迟
     - AOTInductor + 静态 export
   * - 输入 batch / 序列长度变化
     - export 时声明 ``Dim`` + AOTInductor（或在线 ``dynamic=True``）

常见限制
============

AOTInductor 当前仍有一些约束，部署前需要验证：

- **控制流**：``cond``、``while_loop`` 等需要 export 支持的控制流算子，覆盖范围在持续扩展
- **自定义算子**：须通过 ``torch.library`` 注册并提供 meta kernel（见第 9.4 节）
- **动态形状**：必须在 export 阶段用 ``Dim`` 声明，不能像 ``torch.compile`` 那样依赖运行时 guard 自动发现
- **权重更新**：离线 `.so` 中的常量需与训练 checkpoint 版本匹配；热更新权重需要额外的加载机制

小结
======

- **``torch.compile``** 是在线 JIT 路径，适合 Python 生态内的训练与推理
- **``torch.export``** 提取带约束的 ``ExportedProgram``，是跨后端、跨语言部署的图交换格式
- **AOTInductor** 在 export 图上运行 Inductor 完整流水线，输出 C++ 可加载的 `.so`
- 三条路径共享 Dynamo → AOTAutograd → Inductor 核心编译栈，差异在于产物形态与形状约束的声明时机

下一节展望 torch.compile 社区正在推进的改进方向。
