.. _appendix-glossary:

==================
附录 C  术语表
==================

.. glossary::

   TorchDynamo
      PyTorch 2.x 引入的 JIT 编译器前端，通过 CPython 的 PEP 523 框架在 Python
      字节码级别捕获计算图。具有动态性、与现有代码兼容性好的特点。

   AOTAutograd
      Ahead-of-Time 自动微分模块。它在前向传播的同时追踪反向传播的计算图，
      "提前" 生成前向和反向的联合计算图，然后进行图分区。

   Inductor
      PyTorch 2.x 的默认编译器后端。它将计算图降级为循环级 IR，然后为 GPU
      生成 Triton 代码，为 CPU 生成 C++/OpenMP 代码。

   FX Graph
      PyTorch 的计算图中间表示（Intermediate Representation），是一个
      Python 层面的可操作图结构，由 ``torch.fx`` 模块提供。

   Graph Break
      图断裂。当 ``torch.compile`` 遇到无法捕获的 Python 操作时，会在该
      处打断图的捕获，形成多个子图的边界。

   Guard
      TorchDynamo 用于验证缓存编译结果是否仍然有效的运行时检查机制。
      当 guard 检查失败时，会触发重新编译。

   IRNode
      Inductor 内部的循环级中间表示，位于计算图（FX Graph）和最终代码
      之间的抽象层。

   Scheduler
      Inductor 中的调度器，负责将 IRNode 分组、融合，并决定执行顺序。

   Triton
      一种面向 GPU 编程的语言和编译器，提供比 CUDA 更高的抽象层级，
      Inductor 的 GPU 后端默认使用 Triton 作为代码生成目标。

   Functionalization
      将 PyTorch 的原位（in-place）操作转换为纯函数式操作的过程，
      是 AOTAutograd 的重要步骤。

   Min-Cut Recompuation
      AOTAutograd 使用的内存优化策略，通过在反向传播中重新计算某些
      前向中间结果来减少内存占用。

   Joint Graph
      AOTAutograd 生成的前向和反向联合计算图，包含完整的求导信息。

   Symbolic Shapes
      符号形状机制。用 ``SymNode`` / ``SymInt`` 代替具体维度数值，使一份编译
      结果适配多种输入尺寸。由 ``ShapeEnv`` （ ``symbolic_shapes.py`` ）管理。

   ShapeEnv
      PyTorch 中管理符号形状的核心组件，负责创建符号变量、记录约束、
      生成 guard 表达式。位于 ``torch/fx/experimental/symbolic_shapes.py`` 。

   ExportedProgram
      ``torch.export`` 的输出，包含带形状约束的 FX Graph 及元数据，
      可作为 Inductor、TensorRT 等后端的输入。

   AOTInductor
      Inductor 的离线（Ahead-of-Time）变体，在部署前编译所有 kernel，
      输出 C++ 可加载的共享库（ ``.so`` ），消除运行时编译延迟。

   Lowering（降级）
      将高层中间表示（FX Graph）转换为低层中间表示（Inductor IRNode）的过程。
      在 Inductor 中，每个 ATen 算子都有对应的 lowering 函数，将 FX 节点
      映射为 Pointwise、Reduction、TemplateBuffer 等 IR 类型。

   Pointwise（逐元素操作）
      对张量中每个元素独立应用相同计算的操作类型（如 ``sin``、``add``、
      ``mul``），是 Inductor 中最常见、最易融合的 IR 类型。

   Reduction（归约操作）
      将张量沿某个维度聚合为更少元素的操作类型（如 ``sum``、``mean``、
      ``max``），在 Inductor 中以 ``Reduction`` IRNode 表示。

   Decomposition（算子分解）
      将高层算子（如 ``layer_norm``、``softmax``）展开为基本算子
      （``mean``、``rsqrt``、``mul`` 等）的过程，由 AOTAutograd 在
      joint graph 构建时执行，降低后端的 lowering 负担。

   Dynamic Shapes（动态形状）
      编译时形状未知、运行时可能变化的张量维度。Inductor 使用符号形状
      （Symbolic Shapes）机制和 ``sympy`` 表达式处理动态形状，生成
      通用的 Triton kernel 而非为每个形状编译专用版本。

   Fusion（融合）
      将多个连续的操作合并为一个 kernel 执行，减少 kernel launch 开销和
      中间结果的显存读写。Inductor 的 Scheduler 负责在 IRNode 层面
      执行融合决策。
