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
