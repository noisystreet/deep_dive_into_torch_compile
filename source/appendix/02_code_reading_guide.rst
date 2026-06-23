.. _appendix-code-reading:

=============================
附录 B  关键代码阅读指南
=============================

本文档帮助读者快速定位 PyTorch 源码中与 ``torch.compile`` 相关的关键代码位置。

.. list-table::
   :header-rows: 1

   * - 模块
     - 源码路径
     - 核心文件
   * - TorchDynamo
     - ``torch/_dynamo/``
     - ``convert_frame.py``, ``bytecode_analysis.py``, ``guards.py``
   * - AOTAutograd
     - ``torch/_functorch/``
     - ``aot_autograd.py``, ``partitioners.py``, ``functionalize.py``
   * - Inductor
     - ``torch/_inductor/``
     - ``graph.py``, ``ir.py``, ``scheduler.py``, ``pattern_matcher.py``, ``codegen/``
   * - FX Graph
     - ``torch/fx/``
     - ``graph.py``, ``symbolic_trace.py``, ``passes/``
   * - 代码生成（CPU）
     - ``torch/_inductor/codegen/``
     - ``cpp.py``, ``cpp_wrapper.py``, ``triton.py``
   * - 代码生成（GPU）
     - ``torch/_inductor/codegen/``
     - ``triton.py``, ``triton_utils.py``, ``kernel_benchmark.py``
   * - Dynamic Shapes
     - ``torch/fx/experimental/``
     - ``symbolic_shapes.py``, ``sym_node.py``
