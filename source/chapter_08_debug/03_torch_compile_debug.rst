.. _torch-compile-debug:

====================
torch.compile debug
====================

PyTorch 提供了 ``torch.compiler.debug`` 上下文管理器，用于生成**结构化的编译调试报告**。它比原始的日志系统更易读，以 HTML 格式呈现完整的编译过程。

基本用法
============

.. code-block:: python

   import torch
   
   @torch.compile
   def fn(x):
       return (torch.sin(x) + torch.cos(x)).sum()

   x = torch.randn(10)
   
   with torch.compiler.debug():
       result = fn(x)

运行这段代码后，会在当前目录生成一个 ``torch_compile_debug`` 目录，里面包含 HTML 格式的调试报告。

调试报告的目录结构
=========================

.. code-block:: text

   torch_compile_debug/
   └── run_2025-01-01_00-00-00-xxxxxx/
       ├── torchdynamo_debug.html       # 主报告（Dynamo 视角）
       ├── inductor.html                # Inductor 报告
       ├── fx_graph_readable.txt        # 可读的 FX Graph
       ├── fx_graph_runnable.py         # 可运行的 FX Graph
       └── replay.py                    # 编译过程回放脚本

报告的主要部分
===================

**Dynamo 报告** （``torchdynamo_debug.html``）包含了：

1. **Graph Break 摘要**：列出所有触发 graph break 的位置和原因
2. **Guard 列表**：显示生成的所有 guard 表达式
3. **子图列表**：每个子图的 FX Graph 展示
4. **编译统计**：编译时间、节点数、参数大小

**Inductor 报告** （``inductor.html``）包含了：

1. **Lowering 记录**：每个 FX 节点如何降级为 IRNode
2. **融合结果**：哪些 IRNode 被融合在一起
3. **生成的 kernel 列表**：每个 kernel 的 Triton 或 C++ 源代码
4. **性能估算**：每个 kernel 的估算 FLOPs 和内存带宽

读取 FX Graph
==================

调试报告中的 ``fx_graph_readable.txt`` 文件可以快速查看计算图结构：

.. code-block:: text

   // fx_graph_readable.txt 示例
   opcode       name     target              args               kwargs
   --------    ------   --------            ------              ------
   placeholder x        x                   ()                  {}
   call_function sin_1  aten.sin            (x,)                {}
   call_function cos_1  aten.cos            (x,)                {}
   call_function add_1  aten.add            (sin_1, cos_1)      {}
   call_function sum_1  aten.sum            (add_1,)            {}
   output       output  output              (sum_1,)            {}

``fx_graph_runnable.py`` 是一个可以直接运行的 Python 脚本，用于测试图的正确性。

利用调试报告分析 Graph Break
====================================

当模型因 graph break 导致性能下降时，调试报告可以帮助定位原因：

1. 打开 ``torchdynamo_debug.html``
2. 查看 "Graph Break" 部分
3. 每个 graph break 都会显示：
   - 触发的 Python 代码位置（文件名 + 行号）
   - 触发的原因（如 ``Unsupported: call_function print``）
   - 前后子图的分界

使用 ``torch.compiler.reset()``
=====================================

编译缓存可能导致调试信息不完整。在调试前重置缓存：

.. code-block:: python

   torch.compiler.reset()
   
   with torch.compiler.debug():
       result = fn(x)

这样确保每次调试都是一次全新的编译。

``compile_context`` 的更多选项
========================================

除了 ``torch.compiler.debug()``，还可以使用 ``torch._dynamo.config`` 获取更细粒度的控制：

.. code-block:: python

   import torch._dynamo.config as config
   
   # 保存每次编译的 FX Graph
   config.debug_dir_root = "/tmp/torch_debug"
   
   # 写入每个编译步骤的中间状态
   config.replay_record_enabled = True
   config.record_guard_failure = True

这些配置在 ``torch/_dynamo/config.py`` 中定义，提供了比 ``TORCH_LOGS`` 更细粒度的控制。
