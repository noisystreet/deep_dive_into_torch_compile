.. _logging-system:

============
日志系统
============

torch.compile 提供了全面且可配置的日志系统，用于追踪编译流水线的每个环节。理解日志系统是高效调试的第一步。

TORCH_LOGS 环境变量
========================

日志系统的入口是 ``TORCH_LOGS`` 环境变量。它控制哪些模块输出日志，以及输出到什么级别：

.. code-block:: bash

   # 基本用法
   TORCH_LOGS="+dynamo" python train.py        # Dynamo 模块的日志
   TORCH_LOGS="+inductor" python train.py       # Inductor 模块的日志
   TORCH_LOGS="+aot" python train.py            # AOTAutograd 模块的日志

   # 组合多个模块
   TORCH_LOGS="+dynamo,+inductor,+aot" python train.py

   # 完整日志（包含所有模块）
   TORCH_LOGS="+all" python train.py

``TORCH_LOGS`` 的格式为 ``+module_name`` （启用）或 ``-module_name`` （禁用）。多个模块用逗号分隔。

可用的日志模块
====================

.. list-table::
   :header-rows: 1

   * - 模块名
     - 对应源码路径
     - 输出内容
   * - ``dynamo``
     - ``torch/_dynamo/``
     - Graph break 位置、guard 生成、字节码分析、编译缓存
   * - ``aot``
     - ``torch/_functorch/_aot_autograd/``
     - 联合图追踪、图分区、functionalization
   * - ``inductor``
     - ``torch/_inductor/``
     - Lowering 过程、代码生成、编译参数
   * - ``schedule``
     - ``torch/_inductor/scheduler.py``
     - 融合决策、节点依赖分析
   * - ``perf_hints``
     - -
     - 性能提示（如 graph break 导致性能下降）
   * - ``output_code``
     - -
     - 生成的 Triton 或 C++ 代码
   * - ``guards``
     - ``torch/_dynamo/guards.py``
     - Guard 表达式的详细信息

日志输出示例
==================

**Dynamo 日志**：显示 graph break 的位置：

.. code-block:: text

   [dynamo] 图捕获开始...
   [dynamo] Graph break at: call_function print (example.py:10)
   [dynamo] 生成子图 #0 (3 个节点)
   [dynamo] 生成子图 #1 (5 个节点)

**Inductor 日志**：显示 lowering 和融合过程：

.. code-block:: text

   [inductor] lowering aten.sin (fx_node: %sin)
   [inductor] lowering aten.cos (fx_node: %cos)
   [inductor] Pointwise + Pointwise 融合成功

**输出代码日志**：显示生成的 Triton kernel：

.. code-block:: bash

   TORCH_LOGS="+output_code" python -c "
   import torch
   @torch.compile
   def fn(x): return torch.sin(x) + torch.cos(x)
   fn(torch.randn(3))
   "

日志中会看到类似以下的 Triton 代码输出：

.. code-block:: python

   @triton.jit
   def triton_poi_fused_add_cos_sin_0(
       x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
   ):
       ...

日志级别
============

``TORCH_LOGS`` 支持日志级别控制：

.. code-block:: bash

   TORCH_LOGS="+dynamo"                # 默认 INFO 级别
   TORCH_LOGS="dynamo:DEBUG"           # DEBUG 级别（更详细）
   TORCH_LOGS="dynamo:WARNING"         # 仅显示警告

多个模块可以设置不同级别：

.. code-block:: bash

   TORCH_LOGS="dynamo:DEBUG,inductor:INFO"

将日志保存到文件：

.. code-block:: bash

   TORCH_LOGS="+dynamo" python train.py 2> dynamo_log.txt

TorchDynamo 日志系统的实现
=================================

日志系统基于 Python 标准库的 ``logging`` 模块，但在其上做了一层封装以支持按模块过滤。相关的实现在 ``torch/_logging/`` 目录中：

.. code-block:: text

   torch/_logging/
   ├── __init__.py          # 初始化日志系统
   ├── _registrations.py    # 注册日志模块名到 logger 的映射
   └── _internal.py         # 日志格式化和过滤逻辑

``TORCH_LOGS`` 的值在 ``_registrations.py`` 中解析，将模块名映射到对应的 ``logging.getLogger(name)``，然后设置对应 logger 的级别。

对于 Inductor，日志通过 ``getArtifactLogger`` 获取：

.. code-block:: python

   # torch/_inductor/scheduler.py
   schedule_log = torch._logging.getArtifactLogger(
       __name__, "schedule"
   )
   schedule_log.debug("考虑融合 A 和 B")

这种机制允许同一个文件中的不同 logger 独立控制——例如一个文件可以同时输出 ``inductor`` 和 ``schedule`` 两种日志。
