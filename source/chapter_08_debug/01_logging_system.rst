.. _logging-system:

============
日志系统
============

.. tip::

   **"TORCH_LOGS=+all" 的输出量有多大？**
   对于 ResNet50 的一次前向传播，``TORCH_LOGS=+all`` 可以产生超过 10 万行日志。这意味着编译全流程中会触发大量细粒度的调试信息。实践中真正需要的是**精细化的日志控制**——例如只想看 Inductor 的融合决策，就写 ``TORCH_LOGS=+schedule,+inductor``，而不是打开所有模块。这也是日志系统设计为模块化的原因：你可以像搭积木一样自由组合需要的日志模块。

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

日志系统的整体架构可以用以下流程图表示：

.. mermaid::

   graph TD
       ENV["TORCH_LOGS 环境变量<br/>+dynamo:DEBUG,+inductor"] --> REG["_registrations.py<br/>解析模块名与日志级别"]
       REG --> LOGGER["logging.getLogger(name)<br/>为每个模块创建 Logger 实例"]
       LOGGER --> MODULE_LOGGER["模块级 Logger<br/>dynamo / inductor / aot / schedule"]
       MODULE_LOGGER --> OUTPUT["日志输出<br/>标准错误 / 文件"]
       MODULE_LOGGER --> ARTIFACT["getArtifactLogger()<br/>子模块级细粒度日志控制"]
       ARTIFACT --> OUTPUT

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

``getArtifactLogger`` 在模块内的工作流程如下：

.. mermaid::

   sequenceDiagram
       participant Module as 模块代码<br/>(scheduler.py)
       participant AL as getArtifactLogger<br/>(_internal.py)
       participant Logger as logging.Logger
       participant Handler as 日志处理器

       Module->>AL: getArtifactLogger(__name__, "schedule")
       AL->>AL: 查找或创建 Logger 名称<br/>"torch._inductor.scheduler.schedule"
       AL->>Logger: 获取 Logger 实例
       Logger->>Handler: 判断日志级别<br/>决定是否输出
       Handler-->>Module: 输出到 stderr

日志系统的架构设计
===============================

``torch._logging`` 包结构
--------------------------------

日志系统的核心实现在 ``torch/_logging/`` 目录中，包结构如下：

.. code-block:: text

   torch/_logging/
   ├── __init__.py          # 初始化日志系统, 定义 getArtifactLogger
   ├── _registrations.py    # 注册模块名到 Logger 的映射, 解析 TORCH_LOGS
   ├── _internal.py         # 日志格式化、ArtifactLogger 实现、过滤逻辑
   └── _handlers.py         # 自定义日志处理器

- ``__init__.py``：导出 ``getArtifactLogger`` 和 ``set_logs`` 等公共 API
- ``_registrations.py``：维护模块名与 Logger 名的对应关系，解析 ``TORCH_LOGS`` 环境变量的值
- ``_internal.py``：实现 ``ArtifactLogger`` 类，继承自 ``logging.Logger``，添加了 artifact 过滤能力
- ``_handlers.py``：提供将日志输出到文件的自定义处理器

getArtifactLogger 与标准 Python logging 的区别
--------------------------------------------------

标准 Python logging 通过 ``logging.getLogger(name)`` 获取 Logger，其中 ``name`` 通常是模块的 ``__name__``。Logger 的层级关系由名称中的点号分隔决定——例如 ``torch._inductor.scheduler`` 是 ``torch._inductor`` 的子 Logger。

``getArtifactLogger`` 在这之上添加了一个 **artifact 维度**：

.. code-block:: python

   # 标准 Python logging
   log = logging.getLogger("torch._inductor.scheduler")
   log.debug("这是一条普通日志")  # 受全局日志级别控制

   # getArtifactLogger —— 增加 artifact 标签
   schedule_log = torch._logging.getArtifactLogger(
       __name__, "schedule"
   )
   schedule_log.debug("这是一条 schedule 日志")  # 受 "schedule" artifact 控制

区别在于：

- 标准 Logger 的级别由 Logger 名称的层级决定，父子 Logger 之间通过 propagate 传递日志
- ``getArtifactLogger`` 创建的 Logger 额外绑定了一个 **artifact 名称**，这个名称独立于模块层级
- 同一个模块可以有多个 artifact Logger，各自有独立的开关控制

日志级别的层级传播
----------------------

``TORCH_LOGS`` 设置的日志级别会沿着模块层级传播：

.. code-block:: bash

   # 设置 dynamo 为 DEBUG 级别
   TORCH_LOGS="dynamo:DEBUG"

   # 这会影响所有以 torch._dynamo 开头的子模块
   # 包括 torch._dynamo.guards, torch._dynamo.bytecode_transformation 等

传播规则如下：

.. mermaid::

   graph TD
       ROOT["TORCH_LOGS='dynamo:DEBUG'"] --> PARENT["torch._dynamo Logger<br/>级别=DEBUG"]
       PARENT --> CHILD1["torch._dynamo.guards<br/>继承 DEBUG 级别"]
       PARENT --> CHILD2["torch._dynamo.bytecode_transformation<br/>继承 DEBUG 级别"]
       CHILD1 --> OUTPUT1["输出 guards 相关日志"]
       CHILD2 --> OUTPUT2["输出字节码转换日志"]

这种层级传播机制使得设置一个父模块的日志级别，就能控制其下所有子模块的日志输出，而无需逐一设置。

配置日志系统的实用技巧
-------------------------------

除了通过 ``TORCH_LOGS`` 环境变量，还可以在代码中**程序化地配置日志系统**：

.. code-block:: python

   import torch._logging

   # 在代码中启用指定模块的日志
   torch._logging.set_logs(
       dynamo=logging.DEBUG,
       inductor=logging.INFO,
       schedule=True,          # True 等价于 INFO 级别
   )

   # 也可以按字符串名称配置
   torch._logging.set_logs(
       "torch._dynamo"=logging.DEBUG,
       aot=logging.WARNING
   )

.. note::

   ``torch._logging.set_logs`` 接受 ``logging.DEBUG`` 这样的标准日志级别常量，
   也接受 ``True`` / ``False`` 这样的布尔值——``True`` 等价于 ``INFO`` 级别，``False`` 等价于 ``WARNING`` 级别。

程序化配置的主要优势：

- 可以在同一个脚本的不同阶段切换日志级别
- 不需要重启进程或修改环境变量
- 可以结合条件逻辑（如只在第一个 batch 后启用详细日志）
- 适合集成到自定义的测试框架中

.. seealso::

   完整的 ``set_logs`` API 文档见 ``torch/_logging/__init__.py``。
   支持的参数列表对应 ``_registrations.py`` 中注册的所有 artifact 名称。

小结
========

本章详细介绍了 torch.compile 的日志系统。核心要点包括：

- ``TORCH_LOGS`` 环境变量是日志系统的入口，支持按模块细粒度控制
- 日志系统基于 Python 标准 ``logging`` 模块，通过 ``getArtifactLogger`` 增加了 artifact 维度
- 日志级别沿模块层级自动传播，设置父模块即可控制所有子模块
- 除了环境变量，还可以通过 ``torch._logging.set_logs`` 在代码中动态配置
- 理解日志系统是高效调试 torch.compile 问题的第一步

下一节将介绍 Minimizer 工具，它利用二分搜索自动定位编译错误的根本原因。
