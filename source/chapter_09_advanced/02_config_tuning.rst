.. _config-tuning:

==================
配置调优
==================

torch.compile 和 Inductor 提供了大量配置参数，用于控制编译行为、优化策略和调试输出。理解这些配置可以帮助你在不同场景下获得最佳性能。

配置层次结构
================

torch.compile 的配置系统分为三层：

.. code-block:: text

   第一层: torch.* API（用户直接接触）
       torch.compile(mode="default", fullgraph=False, dynamic=False)
       torch.set_float32_matmul_precision("high")

   第二层: torch._dynamo.config（Dynamo 行为）
       torch._dynamo.config.cache_size_limit = 64
       torch._dynamo.config.assume_static_by_default = True

   第三层: torch._inductor.config（Inductor 行为）
       torch._inductor.config.max_autotune = True
       torch._inductor.config.triton.cudagraphs = True

第三层的参数最多，因为 Inductor 的优化策略最复杂。

编译模式
============

``torch.compile(mode=...)`` 是最高层的配置入口：

.. list-table::
   :header-rows: 1

   * - 模式
     - 编译时间
     - 运行性能
     - 适用场景
   * - ``default``
     - 快
     - 好
     - 大多数场景
   * - ``reduce-overhead``
     - 快
     - 更好
     - 小模型、推理
   * - ``max-autotune``
     - 慢
     - 最好
     - 生产环境、大模型
   * - ``max-autotune-no-cudagraphs``
     - 慢
     - 最好
     - CUDA Graph 不适用的场景

``reduce-overhead`` 模式会额外做 CUDA Graph 捕获和 kernel 融合，减少 kernel launch 开销。

``max-autotune`` 不仅做 autotune，还会启用更激进的融合和 padding 优化。

Dynamo 配置
================

关键配置项在 ``torch/_dynamo/config.py`` 中：

.. code-block:: python

   import torch._dynamo.config as config

   # 编译缓存的最大条目（默认 64）
   config.cache_size_limit = 128

   # 是否默认假定形状是静态的
   config.assume_static_by_default = True

   # 是否允许跳过某些不支持的操作用于保持图完整
   config.suppress_errors = False

   # 记录 guard 失败（用于调试）
   config.record_guard_failure = True

.. warning::

   ``suppress_errors = True`` 会静默跳过编译错误回退到 eager。这可能导致生产环境中本来可以通过日志发现的编译问题被忽略。建议仅在开发环境使用。

Inductor 配置
==================

Inductor 的配置在 ``torch/_inductor/config.py`` 中，是最丰富的配置层。

**融合控制 ** ：

.. code-block:: python

   import torch._inductor.config as inductor_config

   # 最大融合大小（影响 kernel 大小）
   inductor_config.max_fusion_size = 8

   # 是否允许融合不同形状的操作
   inductor_config.allow_fusion_across_shapes = True

**Triton 相关 ** ：

.. code-block:: python

   # 启用 autotune
   inductor_config.max_autotune = True

   # 启用 CUDA Graph
   inductor_config.triton.cudagraphs = True

   # Triton kernel 的默认 BLOCK_SIZE
   inductor_config.triton.max_block_size = 4096

**调试选项 ** ：

.. code-block:: python

   # 禁用所有缓存（开发调试时使用）
   inductor_config.force_disable_caches = True

   # 在生成的代码中加入 nan 检查
   inductor_config.nan_asserts = True

   # 保存每个编译步骤的中间状态
   inductor_config.save_args = True

**内存优化** ：

.. code-block:: python

   # 重计算阈值（越大越少保存中间结果）
   inductor_config.recompute_threshold = 50

   # 是否启用 buffer 复用
   inductor_config.buffer_reuse = True

   # 布局优化（为卷积选择更优的内存布局）
   inductor_config.layout_optimization = True

环境变量
============

部分配置通过环境变量设置，在 Python 进程启动时生效：

.. list-table::
   :header-rows: 1

   * - 环境变量
     - 作用
     - 示例
   * - ``TORCH_LOGS``
     - 控制日志输出
     - ``+dynamo,+inductor``
   * - ``TORCHINDUCTOR_CACHE_DIR``
     - 编译缓存目录
     - ``/tmp/cache``
   * - ``TORCHINDUCTOR_MAX_AUTOTUNE``
     - 启用 autotune
     - ``1``
   * - ``TORCH_COMPILE_MODE``
     - 编译模式
     - ``max-autotune``
   * - ``TORCHDYNAMO_REPRO_AFTER``
     - 复现模式
     - ``aot``

通过 ``torch._inductor.config`` 的字典接口可以列出所有可用配置：

.. code-block:: python

   import torch._inductor.config as config
   print(config.__dict__.keys())

性能调优的最佳实践
=====================

**步骤 1：确认编译正常工作 **

.. code-block:: bash

   TORCH_LOGS="+perf_hints" python train.py

确认没有意外的 graph break。如果有，在 ``perf_hints`` 日志中可以看到建议。

** 步骤 2：使用合适的编译模式 **

.. code-block:: python

   # 训练场景
   model = torch.compile(model, mode="default")

   # 推理场景
   model = torch.compile(model, mode="reduce-overhead")

** 步骤 3：启用 autotune**

.. code-block:: python

   torch._inductor.config.max_autotune = True
   # 或
   model = torch.compile(model, mode="max-autotune")

** 步骤 4：启用 CUDA Graph**

.. code-block:: python

   torch._inductor.config.triton.cudagraphs = True

CUDA Graph 可以显着减少 kernel launch 开销，特别适合小 batch 推理场景。

** 步骤 5：内存优化 **

如果遇到 OOM 或显存占用过高：

.. code-block:: python

   # 减少中间结果保存
   torch._inductor.config.recompute_threshold = 100

   # 减少最大融合大小（降低单 kernel 显存需求）
   torch._inductor.config.max_fusion_size = 4

** 步骤 6：精度优化**

如果遇到精度问题：

.. code-block:: python

   # 使用全精度
   torch.set_float32_matmul_precision("highest")

   # 禁用心内核融合（可能改变计算顺序）
   torch._inductor.config.triton.fuse_max = False

``torch.set_float32_matmul_precision`` 控制矩阵乘法的精度：

- ``highest`` ：使用 FP32 累加（最准确，最慢）
- ``high`` ：使用 TF32（默认，平衡性能与精度）
- ``medium`` ：使用 BF16/FP16（快速，可能有精度损失）
