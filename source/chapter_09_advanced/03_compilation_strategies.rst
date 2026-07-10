.. _compilation-strategies:

=====================
编译策略
=====================

编译策略（Compilation Strategies）是 torch.compile 在编译时间与运行性能之间做权衡的一组选择。理解这些策略可以帮助你在不同部署场景下做出最优选择。

策略总览
============

.. code-block:: text

   策略                     编译时间   运行性能   适用场景
   ──────────────────────────────────────────────────────
   eager（无编译）           0         差        开发调试
   default                  快        好        开发、训练
   reduce-overhead          快        更好      推理、小 batch
   max-autotune             慢        最好      生产训练、推理
   progressive              渐进      逐步优化  长时间训练

Eager 模式
==============

不作为后端使用，而是通过 ``torch.compile(backend="eager")`` 作为 baseline：

.. code-block:: python

   @torch.compile(backend="eager")
   def fn(x):
       return torch.sin(x) + torch.cos(x)

这样做会完整经过 Dynamo 捕获和 AOTAutograd 分区，但最后的执行回退到 eager。用于验证编译流水线本身是否有问题。

Default 模式
================

默认模式使用 Inductor 后端的启发式规则，不做大规模 autotune：

.. code-block:: text

   优点: 编译快（秒级），大多数场景下性能提升 1.5x-3x
   缺点: 不是最优的 tiling 参数，kernel 可能不是最高效的

适用于：开发迭代、CI 测试、简单的训练场景。

Reduce-overhead 模式
========================

在 default 基础上额外启用：

- **CUDA Graph 捕获 ** ：将多个 kernel launch 合并为一个图
- **更激进的融合 ** ：减少 kernel 总数

.. code-block:: python

   @torch.compile(mode="reduce-overhead")
   def fn(x):
       ...

适用于：推理场景，特别是小 batch 场景。kernel launch 的开销占比在 batch size 较小时更加显着。

Max-autotune 模式
=====================

最全面的优化模式，启用：

.. code-block:: text

   - Triton autotune: 枚举 BLOCK_SIZE、num_warps、num_stages
   - 矩阵乘法 padding: 对齐到 Tensor Core 的尺寸
   - 布局优化: 为卷积选择 channels-last 等布局
   - 更激进的 fusion: 包括横跨多个操作的融合
   - CUDA Graph: 自动捕获

.. code-block:: python

   @torch.compile(mode="max-autotune")
   def fn(x):
       ...

编译时间可能长达几分钟，但运行性能最好。适用于：

- 生产环境部署
- 大模型训练
- 性能基准测试

Progressive 模式
=====================

渐进式编译的思路是**先快速开始训练，再逐步优化 ** ：

.. code-block:: python

   import os
   os.environ["TORCH_COMPILE_MODE"] = "progressive"

   model = torch.compile(model)

工作流程：

.. code-block:: text

   Step 1: 使用 default 模式快速编译所有 kernel
           模型开始训练（性能一般）

   Step 2: 后台进程逐个对 kernel 做 autotune
           选择最优的 tiling 配置

   Step 3: 更新磁盘缓存
           后续训练 step 自动使用优化后的 kernel

适用于：长时间训练任务（数小时到数天），编译时间可以被训练时间摊平。

动态形状策略
================

当输入形状变化时，编译策略需要结合动态形状处理：

.. code-block:: python

   @torch.compile(dynamic=True)
   def fn(x):
       ...

``dynamic=True`` 改变了编译策略：

- **更多使用符号形状 ** ：guard 表达式中使用 ``s0`` 、 ``s1`` 等符号变量代替具体数值
- **更少的重新编译 ** ：形状在合理范围内的变化不会触发重新编译
- **可能降低单次性能 ** ：生成的 kernel 针对通用形状优化，不如固定形状的极端优化版本

选择合适策略的决策树
============================

.. code-block:: text

   是要快速验证还是生产运行？
   ├─ 快速验证 → default 模式
   └─ 生产运行
       ├─ 推理场景 → reduce-overhead 或 max-autotune
       └─ 训练场景
           ├─ 短时间训练（<1h）→ max-autotune
           ├─ 长时间训练（>1h）→ progressive
           └─ 形状频繁变化 → dynamic=True + default

兼容性注意事项
====================

**CUDA Graph 限制 ** 。CUDA Graph 在以下场景会回退：
- 包含 CPU 操作（如 ``print`` 、 ``torch.tensor.item`` ）
- 动态控制流（ ``if x.sum() > 0`` ）
- 动态形状

如果 CUDA Graph 回退，日志中会有提示：

.. code-block:: bash

   TORCH_LOGS="+perf_hints" python train.py

**Tensor Core 使用 ** 。Tensor Core 要求矩阵尺寸是 8 或 16 的倍数。 ``max-autotune`` 模式下的 padding 优化可以自动对齐非对齐的矩阵乘。

**内存使用** 。 ``max-autotune`` 模式可能会因为更激进的融合而增加峰值显存使用。如果遇到 OOM，尝试减少融合大小或切换到 ``default`` 模式。
