.. _deferred-compilation:

==========================
延迟编译与再编译策略
==========================

.. note::

   **编译缓存相关提交有 434 次——缓存是 Inductor 最活跃的优化方向之一。 **
   Inductor 的缓存机制（PyCodeCache、FxGraphCache、Triton 磁盘缓存）经历了多轮迭代：从最早的简单哈希缓存，到后来的 content-hash-based 源码缓存，再到分布式的远程缓存（ ``FxGraphRemoteCache`` ）。缓存的边界条件极其复杂——需要处理 GPU 架构差异（A100 vs H100 的编译结果不能共享）、PyTorch 版本变化（不同版本的 Tensor 布局可能不同）、动态形状（同一源码在不同形状下是否可复用）。团队投入了大量精力在缓存命中率和正确性上，因为这直接影响用户的编译体验——** 缓存没命中，用户就要再等一遍编译 **。

Inductor 的编译过程可能非常耗时——对于大模型，代码生成和编译时间可能达到几分钟。为了缓解这个问题，Inductor 实现了多种** 延迟编译（deferred compilation） **和** 再编译（recompilation）** 策略。

异步编译（Async Compilation）
===================================

Inductor 支持在后台进程中异步编译 kernel，而主进程可以继续执行其他工作。

.. code-block:: text

   主进程                          编译进程
      │                              │
      ├─ lowering 完成               │
      │  获得 N 个 IRNode            │
      │                              │
      ├─ 将 IRNode 提交给编译进程     │
      │  ──────────────────────────→  │
      │                              ├─ 代码生成 + 编译 kernel 1
      │  继续处理其他工作              ├─ 代码生成 + 编译 kernel 2
      │                              ├─ ...
      │                              │
      ├─ 需要执行 kernel 时          │
      │  等待异步编译完成             │
      │  ←────────────────────────── │
      │  获取编译好的 .so 文件        │
      │  执行 kernel                 │
      │                              │

异步编译通过 ``AsyncCompile`` 类实现（在 ``pytorch/torch/_inductor/async_compile.py`` 中）。它在 Inductor 启动时创建一个 ``ProcessPoolExecutor`` 池，将编译任务分发到子进程。

.. code-block:: python
   :caption: pytorch/torch/_inductor/async_compile.py（简化示意）

   class AsyncCompile:
       def __init__(self):
           # 预热进程池，减少后续编译延迟
           self.pool = ProcessPoolExecutor(max_workers=num_workers)

       def triton(self, kernel_name, src_code, device_props):
           """异步提交一个 Triton kernel 编译任务"""
           future = self.pool.submit(
               compile_triton_kernel, kernel_name, src_code, device_props
           )
           return future

       def wait(self, future):
           """等待编译完成"""
           return future.result()

异步编译的好处在大模型中尤为明显。如果 lowering 产生了 50 个 IRNode，将它们分发到 4 个子进程编译，理想情况下编译时间可以降低到串行编译的 1/4。但实际的加速受限于编译任务之间的依赖——有些 kernel 必须在前置 kernel 编译完成后才能编译。

预热（Warming Up）
========================

``AsyncCompile`` 的预热在 Dynamo 查找 Inductor 后端时就开始：

.. code-block:: python

   # pytorch/torch/_dynamo/backends/inductor.py
   @register_backend
   def inductor(*args, **kwargs):
       from torch._inductor.async_compile import maybe_warm_pool
       maybe_warm_pool()  # 提前创建进程池
       from torch._inductor.compile_fx import compile_fx
       return compile_fx(*args,**kwargs)

``maybe_warm_pool()`` 在编译开始时创建进程池，这样后续的异步编译任务可以立即提交而不需要等待进程池初始化。

进程级缓存共享（跨实例）
=============================

第 2.4 节介绍了 Inductor 的磁盘缓存。这里补充一些与编译策略相关的细节。

当同一个模型在多个 GPU 上以数据并行方式训练时，每个 GPU 上的 Inductor 编译过程是相同的（相同的 FX Graph、相同的 GPU 架构）。如果每个 GPU 进程都独立编译，会产生重复的编译工作。

Inductor 通过磁盘缓存来解决这个问题。当进程 A 完成了编译，kernel 的 ``.so`` 文件被写入磁盘。进程 B 在编译同一个 IRNode 时，会检测到磁盘上已有缓存，跳过代码生成和编译：

.. code-block:: text

   进程 A (GPU 0)                  进程 B (GPU 1)
      │                              │
      ├─ 编译 kernel 1               │
      │  写入 .so 到磁盘             │
      │  ...                         │
      │                              ├─ 编译 kernel 1
      │                              │  发现磁盘缓存命中
      │                              │  直接加载 .so
      │                              │  （跳过代码生成）
      │                              │
      │  ...                         │  ...

对于大规模分布式训练（数十到数百个 GPU），这种跨进程缓存复用可以显著减少总编译时间。

然而需要注意的是，缓存 key 包含了 GPU 架构信息。如果集群中存在多种 GPU 型号（如 A100 和 H100 混部），不同型号之间的缓存无法共享。

再编译策略
===============

当 Inductor 检测到输入形状发生变化时，它不会立即丢弃已有缓存并重新编译，而是先尝试在现有 kernel 上执行：

1.** 如果形状变化可以通过调整 block size 适配 **：Inductor 会尝试使用不同的 grid size 来运行已有的 kernel，而跳过重新编译
2.** 如果形状变化导致 IRNode 的结构变化 **（如新增或删除了某些维度求和操作）：Inductor 必须重新编译

这些决策在 ``compile_fx.py`` 中通过 ``FxGraphCache`` 实现。 ``FxGraphCache`` 是 Inductor 图级别的缓存，它缓存的是完整的 FX Graph 的编译结果，而不是单个 kernel 的编译结果。当输入形状变化时，先检查缓存的 key（基于 FX Graph 的结构哈希）是否匹配。

渐进式编译（Progressive Compilation）
==========================================

Progressive compilation 是 Inductor 中的一种高级编译策略。它的思路是：** 先快速生成一个"可运行但未充分优化"的版本，然后在后台逐步优化 **。

.. code-block:: text

   Step 1: 快速编译
       ├─ 使用默认 heuristic，不做 autotune
       ├─ 快速生成 kernel
       └─ 模型开始训练
   
   Step 2: 后台优化
       ├─ 在后台进程中对 kernel 做 autotune
       ├─ 枚举不同的 block size / num warps
       ├─ benchmark 选择最优配置
       └─ 更新缓存

这个功能由 ``progressive_compile`` 相关代码支持。启用方式：

.. code-block:: bash

   TORCH_COMPILE_MODE=progressive python train.py

在 ``progressive`` 模式下，Inductor 会先用默认配置编译所有 kernel 让模型尽快开始训练，然后在后台逐步用 ``max-autotune`` 重新优化每个 kernel。优化完成后自动更新磁盘缓存，后续的训练 step 会自动使用更优的 kernel。

这种模式特别适合长时间训练任务：训练初期略慢（使用默认 kernel），训练中后期 kernel 逐步被优化到最佳性能。

小结
======

这一节介绍了 Inductor 的延迟编译与再编译策略：

- ** 异步编译 **：通过 ``AsyncCompile`` + ``ProcessPoolExecutor`` 在子进程中并行编译
- ** 预热 **： ``maybe_warm_pool()`` 提前创建进程池
- ** 跨进程缓存 **：多 GPU 场景下共享磁盘缓存，避免重复编译
- ** 再编译 **： ``FxGraphCache`` 管理图级别缓存，形状变化时优先尝试调整 grid size
- ** 渐进式编译** ：先快速编译让模型开始训练，后台逐步 autotune 优化 kernel
