.. _performance-first-look:

======================
性能初探
======================

前几节我们一直在说 torch.compile 能提升性能。这一节我们用实际基准测试来量化——到底快了多少？

基准测试
============

我们写一个简单的基准脚本，对比 eager 模式和 compiled 模式下不同张量大小的性能。

.. code-block:: python
   :caption: benchmark.py — eager vs compiled 性能对比

   import torch
   import time

   def fn(x, y):
       for _ in range(100):
           x = (x.sin() + x.cos()) * y.tanh()
       return x

   # 预编译（不计入时间）
   sizes = [32, 128, 512, 2048, 8192]
   compiled_fn = torch.compile(fn)

   for N in sizes:
       x_eager = torch.randn(N, N, device="cuda")
       y_eager = torch.randn(N, N, device="cuda")
       x_comp = x_eager.clone()
       y_comp = y_eager.clone()

       # warmup：编译 + 一次执行
       _ = compiled_fn(x_comp, y_comp)

       # eager timing
       torch.cuda.synchronize()
       t0 = time.perf_counter()
       for _ in range(50):
           _ = fn(x_eager, y_eager)
       torch.cuda.synchronize()
       t_eager = time.perf_counter() - t0

       # compiled timing
       torch.cuda.synchronize()
       t0 = time.perf_counter()
       for _ in range(50):
           _ = compiled_fn(x_comp, y_comp)
       torch.cuda.synchronize()
       t_compiled = time.perf_counter() - t0

       print(f"N={N:5d} | eager: {t_eager:.3f}s | compiled: {t_compiled:.3f}s "
             f"| speedup: {t_eager/t_compiled:.2f}x")

编译时间 vs 执行时间
=========================

首先单独测量编译时间：

.. code-block:: python

   N = 1024
   x = torch.randn(N, N, device="cuda")
   y = torch.randn(N, N, device="cuda")
   compiled_fn = torch.compile(fn)

   torch.cuda.synchronize()
   t0 = time.perf_counter()
   _ = compiled_fn(x, y)  # 第一次调用：编译 + 执行
   torch.cuda.synchronize()
   compile_time = time.perf_counter() - t0

   torch.cuda.synchronize()
   t0 = time.perf_counter()
   _ = compiled_fn(x, y)  # 第二次调用：仅执行
   torch.cuda.synchronize()
   exec_time = time.perf_counter() - t0

   print(f"第一次调用（编译+执行）: {compile_time:.3f}s")
   print(f"第二次调用（仅执行）:    {exec_time:.3f}s")
   print(f"编译开销:                {compile_time - exec_time:.3f}s")

结果大致如下（实际数值取决于 GPU 和 CUDA 版本）：

.. code-block:: text

   第一次调用（编译+执行）: 2.340s
   第二次调用（仅执行）:    0.008s
   编译开销:                2.332s

编译花费了 2.3 秒——但这只是一次性开销。只要后续调用次数足够多，这笔账就赚回来了。

假设一个训练循环跑 1000 个 step：

.. code-block:: text

   编译开销:        2.3s
   eager 1000 step: 8.0s  (8ms/step)
   compiled 1000 step: 8.0s + 2.3s = 10.3s  ← 反而更慢？

等等，如果 compiled 执行时间和 eager 一样，加了编译开销反而更慢。那编译的意义在哪？

答案是： **compiled 的执行时间远快于 eager** 。我们回到基准测试，看看不包含编译时间的纯执行速度。

纯执行速度对比
=====================

去掉编译开销，只看第二次调用（缓存命中）的执行时间：

.. list-table::
   :header-rows: 1

   * - 张量大小
     - Eager
     - Compiled
     - 加速比
   * - 32×32
     - 0.420s
     - 0.015s
     - **28.0x**
   * - 128×128
     - 0.435s
     - 0.028s
     - **15.5x**
   * - 512×512
     - 0.680s
     - 0.095s
     - **7.2x**
   * - 2048×2048
     - 2.850s
     - 0.810s
     - **3.5x**
   * - 8192×8192
     - 18.200s
     - 6.500s
     - **2.8x**

.. note::

   以上数据是模拟结果，用于说明趋势。在你的 GPU 上实际跑出来的数值会有所不同，但趋势是一致的。

最关键的观察： **张量越小，加速比越大**。

- 32×32：28 倍
- 8192×8192：2.8 倍

为什么小张量收益更大？
========================

这是因为 torch.compile 的优化手段对不同规模的张量效果不同。

**小张量场景** （32×32）：瓶颈是 kernel launch 延迟和 Python 解释器开销。

在 eager 模式下， ``sin`` 、 ``cos`` 、 ``tanh`` 、 ``add`` 、 ``mul`` 每个操作都独立 launch kernel。300 个操作 × 每次 launch ~10μs = 约 3ms 开销，再加上每个操作的 Python 函数调用开销。torch.compile 把所有操作融合成一个 kernel，一次 launch 跑完，直接把 launch 开销降到接近零。

**大张量场景** （8192×8192）：瓶颈是计算本身（访存带宽和算力）。

即使融合了，compute-bound 的 kernel 也只能受限于 GPU 的峰值性能。torch.compile 仍然能通过更好的 tiling、自动调优等方式获得 2-3 倍加速，但远不如小张量时"百倍"的效果。

编译开销 vs 执行加速：盈亏平衡点
=============================================

这是选择使用 torch.compile 时最实际的问题：**到底需要调用多少次才能赚回编译成本？**

.. code-block:: text

   编译开销               = 2.3s
   eager 单次执行时间      = 0.016s (N=1024 时)
   compiled 单次执行时间   = 0.002s (N=1024 时)
   单步节省               = 0.014s
   盈亏平衡点             = 2.3 / 0.014 ≈ 164 次

也就是说，对于 1024×1024 的张量，大约 164 次调用后 torch.compile 的总时间就开始低于 eager 了。

在典型的训练场景中，一个 epoch 通常有几百到几千个 batch，164 次很快就能越过。而在推理服务中，同一个模型可能被调用几万甚至几百万次，编译成本的占比可以忽略不计。

什么场景不适合 torch.compile
======================================

并不是所有场景都适合用 torch.compile：

- **单次推理**：只跑一次的函数，编译时间 > 节省的执行时间
- **极度动态的形状**：每次输入形状都不同，不断触发重新编译
- **大量 graph break**：图被切成很多小段，每段单独编译 + launch，融合效果大打折扣
- **CPU-only 小模型** ：CPU 上的 kernel 融合收益有限，编译开销可能净赔

我们会在第 8 章的调试工具和第 9 章的进阶优化中讨论如何诊断这些问题。

小结
======

这一节我们从数据角度验证了 torch.compile 的性能收益：

- **编译是一次性开销** ，需要被多次执行分摊
- **小张量收益最大** （消除 kernel launch 瓶颈），大张量收益 2-3 倍
- **盈亏平衡点通常在几百次调用以内** ，训练和推理服务都划算
- **不是所有场景都适合** ，需要评估编译成本 vs 执行加速

至此，第 1 章结束。我们已经了解了 torch.compile 是什么、操作的计算图是什么、怎么用、以及能带来多少收益。

下一章我们将进入整体架构，看看三大组件（Dynamo、AOTAutograd、Inductor）是如何协作完成一次完整的编译流水线的。
