.. _profiling:

============
性能分析
============

性能分析（Profiling）是理解编译后模型性能瓶颈的关键手段。这一节介绍如何分析 ``torch.compile`` 生成的 kernel 的性能。

torch.profiler
===================

PyTorch 自带的 ``torch.profiler`` 可以直接用于分析编译后的函数：

.. code-block:: python

   import torch
   
   @torch.compile
   def fn(x):
       return (torch.sin(x) + torch.cos(x)).sum()
   
   x = torch.randn(10000, device='cuda')
   
   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CUDA],
   ) as prof:
       result = fn(x)
   
   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

输出示例：

.. code-block:: text

   ---------------------------------------------------  ------------  ----------  ----------  -----------  ----------  ----------  
   Name                                                Self CPU %     Self CPU    CPU total    CUDA total    # of Calls
   ---------------------------------------------------  ------------  ----------  ----------  -----------  ----------  
   triton_poi_fused_add_cos_sin_0                      23.5%         0.2ms       0.2ms        1.2ms         1
   aten::sum                                           12.1%         0.1ms       0.3ms        0.8ms         1
   cudaLaunchKernel                                    8.3%          0.1ms       0.1ms        0.0ms         1
   ...

Profiler 会显示 Triton kernel 的名称（如 ``triton_poi_fused_add_cos_sin_0``）和执行时间。

``mark_step_begin`` 参数
--------------------------------

对于编译后的模型，可以使用 ``mark_step_begin`` 标记每个 step 的边界：

.. code-block:: python

   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CUDA],
   ) as prof:
       for step in range(10):
           prof.step()  # 标记 step 边界
           result = fn(x)

这会在 profiler 输出中按 step 分组。

理解 Triton kernel 名称
==============================

Inductor 生成的 Triton kernel 名称编码了 kernel 的信息：

.. code-block:: text

   triton_poi_fused_add_cos_sin_0
   ├──     ├──         ├──   ├──  └── kernel 编号
   │      │           │    └── 融合的操作列表
   │      │           └── "fused" 表示这是融合 kernel
   │      └── "poi" = pointwise, "red" = reduction, "mm" = matmul
   └── 固定前缀

常见的 kernel 类型缩写：

- ``poi``：Pointwise（逐元素操作）
- ``red``：Reduction（归约操作）
- ``pers_red``：Persistent Reduction（持久化归约）
- ``mm``：矩阵乘法
- ``split_scan``：前缀和

Chrome Trace
================

Profiler 的输出可以导出为 Chrome Trace 格式，在 ``chrome://tracing`` 中可视化：

.. code-block:: python

   prof.export_chrome_trace("trace.json")

在 Chrome 中打开 ``chrome://tracing``，加载 ``trace.json``，可以看到每个 kernel 的时间线。这对于分析 kernel launch 的间隙和 GPU 空闲时间特别有用。

Kernel Benchmark
====================

Inductor 在 ``max-autotune`` 模式下会自动对 kernel 做基准测试。你也可以手动 benchmark 编译后的函数：

.. code-block:: python

   @torch.compile(mode="max-autotune")
   def fn(x):
       return torch.sin(x) + torch.cos(x)

   x = torch.randn(10000, device='cuda')
   
   # 预热
   for _ in range(10):
       fn(x)
   
   # 基准测试
   torch.cuda.synchronize()
   start = torch.cuda.Event(enable_timing=True)
   end = torch.cuda.Event(enable_timing=True)
   
   start.record()
   for _ in range(100):
       fn(x)
   end.record()
   
   torch.cuda.synchronize()
   print(f"平均时间: {start.elapsed_time(end) / 100:.3f} ms)

这种手动 benchmark 比 profiler 更精确，因为 profiler 本身有开销。

性能瓶颈分析
====================

使用 profiler 数据识别性能瓶颈：

**Kernel launch 开销**。如果 profiler 中 ``cudaLaunchKernel`` 的 CPU 时间占比很高（>20%），说明 kernel 太小、launch 次数太多。解决方案：
- 减少 graph break，让 Scheduler 融合更多操作
- 减小 ``TORCHINDUCTOR_MAX_AUTOTUNE`` 的限制

**内存带宽瓶颈**。如果 kernel 的算术强度很低（计算量小、数据量大），说明受内存带宽限制。解决方案：
- 进一步融合 kernel，减少中间结果写回
- 使用更小的数据类型（FP16、BF16）

**计算瓶颈**。如果 kernel 的算术强度很高（计算量大），说明受计算能力限制。解决方案：
- 使用 ``max-autotune`` 模式优化 tiling 参数
- 启用 Tensor Core（通过 ``tf32`` 或 ``fp16``）
