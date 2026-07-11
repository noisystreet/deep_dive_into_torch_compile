.. _profiling:

============
性能分析
============

.. tip::

   **Chrome Trace 是分析 GPU 空闲时间最好的工具。 **
   在 ``chrome://tracing`` 中加载 ``trace.json`` 后，你可以看到 GPU kernel 的"间隙"——两个 kernel 之间 GPU 没有在做任何工作的空闲时间。这些间隙通常意味着 CPU 在准备下一个 kernel launch 的输入或进行同步操作。对于编译后的模型，理想的 trace 应该是"紧密排列"的——kernel 一个接一个，中间没有可见的间隙。如果你看到大量间隙，说明 kernel launch 开销占了主导，应该尝试 ``reduce-overhead`` 模式或 CUDA Graph。

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

Profiler 会显示 Triton kernel 的名称（如 ``triton_poi_fused_add_cos_sin_0`` ）和执行时间。

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

- ``poi`` ：Pointwise（逐元素操作）
- ``red`` ：Reduction（归约操作）
- ``pers_red`` ：Persistent Reduction（持久化归约）
- ``mm`` ：矩阵乘法
- ``split_scan`` ：前缀和

Chrome Trace
================

Profiler 的输出可以导出为 Chrome Trace 格式，在 ``chrome://tracing`` 中可视化：

.. code-block:: python

   prof.export_chrome_trace("trace.json")

在 Chrome 中打开 ``chrome://tracing`` ，加载 ``trace.json`` ，可以看到每个 kernel 的时间线。这对于分析 kernel launch 的间隙和 GPU 空闲时间特别有用。

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
   print(f"平均时间: {start.elapsed_time(end) / 100:.3f} ms")

这种手动 benchmark 比 profiler 更精确，因为 profiler 本身有开销。

性能瓶颈分析
====================

使用 profiler 数据识别性能瓶颈：

**Kernel launch 开销** 。如果 profiler 中 ``cudaLaunchKernel`` 的 CPU 时间占比很高（>20%），说明 kernel 太小、launch 次数太多。解决方案：
- 减少 graph break，让 Scheduler 融合更多操作
- 减小 ``TORCHINDUCTOR_MAX_AUTOTUNE`` 的限制

**内存带宽瓶颈** 。如果 kernel 的算术强度很低（计算量小、数据量大），说明受内存带宽限制。解决方案：
- 进一步融合 kernel，减少中间结果写回
- 使用更小的数据类型（FP16、BF16）

**计算瓶颈** 。如果 kernel 的算术强度很高（计算量大），说明受计算能力限制。解决方案：
- 使用 ``max-autotune`` 模式优化 tiling 参数
- 使用 Tensor Core（通过 ``tf32`` 或 ``fp16`` ）

Profiling 工作流
====================

下面的流程图展示了完整的 profiling 工作流：

.. mermaid::

   graph TD
       A["编译函数<br/>@torch.compile"] --> B["使用 torch.profiler 捕获"]
       B --> C["导出 Chrome Trace<br/>prof.export_chrome_trace()"]
       C --> D["在 chrome://tracing 中加载"]
       D --> E{"分析 GPU 时间线"}
       E --> F["识别 kernel 间隙"]
       E --> G["识别性能瓶颈"]
       F --> H["Kernel launch 开销过高"]
       G --> I["计算瓶颈 / 内存瓶颈"]
       H --> J["使用 reduce-overhead 模式"]
       H --> K["使用 CUDA Graph"]
       I --> L["使用 max-autotune"]
       I --> M["使用更低精度"]
       J --> N["验证性能提升"]
       K --> N
       L --> N
       M --> N

Kernel Launch 模式对比
==========================

编译后的 kernel launch 模式直接影响 GPU 利用率。下图对比了两种典型的 kernel launch 模式：

.. mermaid::

   graph LR
       subgraph Unfused["未融合模式 - 大量小 kernel 带间隙"]
           K1["Kernel 1"] --> G1["间隙<br/>(GPU 空闲)"]
           G1 --> K2["Kernel 2"]
           K2 --> G2["间隙<br/>(GPU 空闲)"]
           G2 --> K3["Kernel 3"]
           K3 --> G3["间隙<br/>(GPU 空闲)"]
           G3 --> K4["Kernel 4"]
       end

       subgraph Fused["融合模式 - kernel 紧密排列"]
           F1["融合 Kernel 1<br/>(sin+cos+add)"]
           F1 --> F2["融合 Kernel 2<br/>(sum+softmax)"]
           F2 --> F3["融合 Kernel 3<br/>(mm+relu)"]
       end

       Unfused -->|"fusion 优化"| Fused

未融合模式下，每个小 kernel 之间有明显的 GPU 空闲间隙（灰色方块），因为 CPU 需要为每个 kernel launch 进行准备工作。融合模式下，多个操作合并为少数 kernel，kernel 之间紧密排列，GPU 利用率显著提高。

torch.profiler + torch.compile 高级分析
==============================================

当模型被 ``torch.compile`` 编译后，profiler 的输出会比 eager 模式简单得多——不再是数百个 ATen 操作，而是十几个 Triton kernel。这种简化本身就是性能提升的来源。但如何深入分析这些 Triton kernel 的执行效率呢？

对比 Eager 与 Compiled Kernel 的 Trace
--------------------------------------------

最直观的分析方法是分别在 eager 和 compiled 模式下跑 profiler，并对比两者的 trace：

.. code-block:: python

   import torch
   import torchvision.models as models

   model = models.resnet18().cuda().eval()
   x = torch.randn(32, 3, 224, 224, device='cuda')

   # Eager 模式
   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CUDA],
   ) as prof:
       for _ in range(10):
           model(x)

   prof.export_chrome_trace("eager_trace.json")
   print("Eager kernel 数量:", len(prof.key_averages()))
   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

   # Compiled 模式
   compiled_model = torch.compile(model)
   # 预热
   for _ in range(3):
       compiled_model(x)
   torch.cuda.synchronize()

   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CUDA],
   ) as prof:
       for _ in range(10):
           compiled_model(x)

   prof.export_chrome_trace("compiled_trace.json")
   print("\nCompiled kernel 数量:", len(prof.key_averages()))
   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

将两个 ``trace.json`` 同时在 ``chrome://tracing`` 中打开（分别拖入），并排对比：

.. list-table:: Eager vs Compiled 预期对比
   :header-rows: 1

   * - 指标
     - Eager 模式
     - Compiled 模式
   * - Kernel 数量
     - 数百个 ATen 小 kernel
     - 数十个 Triton 融合 kernel
   * - Kernel 间隙
     - 大量间隙
     - 间隙明显减少
   * - GPU 利用率
     - 通常 40-60%
     - 通常 70-95%
   * - 总执行时间
     - 基线
     - 1.5x - 3x 加速

使用 profiler.step() 标记训练步骤
-----------------------------------------

在训练循环中使用 ``profiler.step()`` 可以让 trace 按 step 分组，便于分析每个 step 内的 kernel 分布：

.. code-block:: python

   model = torch.compile(model)
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CUDA,
                   torch.profiler.ProfilerActivity.CPU],
       schedule=torch.profiler.schedule(wait=1, warmup=3, active=5),
       on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
   ) as prof:
       for step in range(20):
           x = torch.randn(32, 3, 224, 224, device='cuda')
           y = model(x)
           loss = y.sum()
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()

           prof.step()  # 标记当前 step
           if step >= 10:
               break

   # 导出 Chrome Trace
   prof.export_chrome_trace("training_trace.json")

``schedule`` 参数控制 profiler 的采样策略：

- ``wait=1`` ：跳过前 1 个 step（避免预热干扰）
- ``warmup=3`` ：接下来 3 个 step 进行预热
- ``active=5`` ：记录 5 个 step 的数据

使用 TensorBoard 插件（ ``tensorboard_trace_handler`` ）可以直接在 TensorBoard 中查看 trace，比 Chrome Tracing 更方便。

分析 Kernel Fusion 机会
----------------------------

从 profiler 输出中可以识别出哪些操作没有被融合，从而手动引导 Scheduler 进行融合：

.. code-block:: python

   prof = torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CUDA],
   )

   # 在 profiler 输出中查找连续的小 kernel
   # 这些 kernel 本应被融合为一个
   kernel_events = [e for e in prof.events() if "triton_" in e.name]
   for i, evt in enumerate(kernel_events[:20]):
       print(f"{i:3d} | {evt.name:50s} | {evt.cuda_time_total:.3f}ms")

通过观察 kernel 名称中的 fused 信息，可以判断哪些操作被合并了。例如 ``triton_poi_fused_add_cos_sin_0`` 表示 add、cos、sin 三个操作被融合为一个 pointwise kernel。

如果发现大量独立的 pointwise kernel（名称中只有单个操作），说明 fusion 失效，需要检查 graph break 或调整 ``max_fusion_size`` 。

使用 profiler.export_chrome_trace 的最佳实践
---------------------------------------------------

``export_chrome_trace`` 是分析 GPU 性能最强大的工具。以下是一些使用技巧：

.. tip::

   **Chrome Trace 分析技巧 ** ：
   1. 使用 WASD 键平移和缩放时间线
   2. 按时间范围选中多个 kernel，底部会显示汇总信息
   3. 在每个 kernel 上悬停可以看到完整的 kernel 参数（grid、block、shared memory）
   4. 使用 ``Ctrl+F`` 搜索特定的 kernel 名称
   5. 注意流（stream）之间的依赖关系——通常 default stream 上的 kernel 是串行执行的

.. code-block:: python

   # 最佳实践：结构化导出 trace 文件
   import os

   trace_dir = "./traces"
   os.makedirs(trace_dir, exist_ok=True)

   # 为不同配置保存独立的 trace
   for name, model in [("eager", model_eager), ("compiled", model_compiled),
                       ("compiled_max_autotune", model_max_autotune)]:
       with torch.profiler.profile(
           activities=[torch.profiler.ProfilerActivity.CUDA],
       ) as prof:
           for _ in range(10):
               model(x)
       prof.export_chrome_trace(os.path.join(trace_dir, f"{name}.json"))
       print(f"Trace saved: {name}.json")

.. seealso::

   更多关于 Chrome Trace 的分析方法和 kernel 模式识别，可以参考第 10 章中的实战案例，特别是 ResNet 优化（10.1 节）和 LLM 推理优化（10.2 节）。

使用 memory profiler 调试显存
====================================

编译后的模型在显存使用模式上与 eager 模式有显著差异。理解这些差异对于避免 OOM 和优化 batch size 至关重要。

Compiled 模式的显存分配特点
----------------------------------

编译后的模型通常比 eager 模式 **使用更多显存** ，原因如下：

1.**中间结果持久化** ：为了支持反向传播，编译图会保留更多的中间 tensor
2.**权重梯度累积** ：编译后的 kernel 可能延长某些 tensor 的生命周期
3.**编译缓存** ：Triton kernel 的代码缓存和参数缓存会占用少量显存

.. note::

**编译后 OOM 了怎么办？ **
   如果 eager 模式能跑但编译后 OOM，可以尝试：
   - 设置 ``torch._inductor.config.recompute_threshold = 100`` 增加重计算
   - 使用 ``torch._inductor.config.triton.cudagraphs = False`` 禁用 CUDA Graph（CUDA Graph 会占用额外显存）
   - 缩小 batch size

使用 torch.cuda.memory_summary()
----------------------------------------

``torch.cuda.memory_summary()`` 提供当前 CUDA 设备显存使用的详细摘要：

.. code-block:: python

   import torch

   model = torch.compile(torchvision.models.resnet18().cuda())
   x = torch.randn(32, 3, 224, 224, device='cuda')

   # 运行前检查显存
   print("=== 运行前 ===")
   print(torch.cuda.memory_summary())

   # 运行模型
   y = model(x)
   y.sum().backward()

   # 运行后检查显存
   print("\n=== 运行后 ===")
   print(torch.cuda.memory_summary())

   # 对比 eager 模式的显存使用
   model_eager = torchvision.models.resnet18().cuda()
   x = torch.randn(32, 3, 224, 224, device='cuda')

   y = model_eager(x)
   y.sum().backward()

   print("\n=== Eager 模式 ===")
   print(torch.cuda.memory_summary())

``memory_summary()`` 输出中的关键信息：

- **Allocated memory** ：当前分配的显存量
- **Cached memory** ：CUDA 分配器缓存的显存量（可能大于 allocated）
- **Active memory** ：实际在用的显存量
- **GPU reserved memory** ：GPU 预留的显存

使用 torch.cuda.memory._dump_snapshot() 获取显存快照
----------------------------------------------------------

对于更精细的显存分析，可以使用 ``memory._dump_snapshot()`` 获取显存分配的时间线快照。

.. code-block:: python

   import torch.cuda.memory as mem

   # 启用内存快照记录
   mem._record_memory_history()

   model = torch.compile(torchvision.models.resnet18().cuda())
   x = torch.randn(32, 3, 224, 224, device='cuda')

   y = model(x)
   y.sum().backward()

   # 导出内存快照
   mem._dump_snapshot("memory_snapshot.pkl")

   # 停止记录
   mem._record_memory_history(enabled=None)

生成的 ``memory_snapshot.pkl`` 可以使用 PyTorch 的 ``memory_viz`` 工具可视化分析：

.. code-block:: bash

   # 使用 PyTorch 自带的可视化工具
   python -m torch.cuda.memory_viz memory_snapshot.pkl

这会在浏览器中打开一个交互式的显存分配时间线图，显示每个 tensor 的生命周期和分配位置。

识别 Compiled 模型中的显存泄漏
--------------------------------------

编译后的模型可能因为以下原因导致显存泄漏：

1.**编译缓存不断增长** ：如果输入形状不断变化，Dynamo 会持续生成新的 kernel 变体
2.**未被释放的中间 tensor** ：编译图的 autograd 中间结果未被正确释放
3.**CUDA Graph 重放缓冲区** ：使用 CUDA Graph 时，需要有意识地管理重放缓冲区

检测显存泄漏的标准方法：

.. code-block:: python

   import gc
   import torch

   def check_memory_leak(model, x, iterations=50):
       """检查模型在多次迭代后是否存在显存泄漏"""
       torch.cuda.reset_peak_memory_stats()
       initial_mem = torch.cuda.memory_allocated()

       for i in range(iterations):
           y = model(x)
           y.sum().backward()

           # 每 10 次迭代检查一次
           if (i + 1) % 10 == 0:
               gc.collect()
               torch.cuda.empty_cache()
               current_mem = torch.cuda.memory_allocated()
               print(f"迭代 {i+1}: 显存 = {current_mem / 1024**2:.1f} MB"
                     f" (增长 = {(current_mem - initial_mem) / 1024**2:.1f} MB)")

       # 最终清理后检查
       del y
       gc.collect()
       torch.cuda.empty_cache()
       final_mem = torch.cuda.memory_allocated()
       leaked = final_mem - initial_mem
       if leaked > 1024**2:  # 泄漏超过 1MB 视为异常
           print(f"\n警告: 可能存在显存泄漏，泄漏量 = {leaked / 1024**2:.1f} MB")
       else:
           print(f"\n显存使用正常，泄漏量 = {leaked / 1024**2:.1f} MB")
       return leaked

   model = torch.compile(torchvision.models.resnet18().cuda())
   x = torch.randn(32, 3, 224, 224, device='cuda')
   check_memory_leak(model, x)

.. warning::

   ** 显存泄漏不等于 memory_summary 中的 cached memory**。
   CUDA 分配器的缓存机制（cached memory）可能让已释放的显存看起来仍在占用。真正的泄漏是指 tensor 引用未被释放，导致显存持续增长。在判断泄漏时，应该关注经过 ``gc.collect() + empty_cache()`` 清理后的显存趋势。

Compiled 模型的显存优化策略
--------------------------------

.. list-table:: Compiled 模型显存优化策略
   :header-rows: 1

   * - 策略
     - 方法
     - 适用场景
   * - 增加重计算
     - ``recompute_threshold = 100``
     - 显存不足但可接受微小性能损失
   * - 减少编译缓存
     - ``cache_size_limit = 8``
     - 形状变化有限的场景
   * - 禁用 CUDA Graph
     - ``triton.cudagraphs = False``
     - 短迭代训练、显存极度紧张
   * - 使用激活检查点
     - ``torch.utils.checkpoint``
     - 深层网络、长序列模型
   * - 减少融合 kernel 大小
     - ``max_fusion_size = 6``
     - 显存碎片化严重时

.. seealso::

   关于 Inductor 的显存优化配置项，详见第 5 章的 Inductor 配置相关章节。关于 CUDA Graph 的内存管理，见第 7 章的 Triton 相关内容。
