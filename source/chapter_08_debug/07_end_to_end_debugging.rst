.. _end-to-end-debugging:

====================
端到端调试实战
====================

.. tip::

   **本节是最好的起点。**
   如果你时间有限，只读这一节就够了。它通过一个真实案例，串联了前面所有调试工具的实际用法。

场景设定
============

假设我们有一个 ResNet-18 模型，使用 ``torch.compile`` 编译后，性能不但没有提升，反而比 eager 模式还慢。这是一个非常典型的"优化失败"场景——通常不是因为 torch.compile 不好，而是因为某些条件没有被满足。

.. code-block:: python

   import torch
   import torchvision.models as models
   import time

   model = models.resnet18().cuda().train()
   x = torch.randn(32, 3, 224, 224, device='cuda')

   # Eager 基线
   def measure(model, x, n_iter=100):
       for _ in range(10):  # 预热
           model(x).sum().backward()
       torch.cuda.synchronize()

       start = time.time()
       for _ in range(n_iter):
           model(x).sum().backward()
       torch.cuda.synchronize()
       return (time.time() - start) / n_iter

   eager_time = measure(model, x)
   print(f"Eager 平均时间: {eager_time*1000:.1f} ms")

   # Compiled 模式
   compiled_model = torch.compile(model)
   compiled_time = measure(compiled_model, x)
   print(f"Compiled 平均时间: {compiled_time*1000:.1f} ms")
   print(f"加速比: {eager_time / compiled_time:.2f}x")

预期输出：

.. code-block:: text

   Eager 平均时间: 45.2 ms
   Compiled 平均时间: 52.8 ms
   加速比: 0.86x

编译后反而慢了 14%。下面我们逐步排查原因。

调试工作流概览
====================

.. mermaid::

   graph TD
       A["编译后比 Eager 慢"] --> B["Step 1: 检查 Graph Break<br/>TORCH_LOGS=+perf_hints,+dynamo"]
       B --> C{"有 graph break?"}
       C -->|"是"| D["Step 2: 修复 graph break<br/>使用 fullgraph=True 定位"]
       C -->|"否"| E["Step 3: Profiler 分析<br/>对比 eager vs compiled trace"]
       D --> E
       E --> F{"有 kernel launch 间隙?"}
       F -->|"是"| G["Step 4: 分析融合效率<br/>检查 kernel 数量"]
       F -->|"否"| H["Step 5: 检查动态形状<br/>guard 命中率"]
       G --> I{"fusion 不充分?"}
       I -->|"是"| J["调整 Scheduler 配置<br/>增大 max_fusion_size"]
       I -->|"否"| K["使用 reduce-overhead 模式<br/>或 CUDA Graph"]
       H --> L["标记动态维度"]
       J --> M["Step 6: 验证性能"]
       K --> M
       L --> M
       M --> N["对比 before/after<br/>确认正确性"]

Step 1 — 检查 Graph Break
============================

首先检查模型中是否有 graph break——这是编译后性能下降最常见的原因。

.. code-block:: bash

   TORCH_LOGS="+perf_hints,+dynamo" python train.py

日志输出示例：

.. code-block:: text

   [perf_hints] 检测到 graph break 在 forward() 的第 3 行
   [perf_hints]   原因: Unsupported: call_function torchvision.ops.nms
   [perf_hints]   产生的子图数量: 5
   [perf_hints]   建议: 使用 torch.compiler.disable 隔离不支持的函数

使用 ``fullgraph=True`` 可以精确定位 graph break 的位置：

.. code-block:: python

   @torch.compile(fullgraph=True)
   def forward(self, x):
       return self.model(x)

运行后会直接报错，显示第一个 graph break 的位置：

.. code-block:: text

   torch._dynamo.exc.Unsupported: call_function torchvision.ops.nms
   在文件 /path/to/model.py:42

   >   boxes = torchvision.ops.nms(boxes, scores, iou_threshold)

查看模型中哪些操作不被支持：

.. code-block:: python

   import torch
   from torch._dynamo import tracing

   # 列出模型中的所有 graph break
   model = models.resnet18().cuda()
   graph_breaks = tracing.detect_graph_breaks(model, (x,))
   for loc, reason in graph_breaks:
       print(f"Graph break at {loc}: {reason}")

.. note::

   **ResNet-18 本身不应该有 graph break**。
   ResNet-18 由 Conv2d、BatchNorm、ReLU 和残差连接组成，这些操作 Dynamo 都原生支持。如果你在 ResNet-18 中看到 graph break，请检查是否使用了自定义操作或第三方库（如 torchvision.ops.nms）。

Step 2 — 使用 Minimizer 定位（若有错误）
==============================================

如果存在编译错误或结果不一致，使用 minimizer 生成最小复现脚本：

.. code-block:: bash

   TORCHDYNAMO_REPRO_AFTER="aot" python train.py

这会在当前目录生成 ``minifier_launcher.py`` 和 ``repro.py``。如果模型很大，minimizer 可能需要数分钟来完成最小化。

.. code-block:: text

   # 输出示例
   Generating minimal repro...
   减少节点数: 200 -> 100 -> 50 -> 25 -> 12 -> 6
   保存最小复现脚本到: repro.py

   # 运行生成的脚本
   python repro.py

生成的 ``repro.py`` 内容示例：

.. code-block:: python

   import torch
   import torch.nn.functional as F

   # 最小复现脚本（由 minimizer 生成）
   def repro():
       x = torch.randn(32, 256, 56, 56, device='cuda')
       weight = torch.randn(512, 256, 3, 3, device='cuda')
       out = F.conv2d(x, weight, padding=1)
       out = F.relu(out)
       return out

   result = torch.compile(repro)()

.. tip::

   **何时使用 minimizer？**
   minimizer 主要用于编译错误或结果一致性错误。对于"只是慢"的性能问题，minimizer 帮助有限——优先使用 profiler 和日志系统。

Step 3 — Profiler 分析
==========================

使用 profiler 对比 eager 和 compiled 模式的 kernel 执行情况：

.. code-block:: python

   import torch.profiler

   def profile_model(model, x, name, n_iter=10):
       with torch.profiler.profile(
           activities=[torch.profiler.ProfilerActivity.CUDA],
       ) as prof:
           for _ in range(n_iter):
               model(x).sum().backward()

       prof.export_chrome_trace(f"{name}_trace.json")
       print(f"\n=== {name} ===")
       print(f"Kernel 数量: {prof.key_averages().count()}")
       print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
       return prof

   model_eager = models.resnet18().cuda().train()
   model_compiled = torch.compile(models.resnet18().cuda().train())

   # 预热
   for _ in range(5):
       model_compiled(x).sum().backward()
   torch.cuda.synchronize()

   prof_eager = profile_model(model_eager, x, "eager")
   prof_compiled = profile_model(model_compiled, x, "compiled")

输出对比：

.. code-block:: text

   === eager ===
   Kernel 数量: 342
   ---------------------------------------------------  ------------  ----------
   Name                                                Self CPU %     CUDA Total
   ---------------------------------------------------  ------------  ----------
   aten::conv2d                                         12.3%         15.2ms
   aten::batch_norm                                      8.7%         10.1ms
   aten::relu                                            3.2%          4.5ms
   ...

   === compiled ===
   Kernel 数量: 48
   ---------------------------------------------------  ------------  ----------
   Name                                                Self CPU %     CUDA Total
   ---------------------------------------------------  ------------  ----------
   triton_poi_fused_conv_batch_norm_relu_0              18.2%          8.3ms
   triton_conv_batch_norm_1                              15.1%          7.1ms
   cudaLaunchKernel                                      12.5%          0.0ms
   ...

注意 ``cudaLaunchKernel`` 的 CPU 时间占比。如果超过 10%，说明 kernel launch 开销显著。

将两个 trace 导出后在 ``chrome://tracing`` 中对比观察：

.. mermaid::

   graph LR
       subgraph EagerTrace["Eager Trace 特征"]
           E1["大量小 kernel"]
           E2["kernel 之间有明显间隙"]
           E3["GPU 利用率低 (40-60%)"]
       end

       subgraph CompiledTrace["Compiled Trace 特征"]
           C1["少量融合 kernel"]
           C2["kernel 之间间隙小"]
           C3["GPU 利用率高 (70-95%)"]
       end

       E1 -->|"对比"| C1
       E2 -->|"对比"| C2
       E3 -->|"对比"| C3

如果 compiled trace 中仍有大量间隙，说明 kernel launch 开销仍是瓶颈。

Step 4 — 分析融合效率
=========================

检查 Inductor 生成的 kernel 数量是否合理：

.. code-block:: python

   # 在编译日志中查看 kernel 数量
   TORCH_LOGS="+inductor" python train.py

.. code-block:: text

   [inductor] FX Graph 节点数: 186
   [inductor] 生成的 Kernel 数量: 48
   [inductor] 融合效率: 186 / 48 = 3.88 节点/kernel

如果 ``节点/kernel`` 的比值小于 3，说明 fusion 不充分。每个 kernel 平均只融合了不到 3 个操作。

检查是否存在 fusion 被阻止的情况：

.. code-block:: python

   import torch._inductor.config as inductor_config

   # 启用详细的 scheduling 日志
   inductor_config.debug = True

   # 增大 fusion 大小限制
   inductor_config.max_fusion_size = 12  # 默认 8

   # 更激进地融合 pointwise 操作
   inductor_config.aggressive_fusion = True

重新编译并对比 kernel 数量：

.. code-block:: python

   import torch._inductor.config as inductor_config
   inductor_config.max_fusion_size = 12

   model = torch.compile(models.resnet18().cuda().train())
   # 预热
   for _ in range(5):
       model(x).sum().backward()
   torch.cuda.synchronize()

   # 查看生成的 kernel 数量
   # 通过 profiler 统计
   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CUDA],
   ) as prof:
       for _ in range(10):
           model(x).sum().backward()

   kernel_count = len([e for e in prof.events() if "triton_" in e.name])
   print(f"调整后 kernel 数量: {kernel_count}")

.. note::

   **Fusion 不是越多越好**。
   过度融合可能导致单个 kernel 过于复杂，占用过多寄存器或共享内存，反而降低性能。``max_fusion_size`` 的推荐范围是 8-16。如果调到 16 后性能仍然没有提升，问题可能不在 fusion 上。

Step 5 — 应用修复
====================

根据前面的诊断结果，应用对应的修复策略。

修复 Graph Break
--------------------

如果存在 graph break，使用 ``torch.compiler.disable`` 将其隔离：

.. code-block:: python

   @torch.compiler.disable
   def custom_nms(boxes, scores, iou_threshold):
       return torchvision.ops.nms(boxes, scores, iou_threshold)

   class MyModel(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.backbone = models.resnet18()

       def forward(self, x):
           features = self.backbone(x)
           # 这个操作不会被编译
           boxes = custom_nms(features, ...)
           return boxes

减少 Kernel Launch 开销
---------------------------

使用 ``mode="reduce-overhead"`` 启用 CUDA Graph：

.. code-block:: python

   # 方式一：使用预设模式
   model = torch.compile(model, mode="reduce-overhead")

   # 方式二：精细控制
   model = torch.compile(model, options={
       "triton.cudagraphs": True,
       "max_autotune": False,
   })

   # 方式三：极致优化（训练场景慎用）
   model = torch.compile(model, mode="max-autotune")

综合修复示例
----------------

.. code-block:: python

   import torch
   import torchvision.models as models
   import torch._inductor.config as inductor_config

   # 配置 Inductor
   inductor_config.max_fusion_size = 12
   inductor_config.aggressive_fusion = True

   # 编译模型
   model = models.resnet18().cuda().train()
   model = torch.compile(model, mode="reduce-overhead")

   # 验证
   x = torch.randn(32, 3, 224, 224, device='cuda')
   for _ in range(10):  # 预热
       model(x).sum().backward()
   torch.cuda.synchronize()

   start = time.time()
   for _ in range(100):
       model(x).sum().backward()
   torch.cuda.synchronize()
   optimized_time = (time.time() - start) / 100

   print(f"优化后平均时间: {optimized_time*1000:.1f} ms")

Step 6 — 验证
================

性能对比
------------

.. code-block:: python

   import time

   def benchmark(model, x, n_iter=100, label=""):
       for _ in range(10):
           model(x).sum().backward()
       torch.cuda.synchronize()

       start = time.time()
       for _ in range(n_iter):
           model(x).sum().backward()
       torch.cuda.synchronize()
       avg = (time.time() - start) / n_iter * 1000
       print(f"{label:20s}: {avg:.1f} ms")
       return avg

   model_eager = models.resnet18().cuda().train()

   inductor_config.max_fusion_size = 12
   inductor_config.aggressive_fusion = True
   model_optimized = torch.compile(
       models.resnet18().cuda().train(),
       mode="reduce-overhead",
   )

   eager_time = benchmark(model_eager, x, label="Eager")
   compiled_before = benchmark(
       torch.compile(models.resnet18().cuda().train()),
       x, label="Compiled (default)",
   )
   compiled_after = benchmark(model_optimized, x, label="Compiled (tuned)")

   print(f"\n默认编译加速比: {eager_time / compiled_before:.2f}x")
   print(f"优化后加速比:   {eager_time / compiled_after:.2f}x")

预期输出：

.. code-block:: text

   Eager             : 45.2 ms
   Compiled (default): 52.8 ms
   Compiled (tuned)  : 22.1 ms

   默认编译加速比: 0.86x
   优化后加速比:   2.05x

正确性验证
--------------

确保优化后的结果与 eager 模式一致：

.. code-block:: python

   def check_correctness(model_eager, model_compiled, x, atol=1e-4, rtol=1e-4):
       model_eager.eval()
       model_compiled.eval()

       with torch.no_grad():
           out_eager = model_eager(x)
           out_compiled = model_compiled(x)

       diff = (out_eager - out_compiled).abs().max().item()
       print(f"最大绝对差异: {diff:.6f}")

       if diff < atol:
           print("正确性验证通过")
       else:
           print(f"警告: 差异超过阈值 {atol}")
           # 使用 minimizer 进一步排查
           print("建议运行: TORCHDYNAMO_REPRO_AFTER='aot' python train.py")

   check_correctness(
       models.resnet18().cuda().eval(),
       model_optimized.eval(),
       x,
   )

优化前后变化总结
-----------------------

.. mermaid::

   graph TD
       subgraph Before["优化前"]
           B1["默认 torch.compile"]
           B2["48 个 kernel"]
           B3["Kernel launch 开销占比 12.5%"]
           B4["52.8 ms / iter"]
           B1 --> B2 --> B3 --> B4
       end

       subgraph After["优化后"]
           A1["mode=reduce-overhead<br/>max_fusion_size=12"]
           A2["32 个融合 kernel"]
           A3["Kernel launch 开销占比 3.2%"]
           A4["22.1 ms / iter"]
           A1 --> A2 --> A3 --> A4
       end

       Before -->|"应用修复"| After

调试工具箱速查表
====================

.. list-table:: 调试工具箱速查表
   :header-rows: 1
   :widths: 25 25 25 25

   * - 工具
     - 用法
     - 解决的问题
     - 关键输出
   * - ``TORCH_LOGS="+dynamo"``
     - 设置环境变量后运行
     - Graph break、Dynamo 追踪问题
     - 编译过程、graph break 位置
   * - ``TORCH_LOGS="+perf_hints"``
     - 设置环境变量后运行
     - 性能瓶颈提示
     - Graph break 数量、子图统计
   * - ``TORCH_LOGS="+guards"``
     - 设置环境变量后运行
     - Dynamic shapes guard 失败
     - Guard failed 信息
   * - ``TORCH_LOGS="+inductor"``
     - 设置环境变量后运行
     - Inductor lowering、fusion 问题
     - Kernel 数量、lowering 过程
   * - ``TORCH_LOGS="+cuda_graphs"``
     - 设置环境变量后运行
     - CUDA Graph 捕获失败
     - 捕获/回退日志
   * - ``TORCH_LOGS="+pattern_matcher"``
     - 设置环境变量后运行
     - Pattern 匹配失败
     - 匹配尝试和失败原因
   * - ``TORCH_LOGS="+dynamic"``
     - 设置环境变量后运行
     - 动态形状诊断
     - 符号变量创建和约束
   * - ``TORCHDYNAMO_REPRO_AFTER``
     - 设置环境变量后运行
     - 编译错误、结果不一致
     - 最小复现脚本 (repro.py)
   * - ``torch.profiler.profile``
     - 在代码中包裹待测函数
     - Kernel 执行时间、launch 开销
     - Chrome Trace 文件
   * - ``torch.cuda.memory_summary``
     - 在代码中调用
     - 显存使用分析
     - 显存分配详细报表
   * - ``torch.cuda.memory._dump_snapshot``
     - 在代码中调用
     - 显存泄漏诊断
     - 内存快照文件 (.pkl)
   * - ``torch._dynamo.utils.counters``
     - 在代码中访问
     - 编译统计
     - compile/recompile/guard_fail 数量
   * - ``torch._dynamo.reset()``
     - 在代码中调用
     - 清除编译缓存
     - 无（副作用函数）
   * - ``@torch.compile(fullgraph=True)``
     - 装饰器参数
     - 确保无 graph break
     - 有 graph break 时报错

.. seealso::

   各个调试工具的详细用法和原理，见本章前面的各节：
   - 日志系统详解：8.1 节
   - Minimizer 使用：8.2 节
   - torch.compile 调试目录：8.3 节
   - Profiling 分析：8.4 节
   - Dynamic Shapes 调试：8.5 节
   - 常见问题排查：8.6 节