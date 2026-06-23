.. _resnet-optimization:

==============================
案例 1：ResNet 优化
==============================

.. note::

   **ResNet 是 torch.compile 最佳 benchmark 之一。**
   在 PyTorch 团队的内测中，ResNet50 在 ``max-autotune`` 模式下可以达到 2.8x 的推理加速比——这意味着一个本来需要 10ms 的前向传播，编译后只需要 3.5ms。这主要得益于三个因素：ResNet 全是 Conv + BN + ReLU 的组合（**Scheduler 最擅长的融合模式**）、没有控制流（**无 graph break**）、固定的输入尺寸（**形状稳定**）。如果你的模型也具备这三个特点，大概率也能获得 2x+ 的加速。

ResNet 是经典的 CNN 架构，也是 torch.compile 优化效果的典型 benchmark。这一节通过具体的优化步骤展示 torch.compile 对 CNN 模型的影响。

基线设置
============

首先建立不带 torch.compile 的基线：

.. code-block:: python

   import torch
   import torchvision.models as models
   import time

   model = models.resnet50(weights=None).cuda().train()
   x = torch.randn(32, 3, 224, 224, device='cuda')

   def measure(model, x, n_warmup=10, n_iter=50):
       # 预热
       for _ in range(n_warmup):
           model(x).sum().backward()
       torch.cuda.synchronize()

       # 测量
       start = time.time()
       for _ in range(n_iter):
           model(x).sum().backward()
       torch.cuda.synchronize()
       return (time.time() - start) / n_iter

   eager_time = measure(model, x)
   print(f"Eager 平均时间: {eager_time*1000:.1f} ms")

应用 torch.compile
======================

最简单的优化方式——直接包装：

.. code-block:: python

   compiled_model = torch.compile(model)
   compile_time = measure(compiled_model, x)
   print(f"Compiled 平均时间: {compile_time*1000:.1f} ms")
   print(f"加速比: {eager_time/compile_time:.2f}x")

对于 ResNet50，预期加速比在 **1.5x 到 2.5x** 之间。如果加速比低于预期，检查是否有 graph break。

检查 Graph Break
====================

.. code-block:: bash

   TORCH_LOGS="+perf_hints" python resnet_example.py

ResNet 中常见的 graph break 来源：

- ``torch.nn.functional.relu_`` （in-place 操作有时会导致 graph break）
- ``torch.nn.functional.batch_norm`` 在 training 模式下的特殊行为
- 自定义的损失函数

如果发现 graph break，可以通过 ``torch.compile(fullgraph=True)`` 强制触发错误来定位：

.. code-block:: python

   compiled_model = torch.compile(model, fullgraph=True)

如果 ``fullgraph=True`` 报错，说明模型中有无法捕获的操作，需要修复。

应用优化配置
================

启用更积极的优化：

.. code-block:: python

   # 使用 reduce-overhead 模式（推理场景）
   model_infer = torch.compile(model, mode="reduce-overhead")

   # 或使用 max-autotune（生产部署）
   model_best = torch.compile(model, mode="max-autotune")

   # 启用 CUDA Graph
   torch._inductor.config.triton.cudagraphs = True

   # 使用 TF32（如果 GPU 支持）
   torch.set_float32_matmul_precision("high")

优化后的预期性能
======================

.. list-table::
   :header-rows: 1

   * - 配置
     - ResNet50 训练加速比
     - ResNet50 推理加速比
   * - eager
     - 1.0x（基线）
     - 1.0x（基线）
   * - default
     - 1.5x - 2.0x
     - 1.8x - 2.5x
   * - reduce-overhead
     - -
     - 2.0x - 3.0x
   * - max-autotune
     - 1.8x - 2.5x
     - 2.5x - 4.0x

推理加速比通常比训练更高，因为训练需要保留中间结果用于反向传播，限制了融合的激进程度。

Profiling 结果分析
=======================

使用 profiler 分析优化后的 kernel：

.. code-block:: python

   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CUDA],
   ) as prof:
       compiled_model(x)

   print(prof.key_averages().table(sort_by="cuda_time_total"))

在 ResNet50 上，你会观察到：

- 大部分 kernel 是 Pointwise（逐元素操作，如 relu、add）
- 卷积操作通过 TemplateBuffer 调用 cuDNN
- 融合后的 kernel 命名包含 ``fused`` 关键字

如果看到大量的小 kernel（每个运行时间 < 10us），说明融合不充分，可以尝试增大 ``max_fusion_size``。
