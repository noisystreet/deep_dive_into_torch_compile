.. _dynamic-shapes-debug:

=========================
Dynamic Shapes 调试
=========================

第 3.8 节介绍了符号形状（symbolic shapes）的原理——``ShapeEnv``、``SymNode`` 以及 ``dynamic=True`` 的工作方式。这一节聚焦 **如何诊断和优化** Dynamic Shapes 场景中的 guard 失败与频繁重编译问题。

判断是否触发了重新编译
===============================

最简单的判断方式是通过日志观察 guard 命中率：

.. code-block:: bash

   TORCH_LOGS="+guards" python train.py

如果日志中频繁出现 ``Guard failed:`` 信息，说明形状变化导致反复重新编译：

.. code-block:: text

   [guards] Guard failed: x.shape[0] == 512
   [guards] 触发重新编译...

也可以使用编译计数器：

.. code-block:: python

   import torch._dynamo.utils as dynamo_utils
   
   # 在每个 epoch 后打印编译统计
   print(dynamo_utils.counters["dynamo"])
   
   # 输出示例：
   # {'compile': 42, 'recompile': 38, 'guard_fail': 35}

``guard_fail`` 次数接近 ``recompile`` 次数，说明几乎每次运行都因 guard 失败而重新编译。

常见的 Dynamic Shapes 问题
================================

**形状变化被过度 guard**。以下代码会导致每次不同长度的输入都重新编译：

.. code-block:: python

   @torch.compile
   def fn(x):
       return x * 2

   fn(torch.randn(100))  # 编译，guard: x.shape[0] == 100
   fn(torch.randn(200))  # guard 失败，重新编译
   fn(torch.randn(300))  # guard 失败，重新编译

**数据集形状不一致**。最常见的原因——训练/验证集的 batch size 或序列长度不一致。

**数据预处理引入了动态形状**。如 ``torch.unique``、``torch.nonzero`` 等操作的输出形状取决于输入数据。

使用 ShapeEnv 日志
==========================

``ShapeEnv`` 是 PyTorch 中管理动态形状的核心组件。启用其日志可以追踪形状相关的决策：

.. code-block:: bash

   TORCH_LOGS="+dynamic" python train.py

日志会显示：

.. code-block:: text

   [dynamic] 创建符号变量 s0 = x.shape[0]
   [dynamic] 约束: s0 >= 1, s0 <= 1024
   [dynamic] Guard expr: s0 == 512
   [dynamic] 重新编译因为 guard: s0 == 512

如果看到 "重新编译因为 guard" 频繁出现，说明这个维度应该被标记为动态。

标记动态维度
====================

当某个维度确实需要动态变化时，应该显式标记它，让编译器为动态形状做优化：

**方式一：在 torch.compile 层面指定**

.. code-block:: python

   @torch.compile(dynamic=True)
   def fn(x):
       return x * 2

``dynamic=True`` 告诉编译器：所有输入都可能变化。编译器会使用符号形状（symbolic shapes）而不是具体数值来生成 guard。

**方式二：使用 ``torch._dynamo.mark_dynamic``**

.. code-block:: python

   import torch._dynamo as dynamo
   
   x = torch.randn(100)
   dynamo.mark_dynamic(x, 0)  # 标记第 0 维为动态

   @torch.compile
   def fn(x):
       return x * 2

   fn(x)  # 第 0 维不再被 guard

**方式三：使用 ``torch._export.DynamicShapes``（导出时指定）**

.. code-block:: python

   import torch._export as export
   
   dynamic_shapes = {"x": {0: torch.export.Dim("batch_size")}}
   exported = export.export(fn, (x,), dynamic_shapes=dynamic_shapes)

配置 Dynamic Shapes 策略
==================================

``torch._dynamo.config`` 中有多个与动态形状相关的配置项：

.. code-block:: python

   import torch._dynamo.config as config
   
   # 自动推断动态维度（默认启用）
   config.assume_static_by_default = False
   
   # 符号形状的最大约束数量
   config.dynamic_shapes = True
   
   # 缓存重新编译结果的最大数量
   config.cache_size_limit = 64

``assume_static_by_default = False`` 告诉 Dynamo：默认将所有维度视为动态的，而不是静态的。这会减少重新编译次数，但生成的 kernel 可能不如静态形状版本高效。

分析重新编译的影响
=========================

使用以下代码评估重新编译对训练时间的影响：

.. code-block:: python

   import time
   import torch._dynamo.utils as dynamo_utils
   
   start = time.time()
   for batch in dataloader:
       output = compiled_fn(batch)
   end = time.time()
   
   stats = dynamo_utils.counters["dynamo"]
   compile_time = stats.get("compile_time", 0)  # 累计编译时间（秒）
   total_time = end - start
   
   print(f"总训练时间: {total_time:.2f}s")
   print(f"编译时间: {compile_time:.2f}s ({compile_time/total_time*100:.1f}%)")
   print(f"编译次数: {stats.get('compile', 0)}")
   print(f"重新编译次数: {stats.get('recompile', 0)}")

如果编译时间占比超过 20%，且重新编译次数超过 100，就需要优化动态形状处理了。
