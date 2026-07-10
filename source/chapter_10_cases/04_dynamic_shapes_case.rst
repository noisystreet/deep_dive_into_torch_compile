.. _dynamic-shapes-case:

==============================
案例 4：Dynamic Shapes
==============================

这一节通过 NLP 场景的 padding 策略对比，展示如何在 torch.compile 中高效处理动态形状。

场景：变长序列批处理
==========================

在 NLP 中，不同句子的长度不同。批处理时有两种策略：

**策略 A：Padding 到固定长度 **

.. code-block:: python

   def collate_fn_pad(batch, max_len=128):
       """将所有序列 padding 到 max_len"""
       padded = []
       for seq, label in batch:
           if len(seq) < max_len:
               seq = torch.nn.functional.pad(seq, (0, max_len - len(seq)))
           else:
               seq = seq[:max_len]
           padded.append(seq)
       return torch.stack(padded), torch.tensor([l for _, l in batch])

** 策略 B：Batch 内 Padding（不跨 batch 固定）**

.. code-block:: python

   def collate_fn_batch_pad(batch):
       """填充到当前 batch 内的最大长度"""
       max_len = max(len(seq) for seq, _ in batch)
       padded = []
       for seq, label in batch:
           seq = torch.nn.functional.pad(seq, (0, max_len - len(seq)))
           padded.append(seq)
       return torch.stack(padded), torch.tensor([l for _, l in batch])

策略 A 不会触发重新编译（序列长度固定），但会浪费计算资源。策略 B 更节省计算，但每次 batch 的长度变化时会触发重新编译。

两种策略的性能对比
========================

.. code-block:: python

   import torch
   import time

   model = BertModel().cuda().eval()
   compiled = torch.compile(model)

   # 测试数据（模拟变长序列）
   lengths = [32, 64, 96, 128, 160, 192, 224, 256]
   
   for strategy, collate_fn in [("fixed_pad", collate_fn_pad), ("batch_pad", collate_fn_batch_pad)]:
       compiled = torch.compile(model)
       times = []
       
       for i in range(100):
           # 随机选择长度
           length = lengths[i % len(lengths)]
           x = torch.randn(16, length, 768, device='cuda')
           
           start = time.time()
           compiled(x)
           torch.cuda.synchronize()
           times.append(time.time() - start)
       
       avg_time = sum(times) / len(times)
       print(f"{strategy}: {avg_time*1000:.2f} ms")

期望结果：策略 B 的首次运行更慢（编译开销），但后续运行更快（更少的 padding 浪费）。

使用动态形状优化
====================

当使用策略 B 时，启用动态形状可以大幅减少重新编译的次数：

.. code-block:: python

   compiled = torch.compile(model, dynamic=True)

``dynamic=True`` 的效果：

- 编译器为宽范围的长度生成统一的 kernel，而不是为每个具体长度编译一个特化版本
- 形状变化不会触发 guard 失败和重新编译
- 生成的 kernel 性能可能略低于静态形状版本（约 5-10%），但避免了重复编译的开销

如果动态形状场景下性能仍然不理想，可以通过 ``torch._dynamo.config`` 调整：

.. code-block:: python

   import torch._dynamo.config as config

   # 允许更大的缓存
   config.cache_size_limit = 128

   # 不要默认假设形状是静态的
   config.assume_static_by_default = False

   # 启用形状的自动推断
   config.dynamic_shapes = True

显式标记动态维度
====================

如果模型只有特定维度是动态的（如序列长度），可以精确标记：

.. code-block:: python

   import torch._dynamo as dynamo

   x = torch.randn(16, 128, 768)  # shape: [batch, seq_len, hidden]
   dynamo.mark_dynamic(x, 1)       # seq_len 是动态的

   @torch.compile
   def fn(x):
       return model(x)

   fn(x)

这告诉编译器：只有 seq_len 维度会变化，batch 和 hidden 维度是固定的。比全局 ``dynamic=True`` 更精确，生成的 kernel 也更好。

特殊处理数据依赖的形状
==========================

有些操作的输出形状取决于输入数据（如 ``torch.nonzero`` 、 ``torch.unique`` ），这些操作会导致 "数据依赖的形状"——编译器无法在编译时确定输出形状。

.. code-block:: python

   @torch.compile
   def fn(x):
       mask = x > 0.5
       indices = torch.nonzero(mask)  # 输出形状取决于 x 的值
       return indices

这些操作会强制生成一个 graph break（因为输出形状无法在编译时确定）， ``torch.nonzero`` 之后的代码在 eager 模式下执行。

如果必须使用数据依赖形状的操作，推荐的方案是：

1. 将其隔离在 ``torch.compiler.disable`` 中
2. 或者重写为形状可预测的版本（如用 ``torch.where`` 替代）

.. code-block:: python

   @torch.compile
   def fn(x):
       # 避免 nonzero，使用固定形状的版本
       return torch.where(x > 0.5, x, torch.zeros_like(x))

选择最佳策略
================

.. list-table::
   :header-rows: 1

   * - 场景
     - 推荐策略
     - 理由
   * - 数据集形状基本固定
     - 固定 padding + default 模式
     - 简单高效
   * - 形状变化但变化范围有限
     - ``dynamic=True``
     - 避免重新编译
   * - 形状变化范围大
     - ``mark_dynamic`` 精确标记
     - 最精确的控制
   * - 无法预测形状
     - ``assume_static_by_default=False``
     - 减少 guard 失败

重新编译与 Padding 浪费的权衡
================================================

动态形状场景的核心矛盾在于： **Padding 浪费计算资源，重新编译浪费编译时间** 。下面的决策树展示了不同场景下的最优选择：

.. mermaid::

   flowchart LR
       A["输入序列\n长度分布"] --> B{"分布范围?"}

       B -->|"窄 (min~max < 2x)"| C["固定 Padding 到 max_len\n无需重新编译"]
       B -->|"宽 (min~max > 2x)"| D{"样本总数?"}

       D -->|"少 (<1000)"| E["Batch 内 Padding +\ndynamic=True\n少量重新编译"]
       D -->|"多 (>10000)"| F{"性能目标?"}

       F -->|"低延迟"| G["Bucket Padding\n按长度分桶\n桶内固定 + 桶间重编译"]
       F -->|"高吞吐"| H["全局 Padding\n一次编译\n浪费 ~30% 计算"]

       style C fill:#e8f5e9
       style E fill:#fff3e0
       style G fill:#e3f2fd
       style H fill:#fce4ec

量化分析：假设序列长度分布在 32~256 之间，每个 batch 包含 16 个样本：

.. list-table::
   :header-rows: 1

   * - 策略
     - 平均有效计算比例
     - 重新编译次数
     - 端到端时间 (估算)
   * - 全局 Padding (256)
     - ~40% (大部分是 padding)
     - 1 次
     - 编译 ~60s + 训练 100s
   * - Batch Padding (dynamic=True)
     - ~85%
     - ~10 次
     - 编译 ~30s + 训练 70s
   * - 分桶 (4 个桶)
     - ~75%
     - 4 次
     - 编译 ~5s + 训练 80s

动态形状与 Export（torch.export）
=============================================

``torch.export`` 是 PyTorch 2.x 引入的静态图导出机制，与动态形状结合可以解决生产部署场景中的形状变化问题。

基本用法
----------------

.. code-block:: python

   import torch
   from torch.export import export
   from torch.export.dynamic_shapes import Dim

   class MyModel(torch.nn.Module):
       def forward(self, x, lengths):
           # x: [batch, seq_len, hidden]
           # lengths: 每个样本的实际长度
           return x.sum(dim=1)

   model = MyModel().cuda().eval()

   # 定义动态维度
   batch = Dim("batch", min=1, max=64)
   seq_len = Dim("seq_len", min=16, max=512)

   # 示例输入
   x = torch.randn(8, 128, 768, device="cuda")
   lengths = torch.randint(16, 128, (8,), device="cuda")

   # 导出带动态形状的图
   exported_program = export(
       model,
       (x, lengths),
       dynamic_shapes={"x": {0: batch, 1: seq_len}, "lengths": {0: batch}},
   )

   # 使用导出的图
   x_new = torch.randn(16, 256, 768, device="cuda")
   result = exported_program.module()(x_new, lengths_new)

Export 的动态形状通过 ``Dim`` 对象定义维度约束。编译器使用这些约束生成能够在指定范围内适应任何形状的 kernel，从而避免重新编译。

Export + Dynamic Shapes 的优化效果
---------------------------------------------------

.. code-block:: text

   没有 Export:
       输入形状 (8, 64, 768) → 编译特化 kernel
       输入形状 (8, 128, 768) → guard 失败 → 重新编译
       总编译时间: ~120 秒 (10 次重新编译)

   使用 Export + Dynamic Shapes:
       输入形状 (8, 64, 768) → 执行通用 kernel
       输入形状 (8, 128, 768) → 同一个通用 kernel
       总编译时间: ~20 秒 (1 次编译)

需要注意的是， ``export`` 生成的图是静态的——它不支持控制流（如 ``if`` 依赖 tensor 值）。如果你的模型包含动态控制流，需要使用 ``torch.export.Dim`` 的 ``dynamic=True`` 配合 ``torch.cond`` 或 ``torch.map`` 。

Guard 调试实战：TORCH_LOGS 输出分析
===============================================

当动态形状导致意外重新编译时，TORCH_LOGS 是最强大的调试工具。下面通过具体的日志输出来理解 guard 机制的行为。

示例场景
----------------

.. code-block:: python

   # 场景：序列长度在 32~128 之间变化
   @torch.compile
   def decode_one_step(model, x, kv_cache):
       return model(x, kv_cache)

   x = torch.randn(1, 64, 768, device="cuda")  # 首次调用
   out = decode_one_step(model, x, kv_cache)

   x2 = torch.randn(1, 96, 768, device="cuda")  # 形状变化
   out2 = decode_one_step(model, x2, kv_cache)   # 触发 guard 检查

启用 guard 日志：

.. code-block:: bash

   TORCH_LOGS="+guard,+recompiles" python dynamic_example.py

你会在日志中看到类似以下的输出：

.. code-block:: text

   [__recompiles] Recompiling function decode_one_step
   [__recompiles]     triggered by the following guard failure:
   [__recompiles]     - local 'x' shape[1] == 64  # guard 期望 seq_len=64
   [__recompiles]     - actual shape[1] = 96      # 实际 seq_len=96
   [__recompiles]     Source: local 'x' (TensorVariable)

   [guard] GUARDS: local 'x' (TensorVariable)
   [guard]   - tensor 'x' device == cuda:0
   [guard]   - tensor 'x' dtype == torch.float32
   [guard]   - tensor 'x' dim == 3
   [guard]   - tensor 'x' size[0] == 1
   [guard]   - tensor 'x' size[1] == 64          # ← 触发重新编译的 guard
   [guard]   - tensor 'x' size[2] == 768
   [guard]   - tensor 'x' requires_grad == False

逐行分析这些日志：

1. ``Recompiling function decode_one_step`` — 告诉你哪个函数触发了重新编译
2. ``guard failure`` — 指出哪个 guard 失败了（shape[1] == 64 vs actual=96）
3. ``Source: local 'x' (TensorVariable)`` — guard 的来源是函数参数 ``x``
4. ``GUARDS`` 列表 — 列出了所有活跃 guard，包括设备、dtype、形状、梯度要求

修复方法：

- 如果 ``size[1]`` 应该动态变化，使用 ``torch._dynamo.mark_dynamic(x, 1)``
- 或在编译时使用 ``dynamic=True`` 让编译器知道形状是可变的

启用 ``TORCH_LOGS="+recompiles"`` 还可以看到重新编译的频率统计：

.. code-block:: text

   [__recompiles] Recompilation history for decode_one_step:
   [__recompiles]   Frame 0:  compiled_size=1, recompiles=12
   [__recompiles]   Unique shapes seen: [(1, 32, 768), (1, 48, 768),
   [__recompiles]                        (1, 64, 768), (1, 80, 768),
   [__recompiles]                        (1, 96, 768), ...]

如果 ``recompiles`` 数量很大（如 > 100），说明形状变化过于频繁，强烈建议使用 ``dynamic=True`` 或 ``mark_dynamic`` 。

.. note::

   在生产环境中，可以通过 ``torch._dynamo.config.cache_size_limit`` 限制缓存大小，避免因过多的特化版本导致内存爆炸。默认值为 128，即最多缓存 128 个特化版本。

生产中识别动态形状导致的重新编译
========================================================

在生产环境中，频繁的重新编译可能成为性能瓶颈。以下是一些实用的识别和监控方法。

方法 1：统计编译时间占比
--------------------------------

在训练循环中添加编译时间统计：

.. code-block:: python

   import time
   from torch._dynamo.utils import CompileTimeMeter

   # 训练循环中
   compile_times = []
   for step, batch in enumerate(dataloader):
       step_start = time.perf_counter()

       output = compiled_model(batch)

       step_end = time.perf_counter()
       step_time_ms = (step_end - step_start) * 1000

       # 检查是否发生了编译
       compile_stats = CompileTimeMeter.get_stats()
       if compile_stats:
           compile_times.append(compile_stats.total_compile_time)

       if step > 100 and sum(compile_times) / sum(step_time) > 0.1:
           print("警告: 编译时间占比超过 10%，请检查动态形状配置")

方法 2：使用 torch._dynamo 的缓存统计
--------------------------------------------

.. code-block:: python

   import torch._dynamo as dynamo

   def check_recompilation_rate(fn):
       """检查函数的重新编译率。"""
       before = dynamo.utils.get_compilation_stats()
       # 执行函数...
       after = dynamo.utils.get_compilation_stats()
       return {
           "unique_graphs": after.unique_graphs - before.unique_graphs,
           "recompiles": after.recompiles - before.recompiles,
           "compile_time": after.compile_time - before.compile_time,
       }

方法 3：使用 prometheus 指标
--------------------------------

对于生产服务，可以将编译统计导出到监控系统：

.. code-block:: python

   from prometheus_client import Histogram, Counter
   import torch._dynamo as dynamo

   compile_duration = Histogram(
       "torch_compile_duration_seconds",
       "torch.compile 编译持续时间",
       buckets=(1, 5, 10, 30, 60, 120, 300),
   )
   recompile_counter = Counter(
       "torch_compile_recompiles_total",
       "重新编译总次数",
   )

   def monitored_compile(model, x):
       """包装 torch.compile 并记录监控指标。"""
       stats_before = dynamo.utils.get_compilation_stats()
       compiled = torch.compile(model)
       compiled(x)
       stats_after = dynamo.utils.get_compilation_stats()

       recompiles = stats_after.recompiles - stats_before.recompiles
       compile_time = stats_after.compile_time - stats_before.compile_time

       if recompiles > 0:
           recompile_counter.inc(recompiles)
           compile_duration.observe(compile_time)

       return compiled

方法 4：设置编译告警
--------------------------

.. code-block:: python

   # 当编译时间超过阈值时抛出告警
   torch._dynamo.config.error_on_recompile = True
   # 或设置一个回调
   torch._dynamo.config.on_guard_error = lambda guard: print(
       f"Guard 失败: {guard}"
   )

.. tip::

   在生产环境中监控重新编译时，关注的是 **编译时间占比** 而非绝对编译次数。对于一个运行 7x24 小时的服务，偶尔的重新编译（每小时几次）是可以接受的。但如果编译时间占总运行时间的 5% 以上，就需要优化动态形状配置了。
