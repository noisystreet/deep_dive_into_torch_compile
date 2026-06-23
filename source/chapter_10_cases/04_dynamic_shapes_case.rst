.. _dynamic-shapes-case:

==============================
案例 4：Dynamic Shapes
==============================

这一节通过 NLP 场景的 padding 策略对比，展示如何在 torch.compile 中高效处理动态形状。

场景：变长序列批处理
==========================

在 NLP 中，不同句子的长度不同。批处理时有两种策略：

**策略 A：Padding 到固定长度**

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

**策略 B：Batch 内 Padding（不跨 batch 固定）**

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

有些操作的输出形状取决于输入数据（如 ``torch.nonzero``、``torch.unique``），这些操作会导致 "数据依赖的形状"——编译器无法在编译时确定输出形状。

.. code-block:: python

   @torch.compile
   def fn(x):
       mask = x > 0.5
       indices = torch.nonzero(mask)  # 输出形状取决于 x 的值
       return indices

这些操作会强制生成一个 graph break（因为输出形状无法在编译时确定），``torch.nonzero`` 之后的代码在 eager 模式下执行。

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
