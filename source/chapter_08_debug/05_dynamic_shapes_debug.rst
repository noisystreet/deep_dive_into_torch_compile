.. _dynamic-shapes-debug:

=========================
Dynamic Shapes 调试
=========================

第 3.8 节介绍了符号形状（symbolic shapes）的原理——``ShapeEnv`` 、 ``SymNode`` 以及 ``dynamic=True`` 的工作方式。这一节聚焦 **如何诊断和优化 **Dynamic Shapes 场景中的 guard 失败与频繁重编译问题。

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

** 形状变化被过度 guard** 。以下代码会导致每次不同长度的输入都重新编译：

.. code-block:: python

   @torch.compile
   def fn(x):
       return x * 2

   fn(torch.randn(100))  # 编译，guard: x.shape[0] == 100
   fn(torch.randn(200))  # guard 失败，重新编译
   fn(torch.randn(300))  # guard 失败，重新编译

**数据集形状不一致 ** 。最常见的原因——训练/验证集的 batch size 或序列长度不一致。

**数据预处理引入了动态形状 ** 。如 ``torch.unique`` 、 ``torch.nonzero`` 等操作的输出形状取决于输入数据。

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

Dynamic Shapes 调试工作流
================================

遇到动态形状导致的性能问题时，可以按照以下决策树逐步排查：

.. mermaid::

   graph TD
       A["遇到性能问题"] --> B{"检查 guard 命中率<br/>TORCH_LOGS=+guards"}
       B -->|"频繁出现 Guard failed"| C{"是否频繁重编译？<br/>检查 compile 计数器"}
       B -->|"Guard 全部命中"| D["排查其他原因<br/>（见 8.4 节 Profiling）"]
       C -->|"是"| E{"确定动态维度"}
       C -->|"否"| F["检查 graph break<br/>（见 8.1 节）"]
       E --> G{"维度范围是否可控？"}
       G -->|"是"| H["使用 dynamic=True<br/>或 mark_dynamic"]
       G -->|"否"| I["使用 padding 统一形状"]
       H --> J["验证编译次数减少"]
       I --> J
       J --> K{"性能是否达标？"}
       K -->|"是"| L["完成"]
       K -->|"否"| M["调整 cache_size_limit<br/>或优化策略"]

静态形状 vs 动态形状的 Guard 行为对比
==============================================

下面的 Mermaid 图展示了静态形状和动态形状下 guard 行为的本质区别：

.. mermaid::

   graph LR
       subgraph Static["静态形状 (默认)"]
           S1["输入 shape=(32, 784)"] --> S2["生成 Guard:<br/>x.shape[0] == 32<br/>x.shape[1] == 784"]
           S2 --> S3{"下次输入<br/>shape=(64, 784)?"}
           S3 -->|"Guard 失败"| S4["重新编译"]
           S3 -->|"shape=(32, 784)"| S5["命中缓存"]
       end

       subgraph Dynamic["动态形状 (dynamic=True)"]
           D1["输入 shape=(32, 784)"] --> D2["生成 Guard:<br/>x.shape[0] >= 1<br/>x.shape[1] == 784"]
           D2 --> D3{"下次输入<br/>shape=(64, 784)?"}
           D3 -->|"Guard 通过"| D4["命中缓存<br/>无需重新编译"]
           D3 -->|"shape=(32, 784)"| D4
       end

       Static -->|"形状变化触发<br/>大量重编译"| Dynamic

静态形状模式下，Dynamo 将每个维度的具体数值写入 guard，任何数值变化都触发重编译。动态形状模式下，guard 只检查维度的符号约束（如 ``s0 >= 1`` ），形状范围内的变化不会触发重编译。

动态形状编译优化策略
============================

动态形状和静态形状各有优劣，选择哪种策略取决于具体场景。

Dynamic Shapes vs Static Shapes 的权衡
-----------------------------------------------

.. list-table:: 动态形状 vs 静态形状对比
   :header-rows: 1

   * - 维度
     - 静态形状 (默认)
     - 动态形状 (dynamic=True)
   * - 编译次数
     - 形状变一次编译一次
     - 形状在约束范围内不变
   * - Kernel 性能
     - 针对固定形状优化，性能最优
     - 使用通用 tiling，性能略低
   * - 适用场景
     - 训练/推理形状固定
     - 推理 batch size 可变、序列长度可变
   * - Guard 类型
     - 具体数值 guard
     - 符号约束 guard
   * - 编译器优化空间
     - 大（可做形状相关的常量折叠）
     - 中（需保留符号表达式）

核心权衡： **动态形状减少了重编译次数，但生成的 kernel 不如静态形状高效 ** 。如果重编译的性能损失大于 kernel 性能下降的损失，就应该使用动态形状。

使用 cache_size_limit 控制编译缓存
--------------------------------------------

``torch._dynamo.config.cache_size_limit`` 控制 Dynamo 为每个函数缓存的编译结果数量：

.. code-block:: python

   import torch._dynamo.config as config

   # 默认值 64，适用于形状变化有限的场景
   config.cache_size_limit = 64

   # 如果形状变化非常多（如 NLP 中每个 batch 的序列长度都不同）
   # 可以增大限制以避免频繁驱逐旧缓存
   config.cache_size_limit = 256

   # 如果形状基本固定，减小限制可以节省内存
   config.cache_size_limit = 8

当缓存被占满时，Dynamo 会驱逐最久未使用的编译结果（LRU 策略）。如果模型的形状种类超过 ``cache_size_limit`` ，就会发生"反复编译-被驱逐-再编译"的现象。

.. tip::

**如何判断 cache_size_limit 是否过小？ **
   如果在 ``TORCH_LOGS="+dynamo"`` 日志中看到 ``cache miss`` 和 ``recompiling`` 交替出现，且编译总数（compile counter）远大于输入的形状种类数，说明 ``cache_size_limit`` 可能过小导致缓存被频繁驱逐。

Inductor 如何应对符号形状
----------------------------------------

当启用动态形状时，Inductor 的 Scheduler 和 Triton codegen 会以符号方式处理形状信息：

.. code-block:: python

   # 动态形状下的 Triton kernel 会使用符号变量
   # 而不是固定的数值常量
   @triton.jit
   def triton_kernel(
       in_ptr0, out_ptr0,
       xnumel, rnumel,
       XBLOCK: tl.constexpr, RBLOCK: tl.constexpr,
   ):
       xnumel = 1024  # 静态形状：固定数值
       # vs
       xnumel = sym_x  # 动态形状：符号变量

符号形状对 Inductor 的影响：

- **Tiling 参数不可静态确定 ** ：Scheduler 基于符号表达式计算 block 大小，可能低于最优值
- **循环边界使用符号变量 ** ：生成的 Triton kernel 需要在运行时计算循环次数
- **部分优化失效** ：常量折叠、形状相关的 dead code elimination 在符号形状上受限

.. code-block:: python

   # 查看 Inductor 如何处理动态形状
   torch._inductor.config.debug = True

   @torch.compile(dynamic=True)
   def fn(x):
       return torch.sin(x) + torch.cos(x)

   # 日志中会显示符号形状的处理过程
   # [inductor] Scheduler 使用符号变量 s0 调度
   # [inductor] Triton codegen 使用 s0 生成循环边界

Padding vs 动态形状：决策矩阵
--------------------------------------------

在某些场景下，Padding（将输入填充到固定形状）比动态形状更优。以下决策矩阵帮助选择：

.. list-table:: Padding vs 动态形状决策矩阵
   :header-rows: 1

   * - 条件
     - 推荐策略
     - 原因
   * - 形状变化范围 < 2x
     - Padding 到最大形状
     - 计算浪费少，kernel 性能最优
   * - 形状变化范围 > 10x
     - 动态形状
     - Padding 会造成大量无效计算
   * - 模型以 pointwise 操作为主
     - Padding
     - Pointwise 的复杂度与数据量线性相关，padding 浪费有限
   * - 模型以矩阵乘法为主
     - 动态形状
     - Padding 改变矩阵维度可能破坏 tile 对齐
   * - 训练场景
     - 优先使用动态形状
     - Padding 可能引入梯度计算中的无效贡献
   * - 推理场景
     - 视延迟要求而定：低延迟用 padding
     - Padding 的额外计算可能增加推理延迟

.. code-block:: python

   # Padding 策略示例：将序列填充到固定长度
   def pad_to_fixed_length(x, max_len=512):
       batch_size, seq_len, hidden = x.shape
       if seq_len >= max_len:
           return x[:, :max_len, :]
       padding = torch.zeros(batch_size, max_len - seq_len, hidden,
                             device=x.device, dtype=x.dtype)
       return torch.cat([x, padding], dim=1)

   # 动态形状策略示例：使用 mark_dynamic
   import torch._dynamo as dynamo

   x = torch.randn(4, 128, 768)
   dynamo.mark_dynamic(x, 1)  # 标记序列长度为动态维度

   @torch.compile
   def transformer_block(x):
       # attention, feed forward 等操作
       return x

动态形状调试实战
======================

以下是一个完整的动态形状调试过程，从发现问题到验证修复。

问题场景：训练中的序列分类模型
----------------------------------------

假设我们有一个文本分类模型，训练时每个 batch 的序列长度不同：

.. code-block:: python

   import torch
   import torch._dynamo.utils as dynamo_utils
   import time

   class TextClassifier(torch.nn.Module):
       def __init__(self, vocab_size=10000, hidden=256, num_classes=2):
           super().__init__()
           self.embedding = torch.nn.Embedding(vocab_size, hidden)
           self.fc = torch.nn.Linear(hidden, num_classes)

       def forward(self, x):
           # x shape: (batch, seq_len)
           x = self.embedding(x)              # (batch, seq_len, hidden)
           x = x.mean(dim=1)                  # (batch, hidden)
           x = self.fc(x)                     # (batch, num_classes)
           return x

   model = TextClassifier().cuda()
   compiled_model = torch.compile(model)

   # 模拟不同长度的序列
   data = [
       torch.randint(0, 10000, (4, 128), device='cuda'),
       torch.randint(0, 10000, (4, 200), device='cuda'),
       torch.randint(0, 10000, (4, 96),  device='cuda'),
       torch.randint(0, 10000, (4, 256), device='cuda'),
       torch.randint(0, 10000, (4, 128), device='cuda'),
       torch.randint(0, 10000, (4, 64),  device='cuda'),
   ]

   # 运行并观察编译统计
   for i, x in enumerate(data):
       out = compiled_model(x)
       stats = dynamo_utils.counters["dynamo"]
       print(f"Batch {i}: shape={x.shape}, "
             f"compile={stats.get('compile', 0)}, "
             f"recompile={stats.get('recompile', 0)}, "
             f"guard_fail={stats.get('guard_fail', 0)}")

输出示例：

.. code-block:: text

   Batch 0: shape=(4, 128), compile=1, recompile=0, guard_fail=0
   Batch 1: shape=(4, 200), compile=2, recompile=1, guard_fail=1
   Batch 2: shape=(4, 96),  compile=3, recompile=2, guard_fail=2
   Batch 3: shape=(4, 256), compile=4, recompile=3, guard_fail=3
   Batch 4: shape=(4, 128), compile=5, recompile=4, guard_fail=4
   Batch 5: shape=(4, 64),  compile=6, recompile=5, guard_fail=5

可以看到几乎每次输入都触发 guard 失败和重新编译。

Step 1：使用 TORCH_LOGS 诊断
---------------------------------------

.. code-block:: bash

   TORCH_LOGS="+guards,+dynamic" python train.py

日志输出：

.. code-block:: text

   [guards] Guard failed: x.shape[1] == 128
   [guards] 触发重新编译...
   [dynamic] 创建符号变量 s0 = x.shape[1]
   [dynamic] 约束: s0 >= 1, s0 <= 1024
   [dynamic] Guard expr: s0 == 128

结论：序列长度（第 1 维）被静态 guard 了，每次变化都触发重编译。

Step 2：应用修复
------------------------

.. code-block:: python

   # 修复方式一：使用 mark_dynamic 标记动态维度
   import torch._dynamo as dynamo

   model = TextClassifier().cuda()

   # 在每次调用前标记动态维度
   for x in data:
       dynamo.mark_dynamic(x, 1)  # 标记序列长度为动态
       out = torch.compile(model)(x)

   # 修复方式二：使用 dynamic=True
   compiled_model = torch.compile(model, dynamic=True)

   # 修复方式三：只标记特定的动态维度（更精确）
   @torch.compile
   def forward_with_dynamic(model, x):
       dynamo.mark_dynamic(x, 1)
       return model(x)

Step 3：验证修复效果
----------------------------

.. code-block:: python

   # 重置计数器
   dynamo_utils.counters.clear()

   compiled_model = torch.compile(model, dynamic=True)

   for i, x in enumerate(data):
       out = compiled_model(x)
       stats = dynamo_utils.counters["dynamo"]
       print(f"Batch {i}: shape={x.shape}, "
             f"compile={stats.get('compile', 0)}, "
             f"recompile={stats.get('recompile', 0)}, "
             f"guard_fail={stats.get('guard_fail', 0)}")

修复后的输出：

.. code-block:: text

   Batch 0: shape=(4, 128), compile=1, recompile=0, guard_fail=0
   Batch 1: shape=(4, 200), compile=1, recompile=0, guard_fail=0
   Batch 2: shape=(4, 96),  compile=1, recompile=0, guard_fail=0
   Batch 3: shape=(4, 256), compile=1, recompile=0, guard_fail=0
   Batch 4: shape=(4, 128), compile=1, recompile=0, guard_fail=0
   Batch 5: shape=(4, 64),  compile=1, recompile=0, guard_fail=0

现在只编译了一次，所有形状的变化都命中同一缓存。

Step 4：性能对比
------------------------

.. code-block:: python

   # 对比修复前后的训练时间
   def benchmark(model, data, n_iter=100):
       model = torch.compile(model)
       # 预热
       for _ in range(10):
           for x in data:
               model(x)
       torch.cuda.synchronize()

       start = time.time()
       for _ in range(n_iter):
           for x in data:
               model(x)
       torch.cuda.synchronize()
       elapsed = time.time() - start

       stats = dynamo_utils.counters["dynamo"]
       total_compiles = stats.get("compile", 0) + stats.get("recompile", 0)
       return elapsed, total_compiles

   # 静态形状（默认设置）
   stats_before = dynamo_utils.counters["dynamo"].copy()
   time_before, compiles_before = benchmark(TextClassifier().cuda(), data)

   # 动态形状
   stats_after = dynamo_utils.counters["dynamo"].copy()
   time_after, compiles_after = benchmark(
       TextClassifier().cuda(), data, dynamic=True)

   print(f"静态形状: {time_before:.2f}s, 编译 {compiles_before} 次")
   print(f"动态形状: {time_after:.2f}s, 编译 {compiles_after} 次")
   print(f"加速比: {time_before / time_after:.2f}x")

.. tip::

   **调试动态形状问题的快速 checklist** ：
   1. 检查 ``TORCH_LOGS="+guards"`` 的 guard failed 频率
   2. 使用 ``dynamo_utils.counters`` 查看编译统计
   3. 标记明确会变化的维度为动态
   4. 验证编译次数是否减少
   5. 权衡：如果编译时间 < kernel 性能损失，保持静态形状

如果编译时间占比超过 20%，且重新编译次数超过 100，就需要优化动态形状处理了。
