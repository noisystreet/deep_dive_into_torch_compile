.. _training-workflow:

==============================
案例 3：训练工作流
==============================

这一节展示 torch.compile 在完整训练循环中的应用，包括前向传播、损失计算、反向传播和参数更新。

编译训练循环
==================

最简单的做法是编译模型：

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader

   model = MyModel().cuda()
   compiled_model = torch.compile(model)

   optimizer = optim.AdamW(compiled_model.parameters(), lr=1e-4)
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   for epoch in range(10):
       for batch in dataloader:
           x, y = batch[0].cuda(), batch[1].cuda()
           
           # 前向传播（编译后）
           output = compiled_model(x)
           
           # 损失计算（不在编译范围内）
           loss = nn.functional.cross_entropy(output, y)
           
           # 反向传播（编译后）
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()

这里只有模型的 ``forward`` 和 ``backward`` 被编译。损失函数通常很简单（如 ``cross_entropy``），编译它的收益不大，反而可能引入不必要的编译开销。

端到端编译
==============

如果损失函数也很复杂，可以将其纳入编译：

.. code-block:: python

   class TrainingStep(nn.Module):
       def __init__(self, model):
           super().__init__()
           self.model = model

       def forward(self, x, y):
           output = self.model(x)
           loss = nn.functional.cross_entropy(output, y)
           return loss

   training_step = TrainingStep(model).cuda()
   compiled_step = torch.compile(training_step)

   for batch in dataloader:
       x, y = batch[0].cuda(), batch[1].cuda()
       loss = compiled_step(x, y)
       loss.backward()
       ...

这样整个前向传播 + 损失计算被编译为一个单独的计算图，Scheduler 可以更好地融合前向和反向的操作。

Gradient Scaling 与混合精度
================================

使用 AMP（Automatic Mixed Precision）时：

.. code-block:: python

   scaler = torch.cuda.amp.GradScaler()

   for batch in dataloader:
       x, y = batch[0].cuda(), batch[1].cuda()
       
       with torch.cuda.amp.autocast():
           output = compiled_model(x)
           loss = nn.functional.cross_entropy(output, y)
       
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
       optimizer.zero_grad()

AMP 在编译下表现良好，因为 ``torch.cuda.amp.autocast()`` 会在图捕获之前设置好类型信息，编译后的 kernel 已经包含正确的数据类型转换。

如果遇到 AMP 下的精度问题，可以调整 matmul 精度：

.. code-block:: python

   torch.set_float32_matmul_precision("medium")

   @torch.compile
   def fn(x):
       ...

Gradient Accumulation
=========================

梯度累积用于模拟更大的 batch size：

.. code-block:: python

   compiled_model = torch.compile(model)

   accumulation_steps = 4
   optimizer.zero_grad()

   for i, batch in enumerate(dataloader):
       x, y = batch[0].cuda(), batch[1].cuda()
       
       output = compiled_model(x)
       loss = nn.functional.cross_entropy(output, y) / accumulation_steps
       loss.backward()

       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()

torch.compile 不需要为梯度累积做特殊处理——每次 ``forward/backward`` 是独立的编译调用。

检查点（Checkpointing）
==============================

使用梯度检查点减少显存占用：

.. code-block:: python

   from torch.utils.checkpoint import checkpoint

   class MyModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.layer1 = nn.Linear(1024, 1024)
           self.layer2 = nn.Linear(1024, 1024)
           self.layer3 = nn.Linear(1024, 1024)

       def forward(self, x):
           x = checkpoint(self.layer1, x)  # 不保存中间结果
           x = self.layer2(x)              # 保存中间结果
           x = self.layer3(x)
           return x

检查点和 torch.compile 配合使用时需要注意：``checkpoint`` 内部的函数也会被编译。如果检查点内部的函数很复杂，编译开销可能超过重计算节省的显存。可以通过 ``torch.compiler.disable`` 禁用检查点内部的编译：

.. code-block:: python

   @torch.compiler.disable
   def checkpointed_layer(x):
       return self.layer1(x)

   x = checkpoint(checkpointed_layer, x)

与学习率调度器配合
========================

.. code-block:: python

   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

   for epoch in range(100):
       for batch in dataloader:
           x, y = batch[0].cuda(), batch[1].cuda()
           loss = compiled_model(x, y)
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()
       scheduler.step()

学习率调度器在 CPU 上运行，不会影响编译图。

常见问题与调试
==================

**"RuntimeError: backward() was called after optimizer.step()"**。如果编译模型在 ``backward()`` 上出错，检查是否在 ``backward()`` 之前调用了 ``optimizer.step()``。这在编译模式下更常见，因为编译后的图可能改变了操作顺序。

**显存泄漏**。如果编译后的训练循环出现显存泄漏，尝试：

.. code-block:: python

   # 训练循环中定期清理缓存
   import gc
   torch.cuda.empty_cache()
   gc.collect()

**编译时间过长**。如果模型很大（如 >10 亿参数），编译时间可能超过 10 分钟。使用 ``progressive`` 模式：

.. code-block:: bash

   TORCH_COMPILE_MODE=progressive python train.py

训练循环编译决策流程
=============================

当决定在训练循环中使用 torch.compile 时，需要根据模型结构、训练规模和硬件配置做出选择。下面的流程图可以帮助你做出决策：

.. mermaid::

   flowchart TD
       Start["开始: 选择训练配置"] --> Q1{"模型是否\n有控制流?"}

       Q1 -->|"是 (if/for 依赖 tensor)"| Q1a["考虑 partial compile\n编译核心 nn.Module\n而非整个训练步"]
       Q1 -->|"否"| Q2{"损失函数\n是否简单?"}

       Q2 -->|"是 (cross_entropy, mse)"| Q3{"使用 AMP?"}
       Q2 -->|"否 (自定义复杂损失)"| Q2a["将损失纳入编译\nTrainingStep 包装"]

       Q3 -->|"是"| Q4{"显存是否\n充足?"}
       Q3 -->|"否"| Q5{"模型 >1B\n参数?"}

       Q4 -->|"充足"| Q6{"需要最大\n吞吐量?"}
       Q4 -->|"紧张"| Q6a["使用 default 模式\n避免 autotune 显存开销"]

       Q5 -->|"是"| Q5a["使用 default 模式\n编译时间可能很长\n考虑 progressive"]
       Q5 -->|"否"| Q6

       Q6 -->|"是"| Q6b["尝试 max-autotune\n注意首次编译时间"]
       Q6 -->|"否"| Q7{"梯度累积?"}

       Q7 -->|"是"| Rec["default 模式\n梯度累积与 compile\n无需特殊配置"]
       Q7 -->|"否"| Rec2["default 或\nreduce-overhead\n取决于训练稳定性"]

       Q1a --> Q2
       Q2a --> Q3
       Q6a --> CompileDone["开始训练"]
       Q5a --> CompileDone
       Q6b --> CompileDone
       Rec --> CompileDone
       Rec2 --> CompileDone

       style Start fill:#e8f5e9,stroke:#2e7d32
       style CompileDone fill:#e3f2fd,stroke:#1565c0

编译模式选择：default vs reduce-overhead
=================================================

在训练场景中，``default`` 和 ``reduce-overhead`` 模式的选择需要综合考虑吞吐量和数值精度。

``default`` 模式的特点
---------------------------

- 不做额外的 kernel 融合，保持与 eager 模式最接近的计算行为
- 兼容性最好，几乎不会引入精度问题
- 训练加速比通常在 1.3x - 2.0x 之间
- 编译时间相对较短（ResNet50 约 30-60 秒）

``reduce-overhead`` 模式在训练中的问题
------------------------------------------------

``reduce-overhead`` 在推理场景中效果显著，但在训练中需要谨慎使用：

1. **CUDA Graph 与反向传播的冲突**。CUDA Graph 要求计算图在捕获后完全固定。但训练时，反向传播的计算图依赖于前向传播的输出——这本身是确定的，但 autograd graph 的构建涉及一些 Python 层面的操作，可能导致 graph break。

2. **梯度更新模式变化**。CUDA Graph 捕获后，权重更新是通过 ``optimizer.step()`` 在 Python 层面完成的，不在 Graph 内。这意味着 CUDA Graph 只覆盖 ``forward + backward`` 部分，**没有覆盖 optimizer step**。

3. **数值精度差异**。由于 kernel 融合改变了浮点运算的顺序（如 A+B+C 可能变为 (A+B)+C 或 A+(B+C)），``reduce-overhead`` 模式可能导致训练结果与 eager 模式略有不同。对于对精度敏感的训练任务，这种差异可能影响模型收敛。

.. list-table::
   :header-rows: 1

   * - 对比维度
     - default
     - reduce-overhead
     - max-autotune
   * - 训练加速比
     - 1.3x - 2.0x
     - 1.0x - 1.5x (收益有限)
     - 1.5x - 2.5x
   * - 编译时间
     - 短 (30-60s)
     - 中等 (60-120s)
     - 长 (120s+)
   * - 数值精度
     - 与 eager 一致
     - 可能有微小差异
     - 可能有微小差异
   * - 显存占用
     - 与 eager 相近
     - 略高 (Graph 缓存)
     - 显著增高 (autotune)
   * - 适用场景
     - 通用训练
     - 仅推理
     - 追求极致吞吐

.. tip::

   对于训练任务，**从 ``default`` 模式开始**。如果模型收敛正常，可以尝试 ``max-autotune`` 提升吞吐量。``reduce-overhead`` 模式主要为推理设计，将其用于训练时务必验证数值准确性。

Gradient Scaling 与编译图的交互
============================================

使用 AMP（Automatic Mixed Precision）时，``GradScaler`` 的作用是防止 fp16 梯度下溢。编译后的计算图与 ``GradScaler`` 的交互有一些微妙的细节。

交互流程
------------

.. code-block:: python

   scaler = torch.cuda.amp.GradScaler()

   for batch in dataloader:
       with torch.cuda.amp.autocast():
           output = compiled_model(x)    # 编译图 1: 前向（fp16 计算）
           loss = criterion(output, y)   # 编译图 2: 损失（fp32 计算）

       scaler.scale(loss).backward()     # 编译图 3: 反向（缩放后的梯度）
       # ^ 这里 loss.backward() 会触发编译图 3 的执行
       #   scaler.scale(loss) 将 loss 乘以 scale 因子
       #   反向传播在缩放后的 loss 上计算梯度

       scaler.step(optimizer)            # 解缩梯度 + 参数更新
       scaler.update()                   # 调整 scale 因子

关键点在于 ``scaler.scale(loss).backward()`` 是一个分两步的调用：

1. ``scaler.scale(loss)`` — 在 Python 层面将 loss 乘以一个标量，这是一个简单的逐元素乘法
2. ``.backward()`` — 触发编译后的反向传播图

由于 ``scale`` 操作在编译图之外，它不会影响编译图的捕获。但是，**scale 因子的变化会导致编译图需要重新捕获** 吗？答案是否定的——因为 ``scale`` 操作不在编译图内，scale 因子的变化不会触发 guard 失败。

不同 scale 因子下的编译图行为：

.. code-block:: text

   Step 1: scale=1024
       loss_fp32 = compiled_forward(x)        # 编译执行
       scaled_loss = loss_fp32 * 1024         # Python 层面
       compiled_backward(scaled_loss)          # 编译执行（梯度值为 1024x）

   Step 2: scale=2048 (scale 更新)
       loss_fp32 = compiled_forward(x)        # 同一个编译图
       scaled_loss = loss_fp32 * 2048         # Python 层面
       compiled_backward(scaled_loss)          # 同一个编译图

   结论: scale 因子变化 *不会* 触发重新编译

无限训练 (Infinity Training) 中的编译策略
==================================================

对于长时间运行的训练任务（如大模型预训练），编译策略需要额外考虑：

- **累积编译时间**：如果模型在训练过程中因数据分布变化而频繁重新编译，累积的编译时间可能抵消编译带来的加速收益
- **动态数据形状**：图像大小变化、序列长度分布变化都会触发重新编译
- **分布式环境**：多 GPU 场景下的编译时间线性增长（详见 :ref:`multi-gpu-scenarios`）

示例代码
============

完整的 compiled 训练循环示例见 ``examples/training_loop.py``。该文件包含：

- ``SmallResNet``：一个用于演示的简化 ResNet 模型
- ``CompiledTrainer``：封装了完整训练流程的 Trainer 类，支持多种编译模式和 AMP
- ``train_with_gradient_accumulation``：展示了梯度累积与 torch.compile 的配合方式
- ``run_benchmark``：训练模式下不同编译模式的吞吐量对比

基准测试工具见 ``examples/benchmark_utils.py``。该文件提供：

- ``timing_median``：中位数计时（排除异常值）
- ``BenchmarkRunner``：多模式自动对比运行器
- ``warmup_cuda``：CUDA 预热函数
- ``cuda_memory_snapshot``：显存快照
- ``format_speedup_table``：加速比表格格式化
