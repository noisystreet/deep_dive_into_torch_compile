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
