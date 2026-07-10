.. _multi-gpu-scenarios:

==============================
案例 5：多 GPU 场景
==============================

torch.compile 与分布式训练框架（DDP、FSDP）的配合是多 GPU 训练的关键。这一节介绍常见的配合模式和最佳实践。

DDP + torch.compile
=========================

Data Distributed Parallel（DDP）是最常用的分布式训练方式：

.. code-block:: python

   import torch
   import torch.distributed as dist
   import torch.nn.parallel as DDP
   from torch.utils.data.distributed import DistributedSampler

   # 初始化进程组
   dist.init_process_group(backend="nccl")
   local_rank = dist.get_rank()
   torch.cuda.set_device(local_rank)

   # 模型
   model = MyModel().cuda(local_rank)
   compiled_model = torch.compile(model)          # 先编译
   ddp_model = DDP(compiled_model)                # 再 DDP

   # DataLoader
   sampler = DistributedSampler(dataset)
   dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

   optimizer = torch.optim.AdamW(ddp_model.parameters())

   for epoch in range(10):
       sampler.set_epoch(epoch)
       for x, y in dataloader:
           x, y = x.cuda(local_rank), y.cuda(local_rank)
           
           output = ddp_model(x)
           loss = nn.functional.cross_entropy(output, y)
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()

**编译和 DDP 的顺序很重要 ** ：先 ``compile`` 再 ``DDP`` 。这样 DDP 包装的是编译后的模型，编译器的优化在 DDP 之前生效。如果反过来，DDP 的 hook 会干扰编译器的图捕获。

**每个 GPU 独立编译** 。在 DDP 中，每个 rank 独立运行 ``compile`` ，因为每个 rank 看到的 ``example_inputs`` 形状相同，编译结果也相同。由于磁盘缓存的存在，rank 0 编译后，其他 rank 可以直接使用缓存，避免重复编译。

FSDP + torch.compile
=========================

Fully Sharded Data Parallel（FSDP）分片模型参数和优化器状态：

.. code-block:: python

   from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
   from torch.distributed.fsdp.fully_sharded_data_parallel import (
       CPUOffload, BackwardPrefetch,
   )

   model = MyModel().cuda(local_rank)
   compiled_model = torch.compile(model)  # 先编译

   fsdp_model = FSDP(
       compiled_model,
       device_id=local_rank,
       cpu_offload=CPUOffload(offload_params=True),
       backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
   )

FSDP 的注意点：

- FSDP 在 forward 之前和 backward 之后执行 all-gather/reduce-scatter 通信。编译时可以看到这些通信操作在图中。
- 如果 FSDP 的通信操作导致 graph break，是因为通信操作无法被 Dynamo 追踪。通常一个 FSDP 包装的 layer 在 forward 时会有一个 graph break。

.. code-block:: text

   未 FSDP 的图:
       [layer1 → relu → layer2 → relu → layer3] → (无 graph break)

   FSDP 的图:
       [all-gather → layer1 → relu → reduce-scatter]  ← graph break
       [all-gather → layer2 → relu → reduce-scatter]  ← graph break
       [all-gather → layer3 → reduce-scatter]          ← graph break

这些 graph break 是预期的，不是性能问题。它们是因为 FSDP 的通信操作必须在 Python 层面触发。

Tensor Parallel + torch.compile
=====================================

Tensor Parallel（TP）将单个层的参数分片到多个 GPU。torch.compile 对 TP 的支持取决于 TP 的实现方式：

.. code-block:: python

   # 使用 PyTorch 原生的 TP API
   from torch.distributed.tensor.parallel import (
       parallelize_module, ColwiseParallel, RowwiseParallel,
   )

   model = MyModel().cuda(local_rank)
   tp_model = parallelize_module(
       model,
       device_mesh=DeviceMesh("cuda", list(range(world_size))),
       parallelize_plan={
           "linear1": ColwiseParallel(),
           "linear2": RowwiseParallel(),
       },
   )
   compiled_tp = torch.compile(tp_model)

PyTorch 原生 TP API 生成的通信操作（all-reduce、reduce-scatter）可以被 Dynamo 追踪，不会导致 graph break。如果使用自定义的 TP 通信（如手写的 ``all_reduce`` ），可能会导致 graph break。

流水线并行 + torch.compile
=================================

流水线并行（Pipeline Parallel）将模型的不同层放在不同的 GPU 上。torch.compile 在每个 GPU 上独立编译该 GPU 上的子图：

.. code-block:: python

   class PipelineStage(nn.Module):
       def __init__(self, layers):
           super().__init__()
           self.layers = layers

       def forward(self, x):
           for layer in self.layers:
               x = layer(x)
           return x

   # GPU 0 上的前几层
   stage1 = PipelineStage(model.layers[:10]).cuda(0)
   compiled_stage1 = torch.compile(stage1)

   # GPU 1 上的后几层
   stage2 = PipelineStage(model.layers[10:]).cuda(1)
   compiled_stage2 = torch.compile(stage2)

每个 stage 独立编译，stage 之间的通信由流水线并行的调度器管理。由于编译是每个 stage 独立进行的，编译时间不会因为 GPU 数量增加而线性增长。

性能基准测试
================

多 GPU 场景下的预期加速比：

.. list-table::
   :header-rows: 1

   * - 配置
     - 单 GPU 加速比
     - 多 GPU 扩展效率
   * - eager + DDP
     - 1.0x
     - ~90%（4 GPU）
   * - compile + DDP
     - 1.5x - 2.0x
     - ~85%（4 GPU）
   * - compile + FSDP
     - 1.3x - 1.8x
     - ~80%（8 GPU）
   * - compile + TP
     - 1.2x - 1.6x
     - ~75%（8 GPU）

注意：多 GPU 场景下，通信开销可能部分抵消编译带来的性能提升，特别是在小 batch 场景。

常见问题
============

**编译时间与 GPU 数量线性增长 ** 。如果 N 个 GPU 各自独立编译，总编译时间 = 单 GPU 编译时间 x N。通过磁盘缓存可以解决：rank 0 先编译，其他 rank 从缓存加载。

**编译结果不一致 ** 。如果不同 rank 的 GPU 型号不同（如 A100 和 H100 混部），编译结果不能共享，因为 Triton 编译结果包含 GPU 架构特定的优化。

**OOM 在编译时发生 ** 。 ``max-autotune`` 模式可能需要额外的显存（用于存储多个 kernel 变体）。如果编译时 OOM，尝试：

.. code-block:: python

   # 限制 autotune 的显存使用
   torch._inductor.config.autotune_in_subproc = True
   # 或在子进程中执行 autotune
   torch._inductor.config.coordinate_descent_tuning = False

DDP + Compile 工作流跨 Rank 示意图
============================================

下面的流程展示了 DDP 与 torch.compile 结合时，在多个 rank 上的协作方式：

.. mermaid::

   sequenceDiagram
       participant Rank0 as Rank 0 (主)
       participant Rank1 as Rank 1
       participant Rank2 as Rank 2
       participant Disk as 磁盘缓存

       Note over Rank0,Rank2: 步骤 1: 模型创建
       Rank0->>Rank0: 创建模型
       Rank1->>Rank1: 创建模型 (相同结构)
       Rank2->>Rank2: 创建模型 (相同结构)

       Note over Rank0,Rank2: 步骤 2: 编译 (各 rank 独立)
       Rank0->>Rank0: torch.compile(model)
       Note over Rank0: 编译中... (30-120s)
       Rank0->>Disk: 写入编译缓存 (.torch_compile_cache)

       Rank1->>Disk: 读取编译缓存 (0-5s)
       Rank1->>Rank1: 从缓存加载编译结果

       Rank2->>Disk: 读取编译缓存 (0-5s)
       Rank2->>Rank2: 从缓存加载编译结果

       Note over Rank0,Rank2: 步骤 3: DDP 包装
       Rank0->>Rank0: DDP(compiled_model)
       Rank1->>Rank1: DDP(compiled_model)
       Rank2->>Rank2: DDP(compiled_model)

       Note over Rank0,Rank2: 步骤 4: 训练循环
       Rank0->>Rank0: forward + backward (编译后)
       Rank1->>Rank1: forward + backward (编译后)
       Rank2->>Rank2: forward + backward (编译后)

       Rank0->>Rank0: optimizer.step() (梯度同步)
       Rank1->>Rank1: optimizer.step()
       Rank2->>Rank2: optimizer.step()

关键观察：每个 rank 独立编译，但通过共享磁盘缓存，非 rank 0 的 GPU 可以直接加载 rank 0 的编译结果。这要求所有 rank 的 GPU 架构一致。

分布式 Tensor（DTensor）与 torch.compile
=========================================================

DTensor 是 PyTorch 分布式张量的核心抽象，它将一个逻辑张量分片到多个设备上。 ``torch.compile`` 对 DTensor 的支持是 PyTorch 2.x 的重要特性。

DTensor 的基本概念
------------------------

.. code-block:: python

   from torch.distributed._tensor import DTensor, Replicate, Shard

   # 创建一个分片到 4 个 GPU 的张量
   # Replicate: 每个 GPU 持有完整副本
   # Shard(dim): 在指定维度上切分

   # 示例: 在 batch 维度上切分
   dt = DTensor.from_local(
       local_tensor,               # 当前 rank 上的分片
       device_mesh=device_mesh,    # 设备网格
       placements=[Shard(0)],      # 在 dim=0 上切分
   )

DTensor + Compile 的优势
-------------------------------------------

.. code-block:: python

   from torch.distributed._tensor import DTensor
   import torch.nn as nn

   class ShardedLinear(nn.Module):
       """使用 DTensor 的分布式线性层。"""
       def __init__(self, in_features, out_features, device_mesh):
           super().__init__()
           self.weight = nn.Parameter(
               torch.randn(out_features, in_features)
           )
           self.device_mesh = device_mesh

       def forward(self, x):
           # 将权重转为 DTensor 并行计算
           w_dt = DTensor.from_local(
               self.weight, self.device_mesh, [Shard(0)]
           )
           x_dt = DTensor.from_local(
               x, self.device_mesh, [Shard(1)]
           )
           return torch.mm(x_dt, w_dt.T)

   # 编译模型
   model = ShardedLinear(1024, 1024, device_mesh).cuda()
   compiled = torch.compile(model)

DTensor 与 torch.compile 配合时，编译器能够识别 DTensor 的通信模式并生成高效的融合代码。编译器会在必要时插入 ``all_reduce`` 、 ``all_gather`` 等通信操作，这些操作可以被 Dynamo 追踪，**不会导致 graph break** 。

.. note::

   DTensor + torch.compile 依赖 PyTorch 2.1+ 的分布式张量实现。在早期版本中，DTensor 的通信操作可能被 Dynamo 视为无法追踪的操作而导致 graph break。建议升级到最新的 PyTorch 版本以获得最佳支持。

编译时 vs 运行时权衡
==========================================

在分布式场景中，torch.compile 引入了一个独特的权衡： **编译时间在 GPU 数量维度上线性扩展** （每个 GPU 独立编译），而优化效果也是每 GPU 独立的。

权衡矩阵
----------------

.. list-table::
   :header-rows: 1

   * - 场景
     - GPU 数量
     - 编译时间 (每 GPU)
     - 总编译时间
     - 运行时加速
     - 净收益
   * - 单卡
     - 1
     - 120s
     - 120s
     - 1.5x
     - 高
   * - DDP (8卡)
     - 8
     - 5s (缓存命中)
     - 120s + 5s
     - 1.5x + 8x 扩展
     - 很高
   * - FSDP (64卡)
     - 64
     - 1s (共享缓存)
     - 120s + 1s
     - 1.3x + 64x 扩展
     - 很高
   * - 流水线并行 (8卡)
     - 8
     - 30s (每 stage)
     - 8 x 30s = 240s
     - 1.4x + 8x 扩展
     - 高
   * - TP + PP (32卡)
     - 32
     - 60s
     - 32 x 60s = 1920s
     - 1.2x + 32x 扩展
     - 中

核心结论：

1. **DDP/FSDP 的场景最适合 torch.compile** 。因为这些场景中编译结果可以跨 rank 共享（通过磁盘缓存），总编译时间 = 单 GPU 编译时间 + 缓存加载时间。

2.**流水线并行中每个 stage 独立编译 ** ，但 stages 之间的编译不共享。如果流水线有 8 个 stages，总编译时间是串行的。可以通过预编译所有 stage 的缓存来缓解。

3.**混合并行（TP + PP + DP）场景** 中编译时间最长，建议只编译关键路径（如 Transformer block），而将 embedding、loss 等部分保留为 eager 模式。

编译时间的优化策略
------------------------

.. code-block:: python

   import os
   import torch

   # 策略 1: 预编译缓存
   # 在启动训练之前，先在各 rank 上运行一次编译

   # 策略 2: 使用环境变量控制编译行为
   os.environ["TORCH_COMPILE_DEBUG"] = "0"
   os.environ["TORCH_COMPILE_CACHE_PATH"] = "/shared/cache"

   # 策略 3: 限制 autotune 范围
   torch._inductor.config.max_autotune = False  # 使用默认的 autotune 级别
   torch._inductor.config.autotune_in_subproc = True  # 子进程 autotune 避免 OOM

   # 策略 4: 使用 progressive 编译
   # 先编译模型的一部分，逐步扩展
   for layer_idx in range(len(model.layers)):
       model.layers[layer_idx] = torch.compile(model.layers[layer_idx])
       # 训练几个 step 让编译完成
       train_step(model, batch)

HuggingFace Trainer + torch.compile
=============================================

HuggingFace 的 ``Trainer`` 从 Transformers 4.26+ 开始内置了对 torch.compile 的支持。

基本用法
----------------

.. code-block:: python

   from transformers import (
       AutoModelForSequenceClassification,
       Trainer, TrainingArguments,
   )

   model = AutoModelForSequenceClassification.from_pretrained(
       "bert-base-uncased", num_labels=2
   )

   training_args = TrainingArguments(
       output_dir="./results",
       per_device_train_batch_size=32,
       fp16=True,
       torch_compile=True,           # 启用 torch.compile
       torch_compile_mode="default", # 编译模式
       dataloader_num_workers=4,
       save_strategy="epoch",
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
   )

   trainer.train()

只需设置 ``torch_compile=True`` ，Trainer 会在内部调用 ``torch.compile`` 。

自定义编译配置
----------------

.. code-block:: python

   from transformers import Trainer, TrainingArguments

   training_args = TrainingArguments(
       output_dir="./results",
       torch_compile=True,
       torch_compile_mode="max-autotune",

       # 分布式相关
       ddp_find_unused_parameters=False,  # compile 模式下建议设为 False
       gradient_checkpointing=False,       # 与 compile 配合可能有冲突
   )

DDP + Trainer + Compile 的最佳实践
------------------------------------------

.. code-block:: python

   # 启动脚本
   #   torchrun --nproc_per_node=8 train.py

   import torch
   import torch.distributed as dist
   from transformers import Trainer, TrainingArguments

   # 在 Trainer 初始化前设置分布式
   local_rank = dist.get_rank()
   torch.cuda.set_device(local_rank)

   training_args = TrainingArguments(
       output_dir="./results",
       per_device_train_batch_size=32,
       torch_compile=True,
       ddp_backend="nccl",
       ddp_find_unused_parameters=False,
       remove_unused_columns=False,  # compile 下建议保留所有列
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=dataset,
       data_collator=collator,
   )

   trainer.train()

.. tip::

   使用 HuggingFace Trainer 时，如果遇到 ``torch_compile=True`` 导致的错误，先尝试 ``torch_compile_mode="default"`` （而非 ``reduce-overhead`` ），因为训练场景下 ``default`` 模式的兼容性最好。

   另一个常见问题是 ``gradient_checkpointing`` 与 ``torch.compile`` 的冲突——两者都试图修改 forward 的计算图。如果同时启用遇到问题，建议只使用其中一种优化。
