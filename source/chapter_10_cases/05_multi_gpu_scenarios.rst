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

**编译和 DDP 的顺序很重要**：先 ``compile`` 再 ``DDP``。这样 DDP 包装的是编译后的模型，编译器的优化在 DDP 之前生效。如果反过来，DDP 的 hook 会干扰编译器的图捕获。

**每个 GPU 独立编译**。在 DDP 中，每个 rank 独立运行 ``compile``，因为每个 rank 看到的 ``example_inputs`` 形状相同，编译结果也相同。由于磁盘缓存的存在，rank 0 编译后，其他 rank 可以直接使用缓存，避免重复编译。

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

PyTorch 原生 TP API 生成的通信操作（all-reduce、reduce-scatter）可以被 Dynamo 追踪，不会导致 graph break。如果使用自定义的 TP 通信（如手写的 ``all_reduce``），可能会导致 graph break。

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

**编译时间与 GPU 数量线性增长**。如果 N 个 GPU 各自独立编译，总编译时间 = 单 GPU 编译时间 x N。通过磁盘缓存可以解决：rank 0 先编译，其他 rank 从缓存加载。

**编译结果不一致**。如果不同 rank 的 GPU 型号不同（如 A100 和 H100 混部），编译结果不能共享，因为 Triton 编译结果包含 GPU 架构特定的优化。

**OOM 在编译时发生**。``max-autotune`` 模式可能需要额外的显存（用于存储多个 kernel 变体）。如果编译时 OOM，尝试：

.. code-block:: python

   # 限制 autotune 的显存使用
   torch._inductor.config.autotune_in_subproc = True
   # 或在子进程中执行 autotune
   torch._inductor.config.coordinate_descent_tuning = False
