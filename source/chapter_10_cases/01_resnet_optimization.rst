.. _resnet-optimization:

==============================
案例 1：ResNet 优化
==============================

.. note::

   **ResNet 是 torch.compile 最佳 benchmark 之一。 **
   在 PyTorch 团队的内测中，ResNet50 在 ``max-autotune`` 模式下可以达到 2.8x 的推理加速比——这意味着一个本来需要 10ms 的前向传播，编译后只需要 3.5ms。这主要得益于三个因素：ResNet 全是 Conv + BN + ReLU 的组合（**Scheduler 最擅长的融合模式 ** ）、没有控制流（**无 graph break** ）、固定的输入尺寸（ **形状稳定** ）。如果你的模型也具备这三个特点，大概率也能获得 2x+ 的加速。

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

如果看到大量的小 kernel（每个运行时间 < 10us），说明融合不充分，可以尝试增大 ``max_fusion_size`` 。

编译时间线
=================

torch.compile 编译 ResNet 的过程可以分为三个阶段。下面的时序图展示了从首次调用到最终执行的全流程：

.. mermaid::

   sequenceDiagram
       participant 用户代码
       participant Dynamo
       participant Inductor
       participant Triton/cuDNN
       participant GPU

       用户代码->>Dynamo: 首次调用 compiled_model(x)
       Note over Dynamo: 图捕获 (Fx Graph)
       Dynamo->>Dynamo: 追踪 forward 中的每个操作
       Dynamo->>Dynamo: Guard 生成 (形状、设备、dtype)
       Dynamo->>Inductor: 传递 GraphModule

       Note over Inductor: 编译优化
       Inductor->>Inductor: Lowering: FX Graph -> IRNode
       Inductor->>Inductor: Pattern Matching: 识别 Conv+BN+ReLU
       Inductor->>Inductor: Fusion: 合并相邻 Pointwise 操作
       Inductor->>Inductor: Scheduler: 生成 Kernel 调用顺序

       Inductor->>Triton/cuDNN: 生成 Triton Kernel / 调用 cuDNN
       Triton/cuDNN->>GPU: 编译并加载 Kernel
       GPU-->>用户代码: 返回编译结果

       Note over 用户代码,GPU: 后续调用: 直接执行已编译的 Kernel
       用户代码->>Dynamo: compiled_model(x) (第 2 次)
       Dynamo->>Dynamo: Guard 验证 (形状不变 → 命中缓存)
       Dynamo->>Inductor: 直接使用缓存的 Kernel
       Inductor->>GPU: 执行编译后的计算图
       GPU-->>用户代码: 返回结果

编译时间中，Inductor 的 Lowering 和 Scheduler 占比最大。对于 ResNet50 的第一次调用，编译时间通常在 30-120 秒（取决于 GPU 和 ``mode`` 参数）。缓存生效后，后续调用仅为 1-3ms。

完整的 Benchmark 脚本
============================

下面是一个完整的、可直接运行的 benchmark 脚本，使用上一节介绍的 ``BenchmarkRunner`` （见 ``examples/benchmark_utils.py`` ）：

.. code-block:: python

   import torch
   import torchvision.models as models
   from benchmark_utils import (
       BenchmarkRunner, warmup_cuda,
       format_speedup_table, torch_compile_info,
   )

   def main():
       # 预热 CUDA
       warmup_cuda()

       # 加载模型和数据
       model = models.resnet50(weights=None).cuda().eval()
       x = torch.randn(32, 3, 224, 224, device="cuda")
       print(torch_compile_info())

       # 对比不同模式
       runner = BenchmarkRunner(
           model, x,
           n_warmup=10,
           n_iter=50,
           sync=True,
       )
       modes = ["eager", "default", "reduce-overhead", "max-autotune"]
       results = runner.compare_modes(modes)
       runner.print_report(results)

   if __name__ == "__main__":
       main()

运行此脚本时，建议首先使用 ``default`` 模式确认无 graph break，再切换到 ``max-autotune`` 追求极致性能。

.. tip::

   在运行 benchmark 之前，通过 ``torch_compile_info()`` 输出当前配置是个好习惯。不同版本的 PyTorch 和 GPU 驱动可能导致加速比有 10-20% 的差异。

Kernel 融合分析：Conv + BN + ReLU
============================================

ResNet 的核心构建块是 ``Conv2d + BatchNorm2d + ReLU`` 三元组。在 eager 模式下，这三个操作分别启动独立的 kernel：

.. code-block:: text

   Eager 模式 (3 个 kernel):
       Conv2d:    [cuDNN Convolution]  → 写回显存
       BatchNorm: [cuDNN BN]           → 写回显存
       ReLU:      [Pointwise]          → 写回显存

   Compiled 模式 (1 个融合 kernel):
       [Fused Conv-BN-ReLU]:
           conv_output = Conv2d(x)
           norm_output = (conv_output - mean) / sqrt(var + eps) * gamma + beta
           output = max(0, norm_output)
           → 只写回一次

融合的核心在于 **减少显存带宽消耗** 。每个 kernel 从显存读取输入、写回输出，带宽是 GPU 上最稀缺的资源之一。三个分离的 kernel 需要 6 次显存访问（3 次读 + 3 次写），而融合后只需要 2 次（1 次读输入 + 1 次写输出）。

对于 ResNet50，整个模型包含约 50 个 Conv-BN-ReLU 三元组。融合前后的 kernel 数量对比如下：

.. list-table::
   :header-rows: 1

   * - 操作类型
     - Eager Kernel 数
     - Compiled Kernel 数
     - 减少比例
   * - 卷积 (Conv2d)
     - 53
     - 53 (cuDNN 调用)
     - 0%
   * - 批归一化 (BatchNorm)
     - 53
     - 0 (已融合)
     - 100%
   * - 激活函数 (ReLU)
     - 49
     - 0 (已融合)
     - 100%
   * - 逐元素操作 (Add, Mul)
     - 30+
     - 10-15 (部分融合)
     - ~60%
   * - 总 kernel 数
     - ~185
     - ~65-70
     - ~65%

.. note::

   卷积操作本身不会与其他操作融合——cuDNN 的卷积实现已经是高度优化的库调用，Inductor 通过 ``extern_kernels`` 调用 cuDNN，而不是将其重写为 Triton kernel。融合发生在卷积输出后的逐元素操作链上。

Inductor 的 Pattern Matcher 如何识别 Conv + BN
========================================================

Inductor 中负责模式匹配的模块位于 ``torch._inductor.fx_passes`` 。它使用 ``@register_graph_pattern`` 装饰器注册了一组针对 CNN 的融合模式。

对于 Conv-BN 融合，核心逻辑如下：

1. **图捕获阶段 ** ：Dynamo 将 ``forward`` 中的 ``Conv2d`` 、 ``BatchNorm2d`` 、 ``ReLU`` 等操作捕获为 FX Graph 中的独立节点。

2.**Lowering 阶段 ** ：Inductor 将每个 FX Node 转化为 ``IRNode`` 。 ``Conv2d`` 变为 ``ExternKernelOut`` （调用 cuDNN），而 ``BatchNorm2d`` 和 ``ReLU`` 变为 ``Pointwise`` 节点。

3.**Pattern Matching** ：Inductor 遍历计算图，寻找以下模式的匹配：

   .. code-block:: text

       [Conv2d] → [BatchNorm2d] → [ReLU]
          ↓            ↓              ↓
       ExternKernel  Pointwise     Pointwise

   匹配条件包括：
   - BatchNorm2d 的输入直接来自 Conv2d 的输出（无中间操作）
   - BatchNorm2d 处于 eval 模式（training 模式下的 BN 行为不同）
   - 所有操作的 dtype 和设备一致

4.**融合决策** ：匹配成功后，Scheduler 将这三个节点标记为一个 ``FusedSchedulerNode`` 。生成的代码如下：

   .. code-block:: python

       # 伪代码：Inductor 生成的融合 kernel
       def fused_conv_bn_relu_kernel(x, w, bn_weight, bn_bias,
                                     running_mean, running_var):
           # Step 1: 计算卷积 (调用 cuDNN)
           conv_out = cudnn_convolution(x, w)

           # Step 2: 计算 BN (在寄存器中完成，无需写回)
           norm_out = (conv_out - running_mean) / sqrt(running_var + eps)
           norm_out = norm_out * bn_weight + bn_bias

           # Step 3: 计算 ReLU (在寄存器中完成，无需写回)
           out = max(0, norm_out)
           return out

这个融合模式在 Inductor 的 ``fx_passes/fuse_conv_bn.py`` 中实现。可以通过 ``TORCH_LOGS="+pattern_matcher"`` 观察匹配过程：

.. code-block:: bash

   TORCH_LOGS="+pattern_matcher" python -c "
   import torch
   import torchvision.models as models
   model = models.resnet50(weights=None).cuda().eval()
   compiled = torch.compile(model)
   x = torch.randn(4, 3, 224, 224, device='cuda')
   compiled(x)
   "

在日志中，你会看到类似 ``matched pattern: conv_bn_relu`` 的输出，以及每个匹配的节点名称。

.. tip::

   如果你想禁用 Conv-BN 融合来观察性能差异（例如用于消融实验），可以设置：
   ``torch._inductor.config.fuse_conv_bn = False`` 。
   这有助于量化融合带来的具体收益。

部署优化：使用 AOTInductor 导出 ResNet
==============================================

对于生产部署场景，每次启动时重新编译模型是不可接受的。PyTorch 2.x 提供了 **AOTInductor**——将编译后的模型导出为一个独立的共享库，部署时无需 Python 环境和编译器。

导出步骤
----------------

.. code-block:: python

   import torch
   import torchvision.models as models
   from torch._export import aot_compile

   model = models.resnet50(weights=None).cuda().eval()
   x = torch.randn(32, 3, 224, 224, device="cuda")

   # 导出为 .so 文件
   so_path = aot_compile(
       model,
       (x,),                    # 示例输入
       options={
           "max_autotune": True,
           "aot_inductor.output_path": "./resnet50_aot.so",
       },
   )
   print(f"导出完成: {so_path}")

部署时加载
----------------

.. code-block:: cpp

   // C++ 部署代码
   #include <torch/torch.h>
   #include <torch/cuda.h>

   int main() {
       // 加载 AOTInductor 编译的模型
       torch::Tensor x = torch::randn({32, 3, 224, 224})
                            .to(torch::kCUDA);
       auto model = torch::jit::load("./resnet50_aot.so");
       model.to(torch::kCUDA);

       // 推理
       auto output = model.forward({x});
       return 0;
   }

使用 AOTInductor 的优点：

- ** 无 Python 依赖 **：部署环境只需要 libtorch，不需要 Python 解释器和编译器栈
- ** 启动即用 **：无需首次运行的编译等待时间
- ** 环境一致性** ：编译时的优化决策与运行时的 GPU 架构完全匹配

需要注意的限制：

- AOTInductor 目前要求输入形状在编译时确定（至少是静态的），动态形状支持仍在开发中
- 导出的 .so 文件与 GPU 架构绑定（例如在 A100 上导出的不能在 V100 上使用）
- 编译后的文件大小约为 10-50MB（取决于模型大小和融合程度）

性能对比
----------------

.. list-table::
   :header-rows: 1

   * - 部署方式
     - 首次推理延迟
     - 后续推理延迟
     - 部署体积
   * - Eager (Python)
     - ~8ms
     - ~8ms
     - 模型权重 + Python 环境
   * - torch.compile (Python)
     - ~30-120s (编译)
     - ~3ms
     - 模型权重 + 缓存目录
   * - AOTInductor (C++)
     - ~1ms (加载)
     - ~3ms
     - 模型权重 + .so 文件 (~20MB)

对于生产环境中的 ResNet 推理服务，推荐使用 ``max-autotune`` 模式配合 AOTInductor 导出，以获得最佳的性能和部署体验。
