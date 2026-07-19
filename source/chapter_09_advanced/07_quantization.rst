.. _quantization:

=========================
量化与 torch.compile
=========================

前几章的优化集中在编译流水线本身——图捕获、算子融合、代码生成。但有一项与编译正交的优化手段同样重要：**模型量化（Quantization）** 。量化通过降低权重和激活值的数值精度（从 FP32 到 INT8/FP16）来缩小模型体积、加速推理，且与 torch.compile 的融合优化可以叠加。

这一节梳理 PyTorch 2.x 中的量化方案及其与 torch.compile 的集成方式。

为什么需要量化
===================

量化的核心收益来自两个维度：

**减少内存带宽压力** 。INT8 权重的体积是 FP32 的 1/4。对于带宽瓶颈型操作（如 LLM 推理中的逐 token 解码），权重加载时间直接减半。

**利用低精度计算单元** 。现代 GPU 和 CPU 都对低精度数据类型有专门的加速指令（如 NVIDIA 的 Tensor Core INT8、CPU 的 VNNI 指令）。量化后的运算可以映射到这些高效指令上。

.. note::

   量化不是免费的。精度损失的大小取决于量化方案、模型类型和校准数据的代表性。
   通常需要在部署前用验证集评估精度退化，确保在可接受的范围内。

动态量化（Dynamic Quantization）
======================================

动态量化是最简单、最通用的量化方案，**无需校准数据**，开箱即用：

- 权重：离线量化为 INT8（或 FP16）
- 激活值：每次推理时动态量化（仍以 FP32 参与计算）
- 适用层：``nn.Linear``、``nn.LSTM``、``nn.GRU``

.. code-block:: text
   :caption: 动态量化的核心概念

   # 权重离线量化
   weight_fp32 = [0.12, -0.55, 0.78, ...]  →  scale, zero_point, weight_int8

   # 激活值运行时动态量化
   # 每次 matmul 前: input_fp32 → quantize → matmul_int8 → dequantize → output_fp32

对于线性层占主导的模型（MLP、BERT、LLM），动态量化通常能实现 **2-4 倍** 的权重压缩，
推理速度提升 **1.5-2 倍** （带宽瓶颈场景）。

.. note::

   动态量化的激活值动态量化本身也有运行时开销。在计算密集型场景（大 batch、大张量），
   这部分开销可能抵消权重量化带来的带宽收益。选择量化方案时需要结合具体部署场景评估。

PT2E 静态量化（PT2E Static Quantization）
==============================================

PT2E（PyTorch 2 Export）静态量化是 PyTorch 2.x 推荐的量化路径，与 torch.compile 的
集成最为紧密。它的核心思想是：**先导出，再量化，最后编译**。

完整流程：

.. mermaid::

   flowchart LR
       A[FP32 Model] --> B[torch.export.export]
       B --> C[ExportedProgram]
       C --> D[prepare_pt2e<br/>插入观察点]
       D --> E[Calibrate<br/>收集激活分布]
       E --> F[convert_pt2e<br/>替换为量化算子]
       F --> G[torch.compile]
       G --> H[量化 + 编译<br/>推理]

关键区别：与动态量化不同，静态量化在 **校准阶段** 观察激活值的分布，将缩放因子（scale）
和零点（zero_point）固定下来。推理时激活值直接以 INT8 参与运算，不需要运行时动态量化，
因此计算效率更高。

.. note::

   PT2E 量化在 PyTorch 2.5+ 中已从 ``torch.ao.quantization`` 迁移到独立的
   ``torchao`` 包（\ `https://github.com/pytorch/ao <https://github.com/pytorch/ao>`_\ ），
   安装方式为 ``pip install torchao``。

量化 + torch.compile 的叠加效应
====================================

量化与 torch.compile 的优化是 **正交的**：

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 优化手段
     - 解决的问题
     - 适用场景
   * - ``torch.compile``
     - Kernel launch 开销、算子融合、自动调优
     - 计算密集型、频繁调用的模型
   * - 动态量化
     - 权重加载带宽、模型体积
     - 带宽瓶颈型推理（LLM 解码）
   * - 静态量化
     - 权重 + 激活值带宽，利用低精度计算单元
     - 推理部署（特别是 CPU）
   * - 量化 + ``compile``
     - 两者叠加
     - 带宽 + 计算双重瓶颈的场景

.. tip::

   量化 + compile 的组合并非在所有场景都加速。对于极小的模型或计算量极低的层，
   量化和编译引入的额外开销可能超过收益。建议在实际部署时用基准测试验证。

常见陷阱与最佳实践
=====================

**量化与精度**
  - 8-bit 量化对大多数推理任务精度损失 < 1%，但某些模型（如极低比特量化、敏感层）需要
    混合精度策略：关键层保持 FP16，非关键层 INT8。

**量化 + compile 的调试**
  - 量化后的模型包含 ``Quantized`` 算子，可能在 Dynamo 的图捕获中引入 graph break。
    使用 ``TORCH_LOGS="+dynamo"`` 观察是否有预期的融合被量化算子打断。
  - 如果量化模型的编译速度明显慢于 FP32 版本，检查是否有量化算子缺少 lowering 规则。

**量化方案选择**
  - **快速验证**：动态量化（一行代码，零校准）
  - **CPU 部署**：PT2E 静态量化（高吞吐，但需要校准）
  - **GPU 部署**：FP16/INT8 静态量化 + compile（利用 Tensor Core）
  - **LLM 推理**：Weight-only INT4/INT8 量化 + compile（``torchao`` 包支持）

.. seealso::

   - PyTorch 2 Export 量化指南：\ `PyTorch Quantization <https://pytorch.org/docs/stable/quantization.html>`_
   - torchao 包：\ `https://github.com/pytorch/ao <https://github.com/pytorch/ao>`_
   - 量化与编译的组合 benchmark：第 10 章 LLM 推理案例中包含了量化前后的性能对比

示例
========

以下示例演示动态量化与 torch.compile 的配合使用，以及性能对比：

.. synced-code-start:: dynamic_quant

   .. code-block:: python
      :linenos:

      import torch
      import torch.nn as nn
      from torch.ao.quantization import quantize_dynamic


      class MLP(nn.Module):
          """多层感知机，线性层占主导，适合演示量化"""

          def __init__(self, in_dim=512, hidden_dim=1024, out_dim=256):
              super().__init__()
              self.fc1 = nn.Linear(in_dim, hidden_dim)
              self.fc2 = nn.Linear(hidden_dim, hidden_dim)
              self.fc3 = nn.Linear(hidden_dim, out_dim)
              self.relu = nn.ReLU()

          def forward(self, x):
              x = self.relu(self.fc1(x))
              x = self.relu(self.fc2(x))
              return self.fc3(x)


      def create_quantized_model():
          """构建 FP32 模型并执行动态量化。

          动态量化将 Linear 层的权重离线量化为 int8，
          激活值在推理时动态量化（保持 fp32）。
          这是对 Linear 密集型模型最简单的加速手段。
          """
          model = MLP()
          model.eval()

          quantized = quantize_dynamic(
              model,
              {nn.Linear},
              dtype=torch.qint8,
          )
          return quantized


      quantized_model = create_quantized_model()

      # 验证输出
      x = torch.randn(1, 512)
      with torch.no_grad():
          out = quantized_model(x)
          print(f"量化模型输出 shape: {out.shape}, dtype: {out.dtype}")

      # 量化模型可直接传给 torch.compile
      compiled_quantized = torch.compile(quantized_model)
      with torch.no_grad():
          out2 = compiled_quantized(x)
          print(f"量化 + 编译输出 shape: {out2.shape}")

.. synced-code-end::

.. synced-code-start:: perf_compare

   .. code-block:: python
      :linenos:

      def benchmark(model, example_inputs, n_warmup=20, n_iter=500, desc=""):
          with torch.no_grad():
              for _ in range(n_warmup):
                  model(*example_inputs)
              t0 = time.perf_counter()
              for _ in range(n_iter):
                  model(*example_inputs)
              elapsed = time.perf_counter() - t0
          avg_ms = elapsed / n_iter * 1000
          print(f"  {desc:35s} {avg_ms:8.2f} ms/iter")
          return avg_ms


      # 重建 FP32 模型做对比
      fp32_model = MLP()
      fp32_model.eval()
      compiled_fp32 = torch.compile(fp32_model)

      print()
      print("=" * 55)
      print("  性能对比 (CPU)")
      print("=" * 55)

      x = torch.randn(1, 512)

      benchmark(fp32_model, (x,), desc="FP32 Eager")
      benchmark(compiled_fp32, (x,), desc="FP32 + compile")
      benchmark(quantized_model, (x,), desc="INT8 (Dynamic) Eager")
      benchmark(compiled_quantized, (x,), desc="INT8 (Dynamic) + compile")

      print()
      # 模型大小
      fp32_size = sum(p.numel() for p in fp32_model.parameters()) * 4 / 1024
      print(f"FP32 权重大小: {fp32_size:.0f} KB")
      print(f"INT8 权重大小（理论）: {fp32_size / 4:.0f} KB")
      print(f"预期压缩比: 4x")
      print()

.. synced-code-end::

小结
======

1. **量化是 compile 的正交优化**，两者叠加可同时压缩模型和加速推理
2. **动态量化** 最简单，零校准数据，推荐作为量化方案的起点
3. **PT2E 静态量化** 精度更高，适合对延迟敏感的 CPU 部署场景
4. **量化方案选择** 取决于部署目标（CPU/GPU）、模型类型（CNN/LLM）和精度要求
5. **量化 + compile 不是万能药**，小模型或极低计算量场景可能收益有限，需要实际 benchmark 验证
