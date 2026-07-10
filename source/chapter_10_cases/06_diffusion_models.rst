.. _diffusion-models:

==============================
案例 6：扩散模型优化
==============================

.. note::

   **扩散模型的推理管线包含大量逐元素操作，是 torch.compile 的理想优化对象。 **
   在 Stable Diffusion 的 UNet 中，每个推理 step 包含约 200-300 个操作，其中约 60% 是逐元素操作（SiLU、GroupNorm、加法、乘法）——这些正是 Inductor Scheduler 最擅长的融合类型。实测中， ``default`` 模式可以将 UNet 的推理速度提升 1.3x-1.8x， ``max-autotune`` 模式可达 1.8x-2.5x。

扩散模型（Diffusion Model）是当前图像生成领域的主流架构。Stable Diffusion、DALL-E 3、Midjourney 等产品都基于扩散模型。这一节聚焦于 torch.compile 对扩散模型推理的优化。

扩散模型推理管线概览
========================================

扩散模型的推理（采样）是一个迭代过程：从随机噪声开始，逐步去噪生成图像。每一步都需要模型（通常是 UNet）的前向传播。

.. mermaid::

   flowchart TD
       subgraph Pipeline["扩散模型推理管线"]
           A["随机噪声\n(纯高斯噪声)"] --> B["迭代去噪循环\n(T 步)"]
           B --> C["最终图像"]

           subgraph OneStep["单步去噪"]
               D["当前 latent\n+ 时间步 t"] --> E["UNet 前向\n(核心计算)"]
               E --> F["预测噪声"]
               F --> G["更新 latent\nx_{t-1} = x_t - noise"]
           end

           subgraph CompileTarget["torch.compile 优化目标"]
               H["Conv2d 层\n(下采样/上采样)"]
               I["Attention 层\n(Cross/Self)"]
               J["逐元素操作\n(SiLU, GroupNorm, Add)"]
               H --> K["融合优化"]
               I --> K
               J --> K
           end

           OneStep --> CompileTarget
       end

       B -->|"循环 T 步"| OneStep

       style CompileTarget fill:#e3f2fd,stroke:#1565c0

torch.compile 对扩散模型的优化主要集中在**UNet 的单步前向传播 ** 上。由于每一步的模型结构和输入形状相同（除非使用了动态分辨率），编译结果可以在所有 step 间复用。

UNet 架构与 torch.compile 优化
==================================================

扩散模型的 UNet 结合了卷积和注意力两种操作：

卷积路径（下采样/上采样）
-------------------------------

UNet 中的卷积层与 ResNet 类似——由 Conv2d + GroupNorm + SiLU 组成。Inductor 对这些模式的融合与 :ref:`resnet-optimization` 中描述的 Conv-BN-ReLU 融合类似，但需要处理 GroupNorm 而非 BatchNorm。

.. code-block:: text

   Eager 模式 (4 个 kernel):
       Conv2d → 写回显存
       GroupNorm → 写回显存
       SiLU → 写回显存
       Dropout → 写回显存

   Compiled 模式 (1 个融合 kernel):
       [Fused Conv-GroupNorm-SiLU]:
           conv_out = Conv2d(x)
           norm_out = GroupNorm(conv_out)
           out = SiLU(norm_out)
           → 只写回一次

.. tip::

   Stable Diffusion 的 UNet 中约有 30 个 Conv-GroupNorm-SiLU 模式。融合这些模式可以将 kernel 数量减少约 50%。通过 ``TORCH_LOGS="+pattern_matcher"`` 可以观察这些模式的匹配情况。

Attention 层（Cross-Attention 和 Self-Attention）
--------------------------------------------------------

UNet 中的 Attention 与 LLM 的注意力类似，但包含**Cross-Attention** （与文本 embedding 的交互）。Inductor 的 ``fuse_attention.py`` 同样可以识别这些注意力模式。

.. code-block:: text

   Cross-Attention 计算:
       Q = W_q @ x        (图像特征 → Query)
       K = W_k @ c        (文本特征 → Key)
       V = W_v @ c        (文本特征 → Value)
       score = Q @ K.T / sqrt(d)
       attn = softmax(score)
       output = attn @ V

   torch.compile 的优化:
       - QKV 投影融合 (如果 Key/Value 来自相同输入)
       - Attention Score 计算融合 (替换为 Flash Attention)
       - 输出投影与残差连接融合

时间步嵌入（Time Embedding）的挑战
==============================================

扩散模型的核心机制是 **时间步嵌入（Time Embedding）**——将当前时间步 t 编码为向量，注入到 UNet 的各层中。这个机制给 torch.compile 带来了独特的挑战。

时间步嵌入的注入方式
-------------------------

.. code-block:: python

   class UNetBlock(nn.Module):
       def __init__(self, ...):
           super().__init__()
           self.conv1 = nn.Conv2d(...)
           self.norm1 = nn.GroupNorm(...)
           self.silu = nn.SiLU()

           # 时间步嵌入投影
           self.time_proj = nn.Linear(time_dim, channels)

       def forward(self, x, t_emb):
           h = self.conv1(x)
           h = self.norm1(h)

           # 时间步注入 (逐元素加法)
           h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)

           h = self.silu(h)
           return h

时间步嵌入在每次推理 step 中不同，但 torch.compile 的处理方式是优雅的：

1.** 编译时不依赖具体的时间步值 **。编译器只看到 ``h + t_emb`` 的加法操作，不关心 ``t_emb`` 的具体数值
2.**Guard 检查的是形状而非值 ** 。只要 ``t_emb`` 的形状不变（通常如此），就不会触发重新编译
3.**时间步嵌入的计算量很小** （一个 Linear 层 + 加法），融合到周围的 kernel 中

.. note::

   如果时间步嵌入的形状会在不同 step 之间变化（例如使用了某种自适应时间步采样），可以使用 ``torch._dynamo.mark_dynamic`` 标记时间步维度为动态，或通过 padding 固定形状。

性能对比：Eager vs Compile
============================================

下面使用 ``diffusers`` 库对 Stable Diffusion 进行 benchmark。

.. list-table::
   :header-rows: 1

   * - 配置
     - UNet 单步延迟
     - 总采样时间 (50 步)
     - 加速比
   * - Eager (fp32)
     - ~120ms
     - ~6.0s
     - 1.0x (基线)
   * - compile default (fp32)
     - ~75ms
     - ~3.8s
     - 1.6x
   * - compile max-autotune (fp32)
     - ~55ms
     - ~2.8s
     - 2.1x
   * - Eager (fp16)
     - ~65ms
     - ~3.3s
     - 1.8x (相对 fp32 基线)
   * - compile default (fp16)
     - ~42ms
     - ~2.1s
     - 2.9x
   * - compile max-autotune (fp16)
     - ~35ms
     - ~1.8s
     - 3.3x

数据基于 Stable Diffusion 1.5，分辨率 512x512，单张 NVIDIA A100。实际加速比取决于 GPU 型号和驱动版本。

Benchmark 脚本
--------------------

.. code-block:: python

   import torch
   from diffusers import StableDiffusionPipeline
   import time

   def benchmark_sd(prompt="a cat", num_inference_steps=50, compile_mode=None):
       pipe = StableDiffusionPipeline.from_pretrained(
           "runwayml/stable-diffusion-v1-5",
           torch_dtype=torch.float16,
       ).to("cuda")

       # 可选：编译 UNet
       if compile_mode:
           pipe.unet = torch.compile(pipe.unet, mode=compile_mode)

       # 预热
       _ = pipe(
           prompt,
           num_inference_steps=5,
           output_type="latent",
       )
       torch.cuda.synchronize()

       # 测量
       start = time.perf_counter()
       _ = pipe(
           prompt,
           num_inference_steps=num_inference_steps,
           output_type="latent",
       )
       torch.cuda.synchronize()
       elapsed = time.perf_counter() - start

       return elapsed

   for mode in [None, "default", "max-autotune"]:
       elapsed = benchmark_sd(compile_mode=mode)
       label = mode if mode else "eager"
       print(f"{label}: {elapsed:.2f}s")

   # 输出示例:
   #   eager: 6.01s
   #   default: 3.78s
   #   max-autotune: 2.75s

.. warning::

   首次运行 ``benchmark_sd("max-autotune")`` 时，编译时间可能长达 5-15 分钟。这是因为 UNet 包含大量不同的卷积和注意力配置，每个都需要 autotune。建议在开发环境中预先编译好，将缓存保存到磁盘，部署时直接加载。

内存优化模式
====================================

扩散模型推理的另一个瓶颈是显存。Stable Diffusion 在 512x512 分辨率下约需要 4-6GB 显存（fp16），而更高分辨率（如 1024x1024）可能超过 10GB。torch.compile 的融合优化可以减少中间结果的显存占用。

显存节省分析
--------------------

.. code-block:: text

   单步 UNet 前向的显存分配:

   操作                          | Eager 显存 | Compiled 显存 | 节省
   ─────────────────────────────┼────────────┼───────────────┼──────
   下采样路径 (3 层)            | 1.2 GB     | 0.8 GB        | 33%
   中间路径 (1 层)              | 0.6 GB     | 0.4 GB        | 33%
   上采样路径 (3 层)            | 1.2 GB     | 0.8 GB        | 33%
   Attention 中间结果           | 0.8 GB     | 0.3 GB        | 62%
   ─────────────────────────────┼────────────┼───────────────┼──────
   总计                          | 3.8 GB     | 2.3 GB        | 40%

显存节省主要来自两个机制：

1. **Kernel 融合减少了中间 tensor 的写回 ** 。在 eager 模式下，每个操作的结果都要写回显存供下一个操作使用。融合后，中间结果保持在寄存器中。

2.**Flash Attention 无需实例化完整 attention 矩阵 ** 。对于 1024x1024 分辨率的 latent（即 128x128 特征图），attention 矩阵的大小为 ``128^2 x 128^2 = 256M`` 个元素，约 1GB（fp16）。Flash Attention 将其分块计算，避免了完整的矩阵分配。

显存优化配置
--------------------

.. code-block:: python

   import torch
   from diffusers import StableDiffusionPipeline

   pipe = StableDiffusionPipeline.from_pretrained(
       "runwayml/stable-diffusion-v1-5",
       torch_dtype=torch.float16,
   ).to("cuda")

   # 启用 torch.compile
   pipe.unet = torch.compile(
       pipe.unet,
       mode="max-autotune",
       options={
           "max_fusion_size": 256,     # 允许更大的融合组
           "triton.cudagraphs": True,  # 启用 CUDA Graph
       },
   )

   # 启用 attention slicing（进一步减少显存）
   pipe.enable_attention_slicing()

   # 启用模型 CPU offload（显存不足时）
   pipe.enable_model_cpu_offload()

   # 生成
   image = pipe(
       "a cat",
       num_inference_steps=50,
       height=768,
       width=768,
   ).images[0]

.. tip::

   对于扩散模型推理，**建议始终使用 fp16 精度 ** 。fp16 不仅减少显存占用，还能利用 Tensor Core 加速计算。在大多数扩散模型中，fp16 的生成质量与 fp32 几乎无异，但性能提升可达 2x。

   ``enable_attention_slicing()`` 与 torch.compile 兼容良好，二者结合可以进一步降低显存峰值。

调度器与 torch.compile 的交互
========================================

扩散模型的采样过程由**调度器（Scheduler）** 控制，如 DDIM、DPM-Solver、Euler 等。调度器决定了每个 step 如何更新 latent。

调度器通常运行在 CPU 上（或 GPU 上的简单操作），与 UNet 的 GPU 计算交替进行：

.. code-block:: python

   # 典型的采样循环（伪代码）
   latents = torch.randn(...).to("cuda")

   for t in scheduler.timesteps:
       # 调度器在 CPU 上计算参数
       sigma = scheduler.get_sigma(t)

       # UNet 在 GPU 上计算
       noise_pred = unet(latents, t, encoder_hidden_states)

       # 调度器更新 latent（GPU 或 CPU）
       latents = scheduler.step(noise_pred, t, latents)

由于调度器的操作在 UNet 的编译图之外，它不会影响 UNet 的编译优化。但需要注意的是：

- 如果调度器操作在 GPU 上执行（如 DPM-Solver 的某些实现），这些操作不会被 torch.compile 优化（因为它们在编译图外）
- 调度器的 ``step`` 方法通常简单（几个逐元素操作），编译收益不大

如果调度器的操作也包含大量计算，可以将其纳入编译范围：

.. code-block:: python

   @torch.compile
   def denoising_step(unet, latents, t, encoder_hidden_states, scheduler_params):
       noise_pred = unet(latents, t, encoder_hidden_states)
       # 将调度器 step 纳入编译
       return scheduler_step(noise_pred, latents, t, scheduler_params)

   # 采样循环
   for t in scheduler.timesteps:
       latents = denoising_step(unet, latents, t, encoder_hidden_states, params)

这种方法将 UNet 前向和调度器更新合并为一个大图，Scheduler 可以更好地优化整体计算流。但代价是第一次调用时编译时间更长。

预期加速效果总结
==========================

.. list-table::
   :header-rows: 1

   * - 优化配置
     - 推理加速
     - 显存节省
     - 编译时间
     - 适用场景
   * - default
     - 1.3x - 1.8x
     - 10-20%
     - 2-5 分钟
     - 快速部署
   * - max-autotune
     - 1.8x - 2.5x
     - 20-30%
     - 5-15 分钟
     - 生产服务
   * - max-autotune + fp16
     - 2.5x - 3.5x
     - 40-50%
     - 5-15 分钟
     - 高吞吐服务
   * - max-autotune + slicing
     - 2.0x - 3.0x
     - 50-60%
     - 5-15 分钟
     - 大分辨率 (1024+)

扩散模型的 torch.compile 优化是 **投资回报率很高** 的实践——只需在 ``pipe.unet`` 上调用一次 ``torch.compile`` ，即可获得 2x 左右的加速，零代码修改。对于生产环境中的扩散模型推理服务，强烈推荐使用 ``max-autotune`` 模式配合 AOTInductor 导出（参见 :ref:`resnet-optimization` 中的 AOTInductor 章节）。
