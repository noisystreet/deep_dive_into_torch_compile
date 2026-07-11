.. _llm-inference:

==============================
案例 2：LLM 推理优化
==============================

.. tip::

   **"reduce-overhead" 模式对 LLM 推理特别有效。 **
   在 LLM 的自回归生成中，每个 step 只生成了一个 token——这意味着每次计算量很小，kernel launch 开销占比很高。CUDA Graph 通过将多个 kernel launch 合并为一次 GPU 操作，可以显著减少 CPU 端的调度开销。实测中，对于 GPT-2 的单 token 生成， ``reduce-overhead`` 模式相比 ``default`` 模式可以减少 30-50% 的延迟。如果你的 LLM 推理是 token-by-token 的，这是最值得尝试的优化。

大语言模型（LLM）推理是 torch.compile 最有价值的应用场景之一。这一节以 GPT-2 为例，展示 torch.compile 对 Transformer 推理的优化效果。

基线设置
============

.. code-block:: python

   import torch
   from transformers import GPT2LMHeadModel, GPT2Tokenizer
   import time

   model_name = "gpt2"
   model = GPT2LMHeadModel.from_pretrained(model_name).cuda().eval()
   tokenizer = GPT2Tokenizer.from_pretrained(model_name)

   prompt = "The future of AI is"
   inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

   @torch.no_grad
   def generate(model, input_ids, max_new_tokens=50):
       for _ in range(max_new_tokens):
           outputs = model(input_ids)
           next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
           input_ids = torch.cat([input_ids, next_token], dim=-1)
       return input_ids

   # 基线
   start = time.time()
   generate(model, inputs.input_ids)
   torch.cuda.synchronize()
   print(f"Eager 生成时间: {time.time() - start:.2f}s")

应用 torch.compile
======================

.. code-block:: python

   # 编译模型主体（不包括 tokenizer）
   compiled_model = torch.compile(model, mode="reduce-overhead")

   start = time.time()
   generate(compiled_model, inputs.input_ids)
   torch.cuda.synchronize()
   print(f"Compiled 生成时间: {time.time() - start:.2f}s")

关键优化点
==============

**KV Cache** 。LLM 推理的核心优化是 KV Cache——每次生成新 token 时，只计算最新的 query 和 key/value，复用之前 step 的结果。在 eager 模式下，KV Cache 会带来显着的加速。在 torch.compile 下，需要确保 KV Cache 的实现不会被 graph break 打断。

**优化序列长度** 。LLM 推理时，序列长度在每次迭代中递增。如果使用动态形状：

.. code-block:: python

   # 编译时标记序列长度为动态
   compiled_model = torch.compile(
       model,
       mode="reduce-overhead",
       dynamic=True,  # 允许序列长度变化
   )

或者通过 padding 固定序列长度，避免重新编译：

.. code-block:: python

   # 固定最大序列长度
   MAX_LENGTH = 512
   input_ids = torch.nn.functional.pad(
       input_ids, (0, MAX_LENGTH - input_ids.size(1))
   )

**算子融合优化** 。Transformer 中的常见融合模式：

.. code-block:: text

   1. QKV 投影：三个线性层合并为一个
      Eager: Q = W_q @ x, K = W_k @ x, V = W_v @ x
      编译: [Q, K, V] = W_qkv @ x  (融合为一个 kernel)

   2. Attention 后的全连接层融合
      Eager: x = W1 @ x; x = relu(x); x = W2 @ x
      编译: 融合为单个 kernel

   3. LayerNorm + 残差连接
      Eager: x = x + sublayer(x); x = layernorm(x)
      编译: 融合为单个 kernel

验证优化效果
================

使用 ``torch.profiler`` 观察优化前后的 kernel profile：

.. code-block:: python

   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CUDA],
   ) as prof:
       generate(compiled_model, inputs.input_ids)

   print(prof.key_averages().table(sort_by="cuda_time_total"))

优化的关键指标：

- **kernel 数量减少** ：融合前可能有 50+ 个 kernel，融合后可减少到 10-15 个
- **显存访问减少** ：中间结果写回显存的次数下降
- **Tensor Core 利用率** ：矩阵乘法使用 ``tl.dot`` 调用 Tensor Core

attention 模式的编译优化
============================

Inductor 的 ``fx_passes/fuse_attention.py`` 会自动识别 attention 模式：

.. code-block:: text

   输入: Q, K, V
   子图匹配:
       score = Q @ K.T
       scale = score / sqrt(d)
       mask = scale.masked_fill(causal_mask, -inf)
       attn = softmax(mask, dim=-1)
       output = attn @ V
       ↓
   替换为: scaled_dot_product_attention(Q, K, V)

这个优化由 Pattern Matcher（第 5.9 节）的 ``@register_graph_pattern`` 自动完成。

Flash Attention
====================

对于支持 Flash Attention 的 GPU（Sm80+），Inductor 会自动用 Flash Attention 替换标准的 attention 实现：

.. code-block:: python

   # Inductor 默认启用 Flash Attention
   # 可以通过配置禁用
   torch._inductor.config.fuse_attention = False

Flash Attention 的核心优势是 **不需要将完整的 attention 矩阵写入显存** ，这对长序列场景格外重要。

预期的加速效果
==================

.. list-table::
   :header-rows: 1

   * - 模型
     - 推理加速比
     - 显存减少
   * - GPT-2 (124M)
     - 1.3x - 1.8x
     - 10-20%
   * - LLaMA-7B
     - 1.5x - 2.0x
     - 15-25%
   * - LLaMA-13B（使用 Flash Attention）
     - 1.8x - 2.5x
     - 20-30%

LLM 的加速比通常不如 CNN 高，因为 Transformer 的核心计算是矩阵乘法（已经是高度优化的 cuBLAS/Triton GEMM），torch.compile 的主要贡献在于融合 attention 前后的逐元素操作和 LayerNorm。

LLM 推理流程与 torch.compile 优化阶段
================================================

LLM 的推理分为两个阶段： **预填充（Prefill）** 和 **解码（Decode）** 。torch.compile 对这两个阶段的优化策略不同。

.. mermaid::

   flowchart TD
       subgraph Prefill["预填充阶段 (Prefill)"]
           A["输入: 完整 prompt tokens"] --> B["并行计算所有位置的 attention"]
           B --> C["生成 KV Cache 初始值"]
           C --> D["输出: 第一个新 token"]
       end

       subgraph Decode["解码阶段 (Decode) —— torch.compile 重点优化"]
           E["当前 token"] --> F["单步 forward"]
           F --> G{"QKV 投影融合<br/>(1 个 kernel vs 3)"}
           G --> H["Attention 计算<br/>(Flash Attention)"]
           H --> I["FFN + LayerNorm 融合<br/>(1 个 kernel vs 5+)"]
           I --> J["输出 logits"]
           J --> K["采样下一个 token"]
           K --> L["更新 KV Cache"]
           L --> E
       end

       Prefill -->|"生成首个 token\n(计算密集)"| Decode

       style Prefill fill:#e3f2fd,stroke:#1565c0
       style Decode fill:#fff3e0,stroke:#e65100

在 **Prefill 阶段** ，模型并行处理所有输入 token，计算量大、kernel launch 开销占比低，torch.compile 的优化效果主要来自 Flash Attention 融合和 QKV 投影合并。

在 **Decode 阶段** ，每次只生成一个 token，计算量很小。此时 **kernel launch 开销成为瓶颈**——每次 forward 需要启动数十个 kernel，每个 kernel 的执行时间可能只有几微秒，而启动开销就占了大头。 ``reduce-overhead`` 模式通过 CUDA Graph 将整个 decode 步骤的 kernel launch 合并为一次，可以大幅降低延迟。

torch.compile 在每个 decode step 中的具体优化效果如下：

.. list-table::
   :header-rows: 1

   * - 计算阶段
     - Eager Kernel 数
     - Compiled Kernel 数
     - 主要优化
   * - QKV 投影 (3 个 Linear)
     - 3
     - 1
     - 垂直融合为一个 kernel
   * - Attention Score
     - 5-8 (softmax, mask, matmul)
     - 1 (Flash Attention)
     - 替换为 scaled_dot_product_attention
   * - FFN + 激活
     - 3-5 (Linear, ReLU/SiLU, Linear)
     - 1-2
     - 水平融合
   * - LayerNorm + 残差
     - 2-3
     - 1
     - Pointwise 融合
   * - 总计
     - 15-20
     - 4-6
     - ~70% 减少

PagedAttention 与 vLLM 集成模式
=========================================

vLLM 是目前最流行的 LLM 推理框架之一，其核心优化 **PagedAttention** 解决了 KV Cache 显存碎片问题。vLLM 与 torch.compile 的集成是一个值得关注的方向。

vLLM 默认使用自定义的 CUDA kernel 实现 PagedAttention，不经过 torch.compile。这意味着 vLLM 中的 attention 计算是手写优化的（使用 triton 或 CUDA），而其他部分（如 QKV 投影、FFN、LayerNorm）可以使用 torch.compile 优化。

集成模式 1：编译非 attention 部分
------------------------------------------------

.. code-block:: python

   import torch
   from vllm import LLM, SamplingParams

   # vLLM 内部使用自定义 kernel 处理 attention
   # 其他层通过 torch.compile 优化
   llm = LLM(
       model="meta-llama/Llama-2-7b-hf",
       compile=True,           # 启用 torch.compile
       compile_mode="reduce-overhead",  # 推荐模式
   )

vLLM 的 ``compile=True`` 参数会在模型加载后对非 attention 部分应用 torch.compile。由于 vLLM 的模型实现与 HuggingFace Transformers 不同，其编译效果可能与标准 Transformers 库有所差异。

集成模式 2：自定义 vLLM 模型
------------------------------------------------

如果需要在 vLLM 中使用自定义的编译优化，可以通过 vLLM 的模型注册机制：

.. code-block:: python

   from vllm.model_executor.models import supports_compile

   class MyCompiledLLM(nn.Module):
       def __init__(self, config):
           super().__init__()
           # ... 模型初始化

       def forward(self, input_ids, positions, kv_caches):
           # 使用 torch.compile 装饰 forward
           return self.compiled_forward(input_ids, positions, kv_caches)

       @torch.compile(mode="reduce-overhead")
       def compiled_forward(self, input_ids, positions, kv_caches):
           # ... 模型前向逻辑
           pass

.. tip::

   vLLM 与 torch.compile 的集成仍在发展中。对于生产环境，建议先使用 vLLM 原生的 PagedAttention kernel（已经过高度优化），再考虑是否启用 torch.compile 优化非 attention 部分。在大多数场景下，vLLM 的 custom kernel 已经足够快，torch.compile 的额外收益在 5-15% 之间。

Benchmark 方法论：如何正确测量 LLM 推理性能
========================================================

LLM 推理的 benchmark 比 CNN 更容易出错。下面是一个标准化的测量脚本：

.. code-block:: python

   import torch
   import time
   from transformers import AutoModelForCausalLM, AutoTokenizer

   def benchmark_llm(
       model_name="gpt2",
       prompt="The future of AI is",
       max_new_tokens=100,
       n_warmup=5,
       n_iter=20,
       mode="reduce-overhead",
   ):
       # 加载模型
       model = AutoModelForCausalLM.from_pretrained(model_name).cuda().eval()
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

       if mode != "eager":
           model = torch.compile(model, mode=mode)

       @torch.no_grad()
       def generate(input_ids, max_new_tokens):
           for _ in range(max_new_tokens):
               outputs = model(input_ids)
               next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
               input_ids = torch.cat([input_ids, next_token], dim=-1)
           return input_ids

       # 预热（包括编译开销）
       for _ in range(n_warmup):
           _ = generate(inputs.input_ids, max_new_tokens=10)
       torch.cuda.synchronize()

       # 测量端到端生成时间
       timings = []
       for _ in range(n_iter):
           # 重置输入（重要！避免 KV Cache 污染）
           inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
           torch.cuda.synchronize()
           start = time.perf_counter()
           _ = generate(inputs.input_ids, max_new_tokens)
           torch.cuda.synchronize()
           elapsed = (time.perf_counter() - start) * 1000
           timings.append(elapsed)

       # 计算统计值（排除首次编译影响）
       timings.sort()
       median_ms = timings[len(timings) // 2]
       avg_ms = sum(timings) / len(timings)

       return {
           "mode": mode,
           "median_ms": median_ms,
           "avg_ms": avg_ms,
           "min_ms": min(timings),
           "max_ms": max(timings),
           "tokens_per_sec": max_new_tokens / (median_ms / 1000),
       }

测量 LLM 性能时需要注意的几个关键点：

1. **预热必须包含相同长度的生成** 。如果预热时生成 10 tokens，测量时生成 100 tokens，第一次测量仍会包含编译开销（因为计算图不同）。

2.**每次测量前重置输入** 。如果不重置，KV Cache 会持续增长，导致每次测量的序列长度不同，无法公平对比。

3.**使用中位数而非平均值** 。LLM 的生成时间分布有长尾——偶尔的编译（如 guard 失败）会导致个别测量值显著偏高，中位数比平均值更稳健。

4.**报告 tokens/sec 而非单纯延迟** 。不同配置可能生成不同数量的 token（如 ``max_new_tokens`` ），tokens/sec 提供了标准化的对比基准。

5.**区分 prefill 和 decode 延迟** 。prefill 阶段处理整个 prompt，延迟与 prompt 长度相关；decode 阶段逐 token 生成，延迟相对稳定。对两者分别测量能更精确地定位瓶颈。

CUDA Graphs： ``reduce-overhead`` 模式的底层机制
===========================================================

当使用 ``mode="reduce-overhead"`` 时，Inductor 会在编译后自动将计算图捕获为 CUDA Graph。

CUDA Graph 的工作原理
--------------------------

在常规的 GPU 执行中，每个 kernel 启动都需要 CPU 通过驱动程序向 GPU 发送一个 launch 命令。对于 LLM decode 这种包含大量小 kernel 的场景，CPU→GPU 的通信开销（约 5-50us/次）可能超过 kernel 本身的执行时间。

CUDA Graph 通过一次捕获、多次重放的机制消除这些开销：

.. mermaid::

   flowchart LR
       subgraph Eager["Eager 模式"]
           A1["CPU: 启动 kernel A"] --> B1["GPU: 执行 A"]
           B1 --> C1["CPU: 启动 kernel B"]
           C1 --> D1["GPU: 执行 B"]
           D1 --> E1["CPU: 启动 kernel C"]
           E1 --> F1["GPU: 执行 C"]
       end

       subgraph CUDAGraph["CUDA Graph 模式 (reduce-overhead)"]
           A2["CPU: 捕获整个图 (一次)"] --> B2["GPU: 一次重放执行 A→B→C"]
           B2 --> C2["CPU: 下一次重放..."]
           C2 --> D2["GPU: 再次执行 A→B→C"]
       end

       Eager -->|"每次启动一个 kernel"| CUDAGraph

捕获过程发生在第一次执行时：

.. code-block:: text

   首次调用 compiled_model(x):
       1. Dynamo 捕获计算图
       2. Inductor 生成 Triton kernel
       3. CUDA Graph 捕获: replay_set = torch.cuda.CUDAGraph()
          with torch.cuda.graph(replay_set):
              outputs = compiled_kernels(x)
       4. 后续调用: replay_set.replay()  # 仅需一次 CPU 操作

CUDA Graph 的限制
--------------------------

CUDA Graph 并非万能，它有以下几个重要限制：

.. list-table::
   :header-rows: 1

   * - 限制
     - 原因
     - 应对方案
   * - 输入输出地址固定
     - Graph 捕获时固定了所有 tensor 的指针地址
     - 每次重放前需确保输入 tensor 的地址不变（或重新捕获）
   * - 不支持动态控制流
     - 图结构在捕获时已固定
     - 避免在编译区域内使用 ``if`` / ``for`` 依赖 tensor 值
   * - 不支持 CPU 操作
     - Graph 内只能包含 GPU kernel
     - 将 CPU 操作（如采样）移到编译区域外
   * - 不支持某些算子
     - 部分 PyTorch 算子未适配 CUDA Graph
     - 升级 PyTorch 版本，或使用 ``torch.compiler.disable`` 排除

在 LLM 场景下，CUDA Graph 最关键的约束是输入指针固定。由于 LLM 的 KV Cache 大小不断增长，每次 decode step 的输入形状不同，这会导致 CUDA Graph 需要重新捕获。Inductor 对此做了特殊处理——当检测到输入形状变化时，会自动触发重新捕获，这比完全重新编译要快得多（毫秒级 vs 分钟级）。

通过 ``TORCH_LOGS`` 观察 CUDA Graph 行为：

.. code-block:: bash

   TORCH_LOGS="+cudagraphs" python llm_example.py

你会看到类似以下的日志：

.. code-block:: text

   [cudagraphs] 正在为模型捕获 CUDA Graph...
   [cudagraphs] Graph 捕获完成: 包含 47 个 kernel
   [cudagraphs] 重放成功: 跳过 47 次 kernel launch
   [cudagraphs] 输入形状变化，触发重新捕获...

.. tip::

   使用 ``mode="reduce-overhead"`` 时，可以通过 ``torch._inductor.config.triton.cudagraphs = False`` 禁用 CUDA Graph，单独评估其贡献。如果发现 CUDA Graph 因频繁的形状变化而反复重捕获，建议启用 ``dynamic=True`` 或使用固定长度的 padding。
