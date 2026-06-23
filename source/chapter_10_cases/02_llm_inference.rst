.. _llm-inference:

==============================
案例 2：LLM 推理优化
==============================

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
   print(f"Compiled 生成时间: {time.time() - start:.2f}s)

关键优化点
==============

**KV Cache**。LLM 推理的核心优化是 KV Cache——每次生成新 token 时，只计算最新的 query 和 key/value，复用之前 step 的结果。在 eager 模式下，KV Cache 会带来显着的加速。在 torch.compile 下，需要确保 KV Cache 的实现不会被 graph break 打断。

**优化序列长度**。LLM 推理时，序列长度在每次迭代中递增。如果使用动态形状：

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

**算子融合优化**。Transformer 中的常见融合模式：

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

- **kernel 数量减少**：融合前可能有 50+ 个 kernel，融合后可减少到 10-15 个
- **显存访问减少**：中间结果写回显存的次数下降
- **Tensor Core 利用率**：矩阵乘法使用 ``tl.dot`` 调用 Tensor Core

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

这个优化由 Pattern Matcher（第 5.7 节）的 ``@register_graph_pattern`` 自动完成。

Flash Attention
====================

对于支持 Flash Attention 的 GPU（Sm80+），Inductor 会自动用 Flash Attention 替换标准的 attention 实现：

.. code-block:: python

   # Inductor 默认启用 Flash Attention
   # 可以通过配置禁用
   torch._inductor.config.fuse_attention = False

Flash Attention 的核心优势是**不需要将完整的 attention 矩阵写入显存**，这对长序列场景格外重要。

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
