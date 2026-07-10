.. _pattern-matcher:

==================
Pattern Matcher
==================

Pattern Matcher 是 Inductor 中的一个 **图变换框架** ，通过预定义的模式匹配规则来识别并替换 FX Graph 中的特定子图结构。它可以捕获 Scheduler 层面无法看到的"跨操作"优化机会。

为什么需要 Pattern Matcher？
================================

Scheduler 的融合是在 IR 层面进行的——它看到的是 ``Pointwise`` 、 ``Reduction`` 等 IR 类型，看不到更高层的语义信息。

考虑一个常见的优化：将 ``x * y + z`` 替换为融合后的 ``fma(x, y, z)`` 。这在 IR 层面是两个节点（一个 Pointwise 用于乘法、一个 Pointwise 用于加法），Scheduler 看到的是两个 pointwise 节点——它确实可以融合它们，但生成的代码是 ``load(x) * load(y) + load(z)`` ，而不是直接使用硬件 FMA 指令。

Pattern Matcher 在 FX Graph 层面运行，可以直接匹配 ``mul + add`` 的模式并替换为 ``fma`` 节点。当 lowering 看到 ``fma`` 节点时，可以生成调用硬件 FMA 指令的代码。

Pattern Matcher 的架构
=============================

Pattern Matcher 定义在 ``pytorch/torch/_inductor/pattern_matcher.py`` 中。它的架构基于：

1. **模式定义 ** ：使用 ``@register_graph_pattern`` 装饰器定义一个匹配规则
2.**匹配引擎 ** ：在 FX Graph 上遍历节点，搜索匹配的模式
3.**替换逻辑** ：匹配成功后执行替换函数

.. code-block:: python
   :caption: pytorch/torch/_inductor/pattern_matcher.py（简化示意）

   # 定义模式：匹配 "sin + cos" 的组合
   @register_graph_pattern(
       torch.add(
           torch.sin(MatchesDict()),
           torch.cos(MatchesDict()),
       ),
   )
   def sin_add_cos_pattern(match, *args, **kwargs):
       """将 sin(x) + cos(x) 替换为专用实现"""
       # match 包含匹配到的节点和参数绑定
       # 返回替换后的新节点
       ...

``MatchesDict()`` 是一个模式占位符，匹配任何表达式。整个装饰器定义了一个树形的匹配模式，当 FX Graph 中出现 ``add(sin(...), cos(...))`` 的结构时，匹配引擎会触发。

预定义模式
===============

Inductor 在 ``fx_passes/`` 目录中预定义了大量的匹配模式。以下是几类有代表性的：

** 数值精度优化 **：位于 ``fx_passes/post_grad.py``

.. code-block:: text

   模式: aten.relu(aten.conv(x, w))  →  融合为 conv_relu
   收益: 减少一次 kernel launch，relu 在 conv 的输出上就地操作

   模式: aten.add(aten.mul(x, y), z)  →  融合为 fma(x, y, z)
   收益: 利用硬件 FMA 指令，减少中间结果写回

**Attention 优化 ** ：位于 ``fx_passes/fuse_attention.py``

.. code-block:: text

   模式: softmax(attention_scores) @ V  →  融合 Flash Attention
   收益: 避免 attention 矩阵显存写入/读取，显著提升训练吞吐

   模式: 各种 SDPA (Scaled Dot-Product Attention) 变体  →  统一为 scaled_dot_product_attention
   收益: 调用 CUDA 上高度优化的 cuDNN 或 Triton kernel

**模式序列化** ：对于复杂模式（如 Flash Attention 的多种变体），Inductor 将模式序列化到 ``fx_passes/serialized_patterns/`` 中。这些是预编译的匹配规则，避免在运行时重新解析。

模式匹配的 FX Pass 执行时机
====================================

Pattern Matcher 在 Inductor 的多个 FX Pass 阶段中被调用：

.. code-block:: text

   compile_fx_inner()
       │
       ├─ pre_grad_passes()              # 在 AOTAutograd 之前
       │   ├─ pattern_matcher 匹配
       │   └─ 其他 FX pass
       │
       ├─ AOTAutograd 求导和分区
       │
       ├─ post_grad_passes()             # 在 lowering 之前（主要匹配阶段）
       │   ├─ pattern_matcher 匹配
       │   ├─ fuse_attention.py 匹配     # attention 专用模式
       │   └─ 其他优化 pass
       │
       └─ Lowering

主要的模式匹配发生在 ``post_grad_passes`` 阶段，因为此时图已从 joint graph 中分离出来，前向和反向子图可以独立优化。

自定义匹配模式
===================

用户可以通过 ``torch._inductor.pattern_matcher`` 注册自己的匹配模式。这在自定义后端场景中非常有用：

.. code-block:: python

   from torch._inductor.pattern_matcher import register_graph_pattern
   from torch._inductor.fx_utils import get_aten_target

   @register_graph_pattern(
       torch.add(torch.sin(MatchesDict()), torch.cos(MatchesDict())),
       pass_dict=post_grad_passes,
   )
   def sin_plus_cos(match, *args,**kwargs):
       """自定义的 sin+cos 融合实现"""
       # 这里可以插入自定义的 lowering 逻辑
       ...

这种方式让用户在不需要修改 Inductor 核心代码的情况下，扩展 Inductor 的优化能力。

小结
======

这一节介绍了 Pattern Matcher：

- ** 定位 **：在 FX Graph 层面识别并替换特定子图结构，捕获 Scheduler 无法看到的优化机会
- ** 架构 **： ``@register_graph_pattern`` 定义模式 → 匹配引擎搜索 → 替换函数执行
- ** 应用场景 **：数值精度优化（conv+relu 融合、FMA）、注意力优化（Flash Attention 融合）
- ** 执行时机** ：主要在 ``post_grad_passes`` 阶段运行
