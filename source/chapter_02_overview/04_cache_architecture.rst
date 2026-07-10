.. _cache-architecture:

======================
编译缓存架构
======================

第 2.1 节提到过一次编译缓存——"编译结果被缓存，后续调用直接命中缓存跳过编译"。第 2.1 节的 **正确性优先 ** 原则在这里体现得最直接：缓存是为了快，但 guard 失败必须重编译，缓存满了必须 fallback eager——**绝不为性能牺牲语义正确 ** 。

但实际上，torch.compile 的缓存不是一个单一的组件，而是**三个组件各自拥有独立的缓存策略 ** 。理解这三层缓存如何协作，是理解 torch.compile 运行时行为的关键。它与第 3.8 节的符号形状、第 3.5 节的 guard 共同回答同一个问题：**如何在输入变化时，既复用编译结果，又不返回错误答案？ **

三层缓存概览
=================

从用户函数被调用到最终执行编译后的代码，中间可能被多达三层缓存拦截：

.. code-block:: text

   compiled_fn(x, y)
       │
       ▼
   ┌──────────────────────────────────┐
   │  第一层：Dynamo Guard 缓存         │
   │  key: 函数身份 + guard 条件       │
   │  位置：code object 的 co_extra    │
   │  命中：直接执行已编译的函数         │
   │  未命中：触发编译流水线             │
   └──────────┬───────────────────────┘
              │ 未命中
              ▼
   ┌──────────────────────────────────┐
   │  第二层：AOTAutograd 缓存          │
   │  key: FX Graph + 分区配置         │
   │  位置：内存 cache dict            │
   │  命中：跳过联合求导和图分区         │
   │  未命中：执行 AOTAutograd          │
   └──────────┬───────────────────────┘
              │ 未命中
              ▼
   ┌──────────────────────────────────┐
   │  第三层：Inductor 代码缓存         │
   │  key: IRNode 哈希 + GPU 配置      │
   │  位置：磁盘 (TORCHINDUCTOR_CACHE) │
   │  命中：跳过代码生成，直接加载 .so   │
   │  未命中：执行代码生成 + 编译        │
   └──────────────────────────────────┘

下面分别看每一层的作用和实现。

第一层：Dynamo Guard 缓存（最快）
======================================

Dynamo 的缓存是 runtime 缓存中最关键的一层。它维护在**code object 的 co_extra 上 **——这是 CPython 为每个编译后的代码对象预留的额外存储空间。

.. code-block:: text

   函数 fn 的 code object
   ┌──────────────────────────────────────┐
   │  co_code (字节码)                     │
   │  co_consts (常量)                     │
   │  co_filename                          │
   │  ...                                  │
   │  ┌──────────────────────────────────┐ │
   │  │  co_extra (Dynamo 缓存)           │ │
   │  │  ┌──────┐ ┌──────┐ ┌──────┐     │ │
   │  │  │entry1│→│entry2│→│entry3│→...  │ │
   │  │  └──────┘ └──────┘ └──────┘     │ │
   │  │  • guard_manager                  │ │
   │  │  • compiled_code (Triton kernel)  │ │
   │  │  • next ─────────────→            │ │
   │  └──────────────────────────────────┘ │
   └──────────────────────────────────────┘

每个 cache entry 包含三要素：

1.**guard_manager**—— 检查输入是否匹配的可调用对象
2.**compiled_code**—— 编译后的可执行函数
3.**next**—— 指向下一个 entry 的指针，形成链表

当 ``compiled_fn(x, y)`` 被调用时，Dynamo 遍历这个链表，依次运行每个 entry 的 guard_manager：

.. code-block:: text

   遍历 cache 链表：
       entry1: guard_manager(x, y) 检查通过?
           yes → 执行 entry1.compiled_code(x, y)  ← 命中
           no  → 检查 entry2

       entry2: guard_manager(x, y) 检查通过?
           yes → 执行 entry2.compiled_code(x, y)  ← 命中
           no  → 检查 entry3

       ...

       全部未命中 → 重新编译 + 在链表尾部插入新 entry

guard_manager 检查的是输入张量的形状、dtype、device，以及 nn.Module 的 identity。例如，第一次调用时输入形状是 ``(32, 784)`` ，生成的 guard 是：

.. code-block:: text

   guard_manager(x, y):
       assert x.shape == (32, 784)
       assert x.dtype == torch.float32
       assert x.device == torch.device('cuda:0')
       assert y.shape == (32, 784)
       ...

当第二次调用传入 ``(64, 784)`` 时，第一个 assert 就失败，链表前进到下一个 entry。

缓存大小限制
------------------

Dynamo 有两层缓存大小限制，定义在 ``torch/_dynamo/cache_size.py`` 中：

1.**recompile_limit** （默认 8）：限制 **单个 nn.Module 实例** 的缓存条目数。这意味着同一个模型的不同实例（不同 ID）各自有 8 条缓存。
2.**accumulated_recompile_limit** （默认 256）：限制 **同一个 code object** 的总编译次数。这是总的安全阀。

为什么需要两个限制？这是因为 **ID_MATCH guard** 的存在。当 Dynamo 遇到 graph break 时，它会在 guard 中放入 nn.Module 实例的 ID 检查。如果你有 16 个不同的模型实例，每个实例的 ID 都不同，它们会在同一 code object 的缓存链表中产生 16 个 entry——每个 entry 的 guard 检查不同的 ID。如果没有 ``accumulated_recompile_limit`` ，编译次数会随着实例数量线性增长。

关于 cache 的实现细节，可以参考 ``torch/_dynamo/cache_size.py`` 中的注释（搜索 ``[Note on cache size limit]`` ）。

清空 Dynamo 缓存
-----------------------

.. code-block:: python

   import torch

   # 清空所有 code object 上的缓存
   torch.compiler.reset()

   # 等价于：
   torch._dynamo.reset()

``reset()`` 遍历所有已加载的 code object，清空其 co_extra 中的缓存链表。这在性能基准测试中非常关键——如果不 reset，第二次运行同样的代码会命中缓存，测到的是纯执行时间而不是"编译 + 执行"时间。

第二层：AOTAutograd 缓存（中等）
========================================

AOTAutograd 缓存（定义在 ``torch/_functorch/_aot_autograd/autograd_cache.py`` ）缓存的是 **联合求导和图分区的结果** 。

它的 key 由以下几部分构成：

.. code-block:: text

   AOTAutograd Cache Key:
   ┌────────────────────────────────────────┐
   │  • FX Graph 的哈希                      │
   │  • 输入张量的 meta（形状、dtype、device） │
   │  • 分区策略（min-cut 配置）              │
   │  • 是否需要 functionalization            │
   │  • autograd 配置（mode 等）              │
   └────────────────────────────────────────┘

当 Dynamo 的 guard 缓存未命中，触发完整编译后，AOTAutograd 在联合求导之前先检查缓存：

.. code-block:: text

   convert_frame 输出 FX Graph
       │
       ▼
   AOTAutograd: 开始求导
       │
       ├─ 检查 autograd_cache
       │   key = hash(FX Graph) + hash(inputs) + hash(config)
       │
       ├─ 命中？
       │   yes → 直接返回分区后的子图
       │   no  → 执行联合求导 → 图分区 → 更新缓存
       │
       ▼
   进入 Inductor

这个缓存的价值在于：如果同一个计算图多次出现（比如训练循环中不同 step 的图结构相同），AOTAutograd 不需要每次都重新跑自动微分。对于大模型，一次 autograd trace 可能耗时几百毫秒，缓存能显著减少重复开销。

第三层：Inductor 磁盘缓存（最慢但最持久）
================================================

Inductor 的缓存是唯一持久化的缓存。它把编译生成的 Triton/C++ 代码写入磁盘， **跨进程、跨重启都有效** 。

缓存结构
--------------

Inductor 的缓存由 ``torch/_inductor/codecache.py`` 管理。每个编译好的 kernel 被存储在 ``TORCHINDUCTOR_CACHE_DIR`` （默认是 ``$HOME/.cache/torch/inductor/`` ）下：

.. code-block:: text

   $TORCHINDUCTOR_CACHE_DIR/
   ├── {cache_key}.so              # 编译后的共享库
   ├── {cache_key}.kernel          # Triton kernel 的 .py 文件
   └── {cache_key}.meta            # 元信息（输入输出规格等）

缓存 key 的计算
------------------

Inductor 的缓存 key 是一个 SHA-256 哈希，由以下因素决定：

.. code-block:: text

   SHA256(
       IRNode 的序列化表示    +     # 计算图结构
       GPU/CPU 架构信息      +     # 如 compute capability 8.0
       CUDA 版本             +     # 如 12.4
       PyTorch 版本           +     # 如 2.12.1
       Triton 版本            +     # 如 3.1.0
       Inductor 配置选项      +     # 如 max_autotune 等
   )

这是一个相当彻底的哈希方案。只要任何一项发生变化（升级 CUDA、更换 GPU 型号、修改编译配置），缓存 key 就不同，kernel 会重新编译。

你可以通过环境变量控制缓存：

.. code-block:: bash

   # 自定义缓存目录
   TORCHINDUCTOR_CACHE_DIR=/ssd/cache torch.compile(...)

   # 禁用缓存（开发调试时）
   TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python train.py

   # 预填充缓存（多机部署时共享）
   TORCHINDUCTOR_CACHE_DIR=/shared/cache python compile_once.py
   TORCHINDUCTOR_CACHE_DIR=/shared/cache python deploy.py  # 跳过编译

分层缓存的效果
====================

用一个实际的训练循环来说明三层缓存的效果。假设模型有 10 个不同的输入形状（来自不同的数据集），每个形状训练 100 个 step：

.. code-block:: text

   Step 1:  shape A
   ┌──────────────────────────────────────────────┐
   │ Dynamo guard:  未命中（首次看到 shape A）     │
   │ AOTAutograd:   未命中（首次看到此图）          │
   │ Inductor:      未命中（首次编译）              │
   │ → 完整编译（~3s）                              │
   └──────────────────────────────────────────────┘

   Step 2-100: shape A
   ┌──────────────────────────────────────────────┐
   │ Dynamo guard:  命中！直接执行已编译的代码      │
   │ → 0ms 编译开销                                 │
   └──────────────────────────────────────────────┘

   Step 101: shape B
   ┌──────────────────────────────────────────────┐
   │ Dynamo guard:  未命中（新形状 → 新 guard）     │
   │ → 重新编译                                     │
   └──────────────────────────────────────────────┘

当训练完毕后，重新启动一个新的训练进程：

.. code-block:: text

   新进程，Step 1: shape A
   ┌──────────────────────────────────────────────┐
   │ Dynamo guard:  未命中（进程中无缓存）           │
   │ AOTAutograd:   未命中                          │
   │ Inductor:      命中！从磁盘加载 .so            │
   │ → 仅代码生成阶段被跳过，仍需 Dynamo + 求导     │
   └──────────────────────────────────────────────┘

注意：即使 Inductor 磁盘缓存命中，前两层缓存（Dynamo guard 和 AOTAutograd）在新进程中仍然是空的——它们存储在内存中。所以完整的"跨进程缓存命中"路径仍然需要走完 Dynamo 捕获和 AOTAutograd 求导，只在代码生成阶段节省时间。

这也是 torch.compile 的一个已知开销来源：大型模型启动时即使磁盘缓存命中了所有 kernel，Dynamo 的捕获和 AOTAutograd 的求导仍然需要时间。PyTorch 团队正在开发更多的持久化方案来解决这个问题（见 ``torch/compiler/_cache.py`` 中的 CacheArtifact 抽象）。

三种缓存的比较
=========================

.. list-table::
   :header-rows: 1

   * -
     - Dynamo Guard 缓存
     - AOTAutograd 缓存
     - Inductor 磁盘缓存
   * - 存储位置
     - code object co_extra（内存）
     - 内存 dict
     - 磁盘文件
   * - 生命周期
     - 进程级别
     - 进程级别
     - 持久化（跨进程）
   * - Key 类型
     - guard 条件（shape/dtype/device + ID）
     - FX Graph 哈希 + 配置
     - IRNode 哈希 + 环境
   * - 命中跳过
     - 全部编译过程
     - 联合求导 + 分区
     - 代码生成 + 编译
   * - 典型命中耗时
     - < 1μs
     - ~1ms
     - ~10ms（磁盘 I/O + 加载 .so）
   * - 核心文件
     - ``torch/_dynamo/cache_size.py``
     - ``torch/_functorch/_aot_autograd/autograd_cache.py``
     - ``torch/_inductor/codecache.py``

理解这三层缓存的协作关系，对于诊断 torch.compile 的性能问题非常有帮助：

- 如果每次调用都 miss 第一层缓存 → 检查输入形状是否过于动态
- 如果第二层频繁 miss → 检查 FX Graph 是否每次不同（可能是 graph break 导致图结构变化）
- 如果第三层 miss → 检查硬件/软件环境是否一致（特别是分布式训练中不同机器的 CUDA 版本）

关于这些问题，我们会在第 8 章调试工具和第 9 章进阶优化中专题讨论。
