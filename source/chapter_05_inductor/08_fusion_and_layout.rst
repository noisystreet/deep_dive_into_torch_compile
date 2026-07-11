.. _fusion-and-layout:

==================
融合与布局优化
==================

第 5.7 节介绍了 Scheduler 的融合算法框架。这一节我们深入具体的融合策略和布局优化——Scheduler 如何利用 IR 类型信息做更智能的融合决策，以及如何通过布局优化减少内存访问。

融合策略的分类
==================

根据 IRNode 的类型组合，Scheduler 使用不同的融合策略：

**Pointwise + Pointwise** ：最高优先级的融合。两个逐元素操作共享相同的循环范围和设备，可以安全地合并。这是最常见的融合模式。

.. code-block:: text

   融合前:
       Kernel 1: sin(x)     → buf1    [Pointwise]
       Kernel 2: cos(buf1)  → output  [Pointwise]

   融合后:
       Kernel 1: sin(x) → cos(sin(x)) → output
       # 中间结果 buf1 在寄存器中传递，不写回显存

**Pointwise + Reduction** ：有条件融合。Reduction 的输出形状小于 Pointwise 的输出，融合时需要特殊处理——通常将 Pointwise 的计算内联到 Reduction 的循环体中。

.. code-block:: text

   融合前:
       Kernel 1: relu(x)    → buf1    [Pointwise]
       Kernel 2: sum(buf1)  → output  [Reduction]

   融合后:
       # Pointwise 的循环被"吸收"到 Reduction 中
       # 减少一次全局读/写
       for i in range(...):
           val = relu(load(x, i))
           accumulator += val
       store(output, accumulator)

**Reduction + Reduction** ：根据归约维度的兼容性决定。如果两个 Reduction 的归约维度相同，可以合并为一个 kernel。

**TemplateBuffer + Pointwise** ：TemplateBuffer（如矩阵乘法）的输出可以与后续的 Pointwise 操作融合。这在 attention 的 forward 中非常常见： ``softmax(scores @ V)`` 中的 ``scores @ V`` 是 GEMM template，后面的乘法是 pointwise。

Fusion Regions
====================

Inductor 中还有一套更先进的融合框架叫做 **Fusion Regions** （位于 ``fx_passes/fusion_regions.py`` ）。它在 FX Graph 级别就进行融合规划，而不是等到 IRNode 级别。

Fusion regions 的思路是：

1. 在 FX Graph 中标记可融合的"区域"
2. 将同一区域的节点合并为一个 ``Region`` 对象
3. 对整个 region 做统一的 lowering 和 codegen

这种方法的优势是可以在 FX 级别利用更多的语义信息（如操作的数据类型、形状关系），做出比 IR 级别更准确的融合决策。目前 fusion regions 处于持续演进中。

布局优化
=============

除了操作融合，Inductor 还通过布局优化来减少内存访问开销。

**内存布局的重要性** ：在 GPU 上，全局内存的访问模式直接影响 kernel 的性能。连续内存访问可以利用 GPU 的内存合并（memory coalescing）特性，每次内存事务传输 128 字节。不连续的访问则会导致多次独立的事务，降低有效带宽。

Inductor 的 ``ir.py`` 中定义了多种布局类型：

.. code-block:: text

   Layout（抽象基类）
   ├── FixedLayout: 固定形状和步长
   ├── AsStridedLayout: view 的偏移和步长
   ├── MutationLayout: in-place 修改的布局
   ├── AliasedLayout: 别名的布局
   └── FlexibleLayout: 允许 codegen 选择最优布局

``FlexibleLayout`` 是最值得关注的。它允许代码生成器为输出 buffer 选择最优的内存布局——在生成 Triton 代码时，可以决定输出的 strides 顺序，以最大化内存访问效率。

.. code-block:: text

   输入: NHWC 格式
   计算: op1 → op2 → op3 (全部是 pointwise)
   
   默认布局（保持输入格式）:
       load(NHWC) → compute → store(NHWC)
       # 如果后续操作需要 NCHW，会产生额外的 transpose
   
   灵活布局:
       load(NHWC) → compute → store(NCHW)
       # 在 store 时直接完成布局转换，避免额外 kernel

这种优化在混合了不同内存格式的模型中特别有价值（如卷积的 NHWC vs 全连接的 NCHW）。

自动调优：max-autotune 模式下的融合
===========================================

在 ``max-autotune`` 模式下，Inductor 会做更激进的融合试探。 ``fuse_if_speedup`` 方法会：

1. 假设融合后生成一个 kernel
2. 估算融合后的运行时间（基于硬件模型或 benchmark）
3. 如果融合后的估计时间 < 融合前两个 kernel 的时间之和，则执行融合

这个基准测试是实时的——Inductor 在编译时实际运行微基准测试来测量性能。这使得融合决策比任何启发式规则都更精确。

可以通过日志观察融合决策：

.. code-block:: bash

   TORCH_LOGS="+schedule" python train.py

日志会输出：

.. code-block:: text

   [schedule] 考虑融合 A 和 B
   [schedule]   - 类型兼容: Pointwise + Pointwise ✓
   [schedule]   - 设备兼容: cuda:0 + cuda:0 ✓
   [schedule]   - 无循环依赖 ✓
   [schedule]   - 性能收益: 0.12ms → 0.08ms (1.5x) ✓
   [schedule] 融合成功

小结
======

这一节介绍了融合与布局优化的具体策略：

- **三种融合模式** ：Pointwise+Pointwise、Pointwise+Reduction、TemplateBuffer+Pointwise
- **Fusion Regions** ：在 FX Graph 级别预先规划融合区域
- **布局优化** ：通过 ``FlexibleLayout`` 让 codegen 选择最优内存布局
- **max-autotune 融合** ：基于运行时基准测试的融合决策
