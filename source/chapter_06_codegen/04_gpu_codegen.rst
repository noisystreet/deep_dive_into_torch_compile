.. _gpu-codegen:

===============
GPU 代码生成
===============

.. note::

   **Inductor 生成的 Triton 代码看起来像手写的吗？ **
   如果你曾经看过 Inductor 生成的 Triton kernel 源码，你会发现它的风格和手写 Triton kernel 非常相似——同样的 ``tl.load`` 、 ``tl.store`` 、掩码检查。这是因为 Inductor 的 ``TritonKernel`` 是逐行生成代码的，相当于一个"编译器即代码模板"的过程。但 Inductor 永远不会生成 ``tl.dot`` 调用——它会将矩阵乘法交给 ``TemplateBuffer`` 处理，使用高度优化的 Triton GEMM 模板，而不是让 codegen 去拼凑矩阵乘法。这是 Inductor 的一个关键设计：** 逐元素和归约操作由编译器生成，而矩阵乘法则交给预定义模板** 。

TritonScheduling 和 TritonKernel 负责将 IRNode 翻译为 Triton 代码——这是 Inductor 在 GPU 上的默认代码生成后端。

TritonScheduling 架构
===========================

TritonScheduling（在 ``codegen/triton.py`` 中）继承自 ``SIMDScheduling`` ，是 GPU 代码生成的核心。它的关键特性：

.. code-block:: python
   :caption: pytorch/torch/_inductor/codegen/triton.py

   class TritonScheduling(SIMDScheduling):
       """GPU kernel 代码生成的调度后端"""

       kernel_type: type[Any] = TritonKernel
       backend_features = OrderedSet([
           BackendFeature.FOREACH,
           BackendFeature.INPLACE_BUFFERS,
           BackendFeature.SCAN,
           BackendFeature.TRITON_TEMPLATES,
           BackendFeature.TUPLE_REDUCTION,
           ...
       ])

``TritonKernel`` 负责实际的代码生成。当接收到一个 ``FusedSchedulerNode`` 时，TritonKernel 执行以下步骤：

.. code-block:: text

   TritonKernel.codegen(node)
       │
       ├─ 1. 确定循环范围和 tiling
       │      计算 block size、grid size
       │      基于输入形状和硬件属性
       │
       ├─ 2. 生成 kernel 签名
       │      def kernel_name(
       │          x_ptr, y_ptr, output_ptr,
       │          n_elements, BLOCK_SIZE: tl.constexpr,
       │      )
       │
       ├─ 3. 生成加载代码
       │      x = tl.load(x_ptr + offsets, mask=mask)
       │      y = tl.load(y_ptr + offsets, mask=mask)
       │
       ├─ 4. 生成计算代码
       │      将 IRNode 的 inner_fn 
       │      翻译为 Triton 操作
       │      sin_x = tl.sin(x)
       │
       ├─ 5. 生成存储代码
       │      tl.store(output_ptr + offsets, result, mask=mask)
       │
       └─ 6. 注册到 wrapper
              将生成的 Triton kernel 注册到 PythonWrapperCodegen

Inductor 选择这种"逐行字符串拼接"的代码生成风格，而不是构建更复杂的编译器 IR，是一个 deliberate 的工程权衡。这种方式的优势是 **透明性和调试性**：生成的 Triton 代码可以直接阅读、逐行断点调试，甚至可以手动复制出来独立运行和修改。对于编译器开发者来说，这意味着可以肉眼验证生成的代码是否正确。代价是编译器很难做跨语句的全局优化——因为代码以字符串形式存在，任何需要"看到整个 kernel"才能做的优化（如全局寄存器分配）都比在正式 IR 上困难得多。这个权衡符合 Inductor 的整体设计哲学（见第 5.1 节）：**用工程速度换编译器理论的完备性**。

Triton Kernel 的结构
=========================

一个典型的由 Inductor 生成的 Triton kernel 如下：

.. code-block:: python

   @triton.jit
   def fused_kernel(
       x_ptr, y_ptr, output_ptr,
       n_elements,
       BLOCK_SIZE: tl.constexpr,
   ):
       pid = tl.program_id(axis=0)
       block_start = pid * BLOCK_SIZE
       offsets = block_start + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements

       # 加载
       x = tl.load(x_ptr + offsets, mask=mask)
       y = tl.load(y_ptr + offsets, mask=mask)

       # 计算（融合后的多个 IRNode）
       sin_x = tl.sin(x)
       cos_y = tl.cos(y)
       result = sin_x + cos_y

       # 存储
       tl.store(output_ptr + offsets, result, mask=mask)

``triton.jit`` 装饰器将 Python 函数编译为 GPU kernel。Triton 编译器在编译时会推断每个变量的类型和形状，并生成优化的 PTX 代码。

Tiling 策略
===============

TritonScheduling 的 tiling 策略决定了 kernel 的并行粒度。核心参数包括：

- **BLOCK_SIZE** ：每个 program（thread block）处理的元素数量
- **num_warps** ：每个 program 的 warp 数量（默认 4 或 8）
- **num_stages** ：软件流水线阶段数（默认 3 或 4）

在 ``default`` 模式下，tiling 参数基于启发式规则决定：

.. code-block:: text

   元素数 < 1024:  BLOCK_SIZE = 元素数（单 program）
   元素数 < 4096:  BLOCK_SIZE = 1024
   元素数 < 16384: BLOCK_SIZE = 2048
   其他:           BLOCK_SIZE = 4096

这些启发式规则的背后是一个根本的权衡：**更大的 BLOCK_SIZE 意味着每个 program 处理更多数据，减少了 program 总数从而降低 launch 开销，但每个 program 需要更多寄存器和共享内存，可能降低 occupancy（每个 SM 上同时运行的 program 数量）**。Inductor 的默认策略偏向保守——优先保证 occupancy 不下降，而不是追求单个 program 的最高吞吐。这是因为在大多数逐元素操作中，内存带宽是瓶颈，而非计算能力。增加 BLOCK_SIZE 不会改善内存带宽利用率（带宽已经被 warp 级的内存合并充分利用），反而可能因降低 occupancy 而损害延迟隐藏能力。

在 ``max-autotune`` 模式下，Inductor 会枚举多组 tiling 配置，运行微基准测试后选择最优组合。autotune 的过程由 ``autotune_process.py`` 管理，在子进程中异步执行。autotune 本质上是用编译时间换运行性能——它通过实际运行来找到最优配置，绕过了启发式规则无法覆盖的特殊场景。

Reduction 的代码生成
=========================

包含 Reduction 的 kernel 比纯 Pointwise 复杂。Reduction 需要跨 program 协作。Inductor 生成两种风格的 Reduction kernel：

**跨 program reduction** （默认）：每个 program 处理一部分数据，通过 ``tl.atomic_add`` 或分阶段归约实现。

.. code-block:: python

   @triton.jit
   def reduction_kernel(x_ptr, output_ptr, ...):
       pid = tl.program_id(axis=0)
       
       # 加载数据块
       offsets = block_start + tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets, mask=mask)
       
       # 块内归约
       block_sum = tl.sum(x, axis=0)
       
       # 跨 program 归约（使用原子操作或分阶段）
       if pid == 0:
           # 收集所有 block 的局部结果
           ...

**Warp-level reduction** （小尺寸优化）：当输入尺寸小于 warp size（32）时，整个归约可以在单个 warp 内完成。

TemplateBuffer 的代码生成
===============================

对于矩阵乘法和卷积，Inductor 使用预定义的 Triton kernel 模板（ ``TemplateBuffer`` ）。这些模板针对特定问题规模调优，通常比通用的 pointwise/reduction 代码生成更高效。这是 Inductor 代码生成中的一个关键分层决策：**逐元素和归约操作由编译器自动生成，矩阵乘法和卷积交给手写优化的模板**。

为什么需要这种分层？根本原因在于两类操作的计算特征完全不同：

- **逐元素和归约操作** 的计算模式是规则的、可预测的——每个元素做同样的操作，数据访问模式是连续的。编译器可以通过几条通用规则（如"生成循环 → 生成加载 → 生成计算 → 生成存储"）覆盖绝大多数场景，不需要为每种操作单独优化。
- **矩阵乘法和卷积** 的计算模式涉及复杂的数据复用和分块策略：矩阵乘法需要将输入分块到 shared memory、利用 Tensor Core 的 warp 级协作、管理软件流水线。这些优化高度依赖于问题规模（M、N、K 的值）和硬件特性（ shared memory 大小、Tensor Core 版本）。通用编译器无法在不进行大量搜索的情况下为每种规模生成最优代码。

这个分层决策的后果是：Inductor 能够 **融合逐元素操作到矩阵乘法的 epilogue 中**（例如将 ``matmul + bias + relu`` 融合为一个 kernel），但不能修改矩阵乘法内核本身的分块策略。融合发生在模板生成的 Triton 代码层面——模板预留了 epilogue 代码的"钩子"，Inductor 将融合的逐元素操作填入这个钩子。这意味着 epilogue 融合是模板支持的、有约束的融合，而非编译器通用的自由融合。

Triton 模板的代码生成路径如下：

.. code-block:: text

   lowering 遇到 aten.mm
       │
       ├─ 生成 TemplateBuffer
       │      template = "atlas_gemm"
       │      inputs = [x, y]
       │      output_shape = [M, K, N]
       │
       └─ codegen 阶段
              TritonKernel.codegen_template()
              加载预定义的 Triton GEMM 模板
              填入 M、N、K 参数
              输出优化后的 Triton GEMM kernel

这些模板位于 ``codegen/triton.py`` 中的 ``TritonTemplateKernel`` 类和相关文件中。

Benchmark 与自动调优
============================

Triton kernel 的性能高度依赖于 tiling 参数。Inductor 提供了 ``codegen_kernel_benchmark`` 方法来对生成 kernel 进行微基准测试：

.. code-block:: text

   生成的 kernel 代码
       │
       ▼
   TritonKernel.codegen_kernel_benchmark()
       │
       ├─ 创建基准测试输入
       │      x = torch.randn(...)
       │      y = torch.randn(...)
       │
       ├─ 编译 kernel
       │      compiled_kernel = triton.compile(kernel_src)
       │
       ├─ 运行多次，测量时间
       │      for _ in range(100):
       │          compiled_kernel(x, y, output, ...)
       │
       └─ 返回 benchmark 结果（GB/s 或 us）

基准测试结果可以被 ``autotune_process.py`` 用于选择最优配置。
