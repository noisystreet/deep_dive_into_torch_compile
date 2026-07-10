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

在 ``max-autotune`` 模式下，Inductor 会枚举多组 tiling 配置，运行微基准测试后选择最优组合。autotune 的过程由 ``autotune_process.py`` 管理，在子进程中异步执行。

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

对于矩阵乘法和卷积，Inductor 使用预定义的 Triton kernel 模板（ ``TemplateBuffer`` ）。这些模板针对特定问题规模调优，通常比通用的 pointwise/reduction 代码生成更高效。

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
