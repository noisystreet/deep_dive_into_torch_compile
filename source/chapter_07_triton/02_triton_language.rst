.. _triton-language:

================
Triton 语言基础
================

Triton 的编程模型基于 Python，通过 ``@triton.jit`` 装饰器将 Python 函数编译为 GPU kernel。这一节介绍 Triton 语言的核心概念和 API。

@triton.jit 装饰器
========================

``@triton.jit`` 告诉 Triton 编译器：这个函数需要被编译为 GPU kernel，而不是在 Python 解释器中执行。所有被装饰函数内部的 ``tl.*`` 调用都会在编译时被解析为 GPU 指令。

.. code-block:: python

   import triton
   import triton.language as tl

   @triton.jit
   def my_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
       ...

.. note::

   ``tl.constexpr`` 标注的参数在编译时必须是常量。Triton 编译器使用这些常量来展开循环和优化内存访问。改变 ``constexpr`` 参数的值会触发 kernel 的重新编译。

核心 API
============

内存操作
--------------

.. list-table::
   :header-rows: 1

   * - API
     - 作用
     - 说明
   * - ``tl.load(pointer, mask, other)``
     - 从全局内存加载数据
     - ``mask`` 控制哪些元素加载， ``other`` 指定 masked-out 的填充值
   * - ``tl.store(pointer, value, mask)``
     - 将数据存储到全局内存
     - ``mask`` 控制哪些元素存储
   * - ``tl.atomic_add(pointer, value, mask)``
     - 原子加法
     - 用于跨 program 的归约
   * - ``tl.arange(start, end)``
     - 生成连续的整数序列
     - 常用于计算偏移量

``mask`` 参数是 Triton 的关键设计。由于数据块可能超出数组边界，必须通过 mask 来避免越界访问。理解 mask 的工作原理对写出正确的 Triton kernel 至关重要：

.. code-block:: python

   offsets = block_start + tl.arange(0, BLOCK_SIZE)
   mask = offsets < n_elements       # 边界检查
   x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

mask 的底层实现是 GPU 的 **谓词执行（Predicated Execution）**。当 warp 内的 32 个线程执行 ``tl.load`` 时，所有线程都发射内存加载指令——即使 mask 为 false 的线程也不会跳过指令。mask 只控制加载结果是否被丢弃：
- mask=True：加载结果正常写入目标寄存器
- mask=False：加载结果被丢弃，目标寄存器写入 ``other`` 参数的值

这意味着 mask **不会节省内存带宽**——被 mask 掉的线程仍然向内存控制器发起请求，只是其结果不会被使用。带宽浪费的程度取决于 mask 为 false 的线程在 warp 中的分布：如果 false 线程集中在 warp 尾部（典型的边界情况），前半部分线程的加载仍然是完全合并的；如果 false 线程在 warp 内随机分布，则可能导致内存事务的严重浪费。

``other`` 参数的行为也值得注意：当 mask=False 时，Triton 编译器插入的是 select 指令而非 predicated store。select 指令根据 mask 选择 ``other`` 或真正的加载结果，这种实现避免了一致性 GPU 上不同线程间的控制流分歧，但代价是多了一条 select 指令的发射和一次额外的寄存器写入。

理解 mask 的内存语义对于性能优化很重要：如果一个 kernel 中 50% 的线程因 mask 被屏蔽，实际带宽利用率只有峰值的一半。这就是为什么选择适当的 ``BLOCK_SIZE`` 让 mask false 线程占比最小化是一个重要的 autotune 目标。

数学运算
--------------

Triton 提供了丰富的逐元素数学函数，用法与 NumPy/PyTorch 类似：

.. code-block:: python

   # 基本运算
   c = a + b        # 逐元素加法
   d = a * b        # 逐元素乘法
   e = a / b        # 逐元素除法

   # 数学函数
   sin_x = tl.sin(x)
   cos_x = tl.cos(x)
   exp_x = tl.exp(x)
   sqrt_x = tl.sqrt(x)

   # 归约操作
   sum_x = tl.sum(x, axis=0)     # 求和
   max_x = tl.max(x, axis=0)     # 最大值
   min_x = tl.min(x, axis=0)     # 最小值
   argmax = tl.argmax(x, axis=0)

program_id 与网格
=========================

Triton 使用 ``tl.program_id`` 获取当前 program（block）的 ID，类似于 CUDA 中的 ``blockIdx`` ：

.. code-block:: python

   pid_0 = tl.program_id(axis=0)  # 类似于 CUDA 的 blockIdx.x
   pid_1 = tl.program_id(axis=1)  # 类似于 blockIdx.y
   pid_2 = tl.program_id(axis=2)  # 类似于 blockIdx.z

网格（grid）在 kernel launch 时指定：

.. code-block:: python

   grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # 一维网格
   kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

   grid = (grid_x, grid_y)  # 二维网格
   kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

数据类型
============

Triton 支持常见的数据类型，类型标注在 kernel 参数中自动推断：

.. list-table::
   :header-rows: 1

   * - Triton 类型
     - 对应 PyTorch 类型
     - 备注
   * - ``tl.float32``
     - ``torch.float32``
     - 默认浮点类型
   * - ``tl.float16``
     - ``torch.float16``
     - 半精度
   * - ``tl.bfloat16``
     - ``torch.bfloat16``
     - BF16
   * - ``tl.int32``
     - ``torch.int32``
     - 默认整数类型
   * - ``tl.int64``
     - ``torch.int64``
     - 64 位整数

可以使用 ``tl.cast`` 进行类型转换：

.. code-block:: python

   x_f32 = tl.cast(x_f16, tl.float32)

constexpr 参数
====================

``tl.constexpr`` 类型的参数在编译时确定，允许 Triton 编译器进行循环展开和常量折叠：

.. code-block:: python

   @triton.jit
   def kernel(BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr):
       # BLOCK_SIZE 在编译时确定
       offsets = tl.arange(0, BLOCK_SIZE)  # 会被展开
       ...

改变 ``constexpr`` 参数的值会生成不同的编译结果。这就是 Triton autotune 的工作方式——Inductor 的 autotune 进程会枚举多组 ``constexpr`` 参数（如不同的 ``BLOCK_SIZE`` ），为每组生成一个 kernel 并进行基准测试。

高级 tl.* API
=====================

除了基础的内存操作和数学运算外，Triton 还提供了一系列高级 API，用于处理更复杂的计算模式。

矩阵乘法原语 tl.dot
----------------------

``tl.dot`` 是 Triton 中最重要的高级 API 之一。它执行分块矩阵乘法，并自动利用 NVIDIA Tensor Core：

.. code-block:: python

   @triton.jit
   def kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
       # a: (BLOCK_SIZE, BLOCK_SIZE), b: (BLOCK_SIZE, BLOCK_SIZE)
       acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
       
       for k in range(0, K, BLOCK_SIZE):
           a = tl.load(a_ptr + offsets)
           b = tl.load(b_ptr + offsets)
           acc = tl.dot(a, b, acc)  # Tensor Core 上的矩阵乘法

       tl.store(c_ptr + offsets, acc)

``tl.dot`` 的"自动利用 Tensor Core"意味着编译器将每条 ``tl.dot`` 调用映射为 NVIDIA 的 ``mma.sync``（矩阵乘累加）PTX 指令。在这个过程中，warp 内的 32 个线程协作完成一个分块矩阵乘法：每个线程持有输出矩阵的一部分（通常是 16 个元素），线程之间通过 warp 级别的 shuffle 指令交换部分和。开发者不需要关心这些线程间的协作细节——编译器根据 ``BLOCK_SIZE`` 自动决定每个线程负责的元素数量和同步策略。

这种自动化也带来了约束：``tl.dot`` 要求 ``BLOCK_SIZE`` 是 16 的倍数（在 Ampere 架构上），因为 ``mma.sync`` 指令的硬件 tile 大小为 16×16。如果传入的 block 形状不是 16 的倍数，编译器需要插入额外的 padding 或分块处理逻辑，可能损失性能。

``tl.dot`` 支持的可选参数包括：

- ``input_precision`` ：控制内部累加精度（"ieee"、"tf32"、"tf32x3"）。"ieee" 严格遵循 IEEE 754 标准，最慢但最精确；"tf32" 使用 Tensor Core 的 TF32 格式，在 Ampere 上提供接近 FP32 的动态范围但只需 1/4 的带宽；"tf32x3" 使用三次 TF32 累加来提高精度
- ``max_num_imprecise_acc`` ：允许的不精确累加次数

选择 ``input_precision`` 需要权衡模型精度和计算吞吐：对于训练场景，通常推荐 ``input_precision="ieee"``；对于推理场景，``"tf32"`` 通常可以在不显著损失精度的情况下获得 2-3 倍的矩阵乘法加速。

``acc`` 参数的显式传递是 Triton 设计的一个关键细节。它让编译器能够追踪累加值的精度和生命周期——编译器知道 ``acc`` 是一个跨 ``tl.dot`` 调用持续存在的累加器，从而可以优化寄存器的分配策略（将累加器值保持在寄存器中而非溢出到 local memory）。这也使 Triton 支持更灵活的融合模式：你可以对 ``tl.dot`` 的结果进行逐元素操作后，再传给下一个 ``tl.dot``，编译器仍然能正确追踪数据流。

原子操作
--------------

除了 ``tl.atomic_add`` ，Triton 还支持 ``tl.atomic_max`` 、 ``tl.atomic_min`` 和 ``tl.atomic_xchg`` ：

.. code-block:: python

   @triton.jit
   def atomic_kernel(ptr, value):
       # 多个 program 可以安全地累加到同一个位置
       old = tl.atomic_add(ptr, value)
       
       # 原子最大值
       old = tl.atomic_max(ptr, value)

原子操作通常用于跨 block 的归约（如计算全局最大值）、直方图统计等场景。

条件选择 tl.where
--------------------

``tl.where(condition, x, y)`` 根据条件从两个张量中选择元素，类似于 PyTorch 的 ``torch.where`` ：

.. code-block:: python

   @triton.jit
   def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
       
       # tl.where 实现逐元素条件
       output = tl.where(x > 0, x, 0.0)
       
       tl.store(output_ptr + offsets, output, mask=offsets < n_elements)

``tl.where`` 是实现条件运算的主要方式，它比使用 ``if`` 语句更高效（因为它在编译时生成无分支的 select 指令）。

排序 tl.sort
----------------

``tl.sort`` 对块内的元素进行排序：

.. code-block:: python

   @triton.jit
   def sort_kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets)
       
       # 按升序排序
       sorted_x = tl.sort(x, dim=0)
       
       tl.store(output_ptr + offsets, sorted_x)

排序在 Top-K 选择、中值滤波等 kernel 中非常有用。Triton 编译器为排序生成高效的 GPU 归并排序或双调排序代码。

tl.ravel 与 tl.broadcast
---------------------------

``tl.ravel`` 将多维张量展平为一维。 ``tl.broadcast`` 显式地将张量广播到指定形状：

.. code-block:: python

   @triton.jit
   def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
       # 创建二维偏移量
       row_offsets = tl.arange(0, 16)[:, None]  # (16, 1)
       col_offsets = tl.arange(0, 16)[None, :]   # (1, 16)
       
       # 显式广播
       row_bcast = tl.broadcast(row_offsets, (16, 16))  # (16, 16)
       col_bcast = tl.broadcast(col_offsets, (16, 16))  # (16, 16)
       
       # 展平
       flat = tl.ravel(row_bcast + col_bcast)  # (256,)

在大多数情况下，Triton 的隐式广播（通过维度处理）已经足够。显式 ``tl.broadcast`` 主要用于需要清晰表达广播意图的场景。

精度处理与类型提升规则
===============================

Triton 有一套明确的类型提升规则，理解这些规则对于编写数值稳定的 kernel 至关重要。

隐式类型提升
------------------

当不同精度的操作数参与同一运算时，Triton 遵循 "向更高精度提升" 的规则：

.. code-block:: python

   @triton.jit
   def kernel():
       a = tl.zeros((1024,), dtype=tl.float16)   # fp16
       b = tl.zeros((1024,), dtype=tl.float32)   # fp32
       c = a + b  # a 被隐式提升为 float32，结果为 float32

       d = tl.zeros((1024,), dtype=tl.int32)     # int32
       e = tl.zeros((1024,), dtype=tl.float32)   # fp32
       f = d + e  # d 被隐式提升为 float32，结果为 float32

提升优先级： ``float32 > float16 == bfloat16 > int32 > int16 > int8`` 。

需要注意的是， ``float16`` 和 ``bfloat16`` 之间没有隐式提升——混合使用这两种类型会导致编译错误，必须显式转换。

混合精度技巧
----------------

在矩阵乘法中，通常的做法是以较低精度加载数据，以较高精度累加：

.. code-block:: python

   @triton.jit
   def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
       # 以 fp16 加载
       a = tl.load(a_ptr + offsets).to(tl.float16)
       b = tl.load(b_ptr + offsets).to(tl.float16)
       
       # 以 fp32 累加
       acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
       acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc)

       # 结果转回 fp16 存储
       tl.store(c_ptr + offsets, acc.to(tl.float16))

这种模式在保证计算精度的同时，减少了全局内存带宽需求。

.. tip::

   **Tensor Core 的精度要求。**
   ``tl.dot`` 对输入精度有特定要求。在 NVIDIA Ampere 架构上， ``tl.dot`` 接受 ``float16`` 或 ``bfloat16`` 输入，以 ``float32`` 累加。在 Hopper 架构上，还支持 ``float8`` 输入（通过 ``tl.float8e4nv`` 和 ``tl.float8e5b16`` 类型）。

tl.constexpr 高级模式
=============================

``tl.constexpr`` 不仅仅是编译时常量，它还可以用于实现条件编译和编译时多态。

条件编译
----------

通过 constexpr 参数，可以在编译时选择不同的代码路径：

.. code-block:: python

   @triton.jit
   def flexible_kernel(x_ptr, output_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr,
                       USE_RELU: tl.constexpr,
                       USE_BIAS: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
       
       # 条件编译：不满足条件的分支在编译时被删除
       if USE_BIAS:
           bias = tl.load(bias_ptr + offsets, mask=offsets < n_elements)
           x = x + bias
       
       if USE_RELU:
           x = tl.where(x > 0, x, 0.0)
       
       tl.store(output_ptr + offsets, x, mask=offsets < n_elements)

在编译时， ``USE_RELU=True, USE_BIAS=False`` 会生成一个仅包含 ReLU 的 kernel，不包含 bias 相关的指令。这比运行时条件判断更高效，因为 GPU 上的分支发散（branch divergence）会显著降低性能。

编译时多态
--------------

``tl.constexpr`` 还可以用于根据张量维度选择算法：

.. code-block:: python

   @triton.jit
   def adaptive_kernel(x_ptr, output_ptr, DIM: tl.constexpr):
       if DIM == 1:
           # 一维处理逻辑
           offsets = tl.arange(0, 1024)
           x = tl.load(x_ptr + offsets)
       else:
           # 二维处理逻辑
           offsets = tl.arange(0, 1024)[:, None] * stride + tl.arange(0, 1024)[None, :]
           x = tl.load(x_ptr + offsets)
       
       tl.store(output_ptr + offsets, x)

在 Inductor 生成的代码中，这种模式非常常见——Inductor 根据张量的秩（rank）设置 ``constexpr`` 参数，让 Triton 编译器生成针对特定维度的优化代码。

共享内存管理
====================

Triton 自动管理 shared memory，但提供了若干机制让开发者可以影响 shared memory 的使用方式。

tl.max_contiguous
---------------------

``tl.max_contiguous`` 用于检查指针偏移量中连续元素的最大数量。这个信息可以帮助 Triton 编译器生成更好的内存访问指令：

.. code-block:: python

   @triton.jit
   def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       # 检查连续元素数量
       contiguous = tl.max_contiguous(offsets, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets)  # 编译器根据连续信息优化

在大多数场景下，编译器可以自动推断连续访问模式，因此 ``tl.max_contiguous`` 主要用于编译器无法自动推断的复杂访问模式。

tl.multiple_of 装饰器
-------------------------

``tl.multiple_of`` 是一个提示性装饰器，告诉编译器某个张量的维度是特定数值的倍数。这可以帮助编译器生成更简单的边界检查代码：

.. code-block:: python

   @triton.jit
   def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       # 提示编译器 BLOCK_SIZE 是 16 的倍数
       # 编译器可以省略对齐检查
       tl.store(output_ptr + offsets, x)

这些提示在 Inductor 生成的代码中大量使用。Inductor 知道张量的对齐属性，会生成 ``tl.multiple_of`` 注释来帮助 Triton 编译器生成更高效的代码。

块指针 API
==================

Triton 较新版本引入了块指针（block pointer）API，为分块内存访问提供了更简洁、更高效的接口。这些 API 生成更少的地址计算指令，并且在 Hopper 架构上可以直接利用 TMA（Tensor Memory Accelerator）硬件单元。

tl.make_block_ptr
---------------------

``tl.make_block_ptr`` 创建一个块指针，描述一个张量的分块视图：

.. code-block:: python

   @triton.jit
   def kernel(
       a_ptr, b_ptr, c_ptr,
       M, N, K,
       BLOCK_SIZE: tl.constexpr,
   ):
       # 创建块指针
       a_block_ptr = tl.make_block_ptr(
           base=a_ptr,
           shape=(M, K),       # 张量形状
           strides=(K, 1),     # 步幅
           offsets=(0, 0),     # 起始偏移
           block_shape=(BLOCK_SIZE, BLOCK_SIZE),  # 块形状
           order=(1, 0),       # 访问顺序（列优先）
       )
       
       b_block_ptr = tl.make_block_ptr(
           base=b_ptr,
           shape=(K, N),
           strides=(N, 1),
           offsets=(0, 0),
           block_shape=(BLOCK_SIZE, BLOCK_SIZE),
           order=(1, 0),
       )

tl.advance
--------------

``tl.advance`` 在迭代中前进块指针：

.. code-block:: python

       c_block_ptr = tl.make_block_ptr(
           base=c_ptr,
           shape=(M, N),
           strides=(N, 1),
           offsets=(0, 0),
           block_shape=(BLOCK_SIZE, BLOCK_SIZE),
           order=(1, 0),
       )
       
       acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
       
       for k in range(0, K, BLOCK_SIZE):
           # 使用块指针加载
           a = tl.load(a_block_ptr)
           b = tl.load(b_block_ptr)
           acc = tl.dot(a, b, acc)
           
           # 前进 K 维度
           a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE))
           b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE, 0))
       
       tl.store(c_block_ptr, acc)

块指针加载的优势
----------------------

与手动计算偏移量相比，块指针 API 有以下优势：

1. **代码更简洁** 。不再需要手动拼接 ``m_start[:, None] * stride_am + k_offsets[None, :] * stride_ak`` 这样的复杂下标表达式。
2. **更少的地址计算** 。编译器可以预计算块指针的地址，避免每次迭代重复计算。
3.**Hopper TMA 支持** 。在 SM90（Hopper）架构上， ``tl.make_block_ptr`` 可以被编译为 TMA 指令，利用硬件的张量内存加速器进行异步数据传输。

.. note::

   **块指针 API 的兼容性。**
   块指针 API 在 Triton 2.x 中引入，在 Triton 3.x 中全面推广。如果使用较旧版本的 Triton，可能需要回退到手动偏移量计算方式。Inductor 在检测到 Triton 版本支持块指针 API 时会自动使用它。
