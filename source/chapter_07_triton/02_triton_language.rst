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
     - ``mask`` 控制哪些元素加载，``other`` 指定 masked-out 的填充值
   * - ``tl.store(pointer, value, mask)``
     - 将数据存储到全局内存
     - ``mask`` 控制哪些元素存储
   * - ``tl.atomic_add(pointer, value, mask)``
     - 原子加法
     - 用于跨 program 的归约
   * - ``tl.arange(start, end)``
     - 生成连续的整数序列
     - 常用于计算偏移量

``mask`` 参数是 Triton 的关键设计。由于数据块可能超出数组边界，必须通过 mask 来避免越界访问：

.. code-block:: python

   offsets = block_start + tl.arange(0, BLOCK_SIZE)
   mask = offsets < n_elements       # 边界检查
   x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

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

Triton 使用 ``tl.program_id`` 获取当前 program（block）的 ID，类似于 CUDA 中的 ``blockIdx``：

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

改变 ``constexpr`` 参数的值会生成不同的编译结果。这就是 Triton autotune 的工作方式——Inductor 的 autotune 进程会枚举多组 ``constexpr`` 参数（如不同的 ``BLOCK_SIZE``），为每组生成一个 kernel 并进行基准测试。
