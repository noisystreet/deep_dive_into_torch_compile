.. _kernel-launch:

=============
Kernel Launch
=============

代码生成完成后，Inductor 需要将生成的 kernel 组织为可执行的调用序列。这一过程由 **Wrapper 代码生成器 ** 处理——它生成 Python 或 C++ 代码，负责在运行时 launch 所有 kernel。

Wrapper 的角色
=================

Wrapper 层（即 wrapper 代码生成器）位于 Kernel 代码和运行时执行之间。它的职责是：

1.**生成 launch 顺序 ** ：按照 Scheduler 确定的依赖关系，依次调用每个 kernel
2.**管理 buffer 生命周期 ** ：分配临时 buffer、处理 buffer 复用
3.**处理输入/输出** ：将用户输入映射到 kernel 参数，将 kernel 输出映射回用户返回值

.. code-block:: text

   Scheduler 决定执行顺序：
       buf0 = kernel_1(x, y)
       buf1 = kernel_2(buf0)
       output = kernel_3(buf1)

   Wrapper 生成的代码：
       def compiled_fn(x, y):
           buf0 = torch.empty(...)      # 分配
           triton_kernel_1[(grid,)](x, y, buf0)  # launch 1
           buf1 = torch.empty(...)      # 分配
           triton_kernel_2[(grid,)](buf0, buf1)  # launch 2
           output = torch.empty(...)    # 分配
           triton_kernel_3[(grid,)](buf1, output)  # launch 3
           return output                # 返回

Wrapper 的类型
==================

Inductor 支持多种 wrapper 类型：

.. list-table::
   :header-rows: 1

   * - Wrapper 类型
     - 设备
     - 输出语言
     - 文件
   * - PythonWrapperCodegen
     - GPU / CPU
     - Python
     - ``codegen/wrapper.py``
   * - CppWrapperCpu
     - CPU
     - C++
     - ``codegen/cpp_wrapper_cpu.py``
   * - CppWrapperGpu
     - GPU
     - C++
     - ``codegen/cpp_wrapper_gpu.py``

默认情况下，GPU 使用 ``PythonWrapperCodegen`` （生成 Python 代码调用 Triton kernel），CPU 使用 ``CppWrapperCpu`` （生成 C++ 代码直接执行）。使用 C++ wrapper 可以减少 Python 解释器的参与，在某些场景下降低延迟。

GPU Kernel Launch 流程
============================

对于 GPU kernel，Wrapper 生成的 Python 代码执行以下 launch 流程：

.. code-block:: text

   Wrapper 生成的 Python 代码:
       def compiled_fn(x, y):
           │
           ├─ 1. 分配输出 buffer
           │      buf0 = torch.empty(M, N, device="cuda")
           │
           ├─ 2. 计算 grid 大小
           │      grid = (triton.cdiv(N, BLOCK_SIZE),)
           │
           ├─ 3. 启动 kernel
           │      triton_kernel[grid](
           │          x, y, buf0,
           │          N,
           │          BLOCK_SIZE=1024,
           │          num_warps=4,
           │      )
           │      # 注意：Triton kernel launch 是异步的
           │
           ├─ 4. 处理下一个 kernel（如果有关联）
           │      buf1 = torch.empty(...)
           │      next_kernel[grid2](
           │          buf0, buf1, ...
           │      )
           │
           └─ 5. 返回结果
                  return buf1

``PythonWrapperCodegen`` 在 ``codegen/wrapper.py`` 中实现，它维护了一个 ``IndentedBuffer`` 来逐行生成代码：

.. code-block:: python
   :caption: pytorch/torch/_inductor/codegen/wrapper.py（简化示意）

   class PythonWrapperCodegen:
       def __init__(self):
           self.header = IndentedBuffer()   # import 和其他前置代码
           self.kernel = IndentedBuffer()   # kernel 代码
           self.wrapper = IndentedBuffer()  # launch 代码

       def generate(self):
           """生成完整的 wrapper 代码"""
           return "\n".join([
               self.header.getvalue(),
               self.kernel.getvalue(),
               "def compiled_fn(inputs):",
               self.wrapper.indent().getvalue(),
           ])

       def define_kernel(self, name, kernel_src):
           """注册一个 kernel 到 wrapper"""
           self.kernel.writeline(kernel_src)

       def call_kernel(self, name, grid, args):
           """生成 kernel launch 代码"""
           self.wrapper.writeline(f"{name}[{grid}]({', '.join(args)})")

Buffer 管理与复用
=======================

Wrapper 的一个关键职责是管理 buffer 的分配和复用。Inductor 在 scheduler 阶段就确定了每个 buffer 的 lifetime，wrapper 根据这个信息决定：

- **分配时机 ** ：在第一个需要它的 kernel 之前分配
- **复用策略 ** ：如果两个 buffer 的 lifetime 不重叠，可以复用同一块显存
- **释放时机** ：在最后一个使用它的 kernel 之后释放

.. code-block:: python

   # Wrapper 生成的 buffer 复用代码
   def compiled_fn(x, y):
       buf0 = torch.empty(...)       # 分配 buf0
       kernel_1(x, y, buf0)
       # buf0 的 lifetime 结束
       
       # buf1 复用 buf0 的存储（因为 buf0 已不再需要）
       buf1 = buf0                   # 存储复用
       kernel_2(buf1, ...)
       
       output = torch.empty(...)     # 分配输出
       kernel_3(buf1, output)
       return output

这个优化对显存密集型模型（如大语言模型）非常关键——它可以显著降低峰值显存占用。Buffer 复用的决策在 ``Scheduler`` 的 ``_init`` 方法中根据依赖图计算。

Stream 管理
=================

GPU 上的 kernel launch 默认在默认 stream（stream 0）上执行，保证按 launch 顺序执行。Inductor 也支持多 stream 并发执行——但默认不开启，因为多 stream 调度需要更复杂的同步管理。

当启用多 stream 时，wrapper 会在不同 stream 上 launch 独立的 kernel：

.. code-block:: python

   # 多 stream launch
   stream_1 = torch.cuda.Stream()
   stream_2 = torch.cuda.Stream()

   with torch.cuda.stream(stream_1):
       kernel_1[grid1](x, y, buf0)

   with torch.cuda.stream(stream_2):
       kernel_2[grid2](buf0, buf1)

   # 同步
   torch.cuda.synchronize()

多 stream 主要用于流水线并行和数据加载并行。对于单个模型的前向/反向传播，单 stream 通常更优（避免同步开销）。

C++ Wrapper
================

``CppWrapperCpu`` 和 ``CppWrapperGpu`` 生成 C++ 而非 Python 的 wrapper 代码。这对于减少 Python 解释器开销、降低延迟有帮助：

.. code-block:: cpp

   // CppWrapperCpu 生成的代码
   void compiled_fn(float* x, float* y, float* output) {
       #pragma omp parallel for
       for (int i = 0; i < N; i++) {
           output[i] = cpp_kernel_1(x[i], y[i]);
       }
       
       #pragma omp parallel for
       for (int i = 0; i < N; i++) {
           output[i] = cpp_kernel_2(output[i]);
       }
   }

C++ wrapper 被编译为 ``.so`` ，通过 ``ctypes`` 或 ``torch.utils.cpp_extension`` 加载。

小结
======

这一节介绍了 kernel launch 的机制：

- **Wrapper 角色 ** ：生成 launch 代码，管理 buffer 生命周期
- **GPU launch** ：Python wrapper 调用 Triton kernel，支持异步执行
- **Buffer 复用 ** ：通过 lifetime 分析复用存储，降低峰值显存
- **Stream 管理 ** ：可选的多 stream 并发执行
- **C++ Wrapper** ：替代 Python wrapper 以减少解释器开销

至此，第 6 章内容全部完成。下一章我们将深入 Triton 编程——了解 Triton 语言本身、自定义 kernel 的编写方法、以及 Triton 与 CUDA 的对比。
