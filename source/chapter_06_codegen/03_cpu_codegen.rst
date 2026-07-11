.. _cpu-codegen:

==============
CPU 代码生成
==============

.. note::

   **CPU 后端的优化主要由 Intel 团队贡献。 **
   在 PyTorch 的公开 commit 历史中，CPU 后端相关提交（mps/xpu/rocm/cpu）共约 676 次，其中来自 Intel 工程师的贡献占了绝大多数。CPU 后端的演进路线是：**2022.10** 实现基础向量化（AVX2/AVX512）→**2023.03**Conv/GEMM fusion 和 CppWrapper →**2023.08**BF16 支持 →**2024** GEMM 模板和 max-autotune。截止 2024 年底，Inductor CPU 后端的向量化覆盖率已从最初的 ~70% 提升到 94%。CPU 后端比 GPU 后端更难优化——GPU 后端有 Triton 这个第三方编译器，而 CPU 后端需要自己处理向量化、缓存行对齐、OpenMP 调度等底层细节。

CPPScheduling 和 CPPKernel 负责将 IRNode 翻译为 C++/OpenMP 代码。这是 Inductor 在 CPU 上的默认代码生成后端。

CPPScheduling 架构
=======================

CPPScheduling（在 ``codegen/cpp.py`` 中）继承自 ``SIMDScheduling`` ，负责 CPU 端的调度和代码生成。它的核心职责是：

1. 接收 Scheduler 传入的 ``SchedulerNode`` 或 ``FusedSchedulerNode``
2. 为每个节点生成优化的 C++ 代码
3. 利用 OpenMP 实现多线程并行

.. code-block:: text

   CPPScheduling.codegen(node)
       │
       ├─ 1. 确定循环范围和 tiling 策略
       │      基于输入端大小和缓存行对齐
       │
       ├─ 2. 生成循环结构
       │      #pragma omp parallel for
       │      for (int i = 0; i < N; i++) {
       │
       ├─ 3. 生成循环体
       │      将 IRNode 的 inner_fn 
       │      翻译为 C++ 操作
       │
       └─ 4. 注册到 wrapper
              将生成的代码注册到 CppWrapperCpu

OpenMP 并行
================

对于逐元素操作，Inductor 自动插入 OpenMP ``parallel for`` 指令。Scheduler 在融合阶段已经确定了哪些节点可以合并，CPPScheduling 在此基础上选择最优的并行策略：

.. code-block:: cpp

   // Pointwise 操作：单层 parallel for
   #pragma omp parallel for
   for (int i = 0; i < N; i++) {
       float x = input[i];
       output[i] = std::sin(x) + std::cos(x);
   }

对于融合后的节点（如 Pointwise + Reduction），OpenMP 的使用更复杂——需要区分 outer loop（并行）和 inner loop（归约）：

.. code-block:: cpp

   // Pointwise + Reduction 融合
   #pragma omp parallel for reduction(+:sum)
   for (int i = 0; i < N; i++) {
       float val = std::sin(input[i]);
       sum += val;
       output[i] = val;
   }

CPPScheduling 会根据融合节点中 Reduction 的维度自动确定 OpenMP 的并行策略，选择是在最外层还是内层添加 ``#pragma omp parallel`` 。

向量化（Vectorization）
===========================

CPU 上性能的关键是向量化。Inductor 的 C++ 代码生成器会尽可能生成使用 SIMD 指令的代码。 ``CPPKernel`` 在生成循环体时，会根据数据类型和目标 CPU 特性自动选择向量化宽度：

.. code-block:: cpp

   // 自动向量化：编译器自动识别连续访问并生成 SIMD 指令
   for (int i = 0; i < N; i++) {
       output[i] = input[i] * 2.0f;
   }

   // 显式向量化（当 autovec 无法生效时）
   #pragma omp simd
   for (int i = 0; i < N; i++) {
       output[i] = complex_function(input[i]);
   }

是否启用向量化取决于 ``config.cpp.enable_autovec`` 配置项。默认情况下，GCC/Clang 的自动向量化能力足够好，Inductor 不强制插入 ``#pragma omp simd`` 。

内存布局与缓存优化
========================

CPU 代码生成还关注缓存友好性。CPPScheduling 在以下方面做优化：

**循环重排** ：对于多维循环，将最内层循环对应到连续内存维度，最大化缓存行利用率。

**Tiling** ：对于处理大张量的融合 kernel，将循环分块（tiling）以利用 L1/L2 缓存。

.. code-block:: cpp

   // 循环重排前：内层循环跨步访问
   for (int i = 0; i < M; i++) {
       for (int j = 0; j < N; j++) {
           output[i * N + j] = input[j * M + i];  // 内层 j 跳步
       }
   }

   // 循环重排后：内层循环连续访问
   for (int j = 0; j < N; j++) {
       for (int i = 0; i < M; i++) {
           output[i * N + j] = input[j * M + i];  // 内层 i 连续
       }
   }

C++ 代码的编译
====================

生成的 C++ 代码通过 ``codecache.py`` 中的 ``CppCodeCache`` 编译为 ``.so`` 共享库：

.. code-block:: text

   生成的 .cpp 文件
       │
       ▼
   CppCodeCache
       │
       ├─ 选择编译器（GCC/Clang/MSVC）
       │
       ├─ 设置编译标志
       │      -O2 -fopenmp -march=native -ffast-math
       │
       ├─ 编译为 .so
       │      g++ -shared -o kernel.so kernel.cpp ...
       │
       └─ 加载 .so，返回 callable

编译结果被磁盘缓存，key 基于源代码哈希和编译选项。这意味着只要源代码和配置不变，跨进程可以共享编译好的 ``.so`` 文件。

适用场景
===============

CPP 代码生成适用于：

- **CPU-only 推理** ：模型在 CPU 上运行，通过 OpenMP 加速并行计算
- **混合设备场景** ：部分操作在 CPU 上执行（如数据预处理、后处理）
- **TorchDense 替代** ：实验性的 ``torch.compile`` CPU 推理场景

对于 GPU 上的训练场景，Triton 代码生成是更高效的选择——我们下一节讨论。
