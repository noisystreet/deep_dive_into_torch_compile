.. _triton-in-inductor:

=======================
Inductor 中的 Triton
=======================

Inductor 的 GPU 代码生成后端以 Triton 为核心。这一节从 Inductor 的角度看 Triton——它如何生成 Triton kernel、如何管理编译和缓存，以及如何与 Triton 编译器交互。

TritonScheduling 与 TritonKernel
=======================================

Triton 相关的代码生成在 ``pytorch/torch/_inductor/codegen/triton.py`` 中实现。核心类是 ``TritonScheduling`` 和 ``TritonKernel``：

.. code-block:: text

   TritonScheduling（继承 SIMDScheduling）
       │
       ├─ 负责：融合决策（can_fuse）、tiling 选择
       ├─ 入口：codegen(node_schedule)
       │       → 创建 TritonKernel
       │       → 两遍扫描生成循环体
       │       → 调用 codegen_kernel() 生成源码
       └─ 调用：define_kernel() 注册到 wrapper

   TritonKernel（继承 SIMDKernel）
       │
       ├─ 负责：生成 Triton 源码片段
       ├─ loads:  生成 tl.load 调用
       ├─ compute: 生成 tl.sin/tl.add 等计算
       ├─ stores: 生成 tl.store 调用
       └─ codegen_kernel(): 组装为完整 @triton.jit 函数

第 6.2 节已经介绍了从 ``FusedSchedulerNode`` 到 Triton 源码的变换过程。这里补充几个关键机制。

SIMDKernel 与 TritonKernel 的关系
=========================================

理解 ``SIMDKernel`` 和 ``TritonKernel`` 的关系，有助于看清 Inductor 的代码生成架构。

继承层次
------------

.. code-block:: text

   Kernel（基类）
       │
       ├─ SIMDKernel（CPU/GPU 通用的 SIMD 风格 kernel）
       │   │
       │   ├─ TritonKernel（GPU 后端，生成 Triton 源码）
       │   │
       │   ├─ CPPKernel（CPU 后端，生成 C++ 代码）
       │   │
       │   └─ CppTemplateKernel（CPU 端模板 kernel）
       │
       └─ ExternKernel（调用外部库，如 cuBLAS）

``SIMDKernel`` 提供了与设备无关的通用 kernel 生成能力：

.. code-block:: text

   SIMDKernel 的核心方法
       │
       ├─ load()         → 生成加载数据的代码
       │                    （Triton 中为 tl.load，CPP 中为数组读取）
       ├─ store()        → 生成存储数据的代码
       ├─ compute()      → 在节点之间添加计算代码
       ├─ codegen_body() → 生成 kernel 的循环体
       └─ codegen_kernel() → 组装完整的 kernel 函数

``TritonKernel`` 重写了这些方法，将通用的 load/store/compute 操作翻译为 Triton 特有的 API：

.. list-table::
   :header-rows: 1

   * - SIMDKernel 方法
     - TritonKernel 实现
     - 说明
   * - ``load()``
     - ``tl.load(ptr + offsets, mask=mask, other=0.0)``
     - 自动生成 mask 和边界检查
   * - ``store()``
     - ``tl.store(ptr + offsets, value, mask=mask)``
     - 自动生成 mask
   * - ``compute()``
     - ``tl.add(x, y)``, ``tl.sin(x)`` 等
     - 将 ATen 操作映射到 tl.* 函数
   * - ``codegen_kernel()``
     - 组装为 ``@triton.jit def kernel(...)`` 函数
     - 添加装饰器、网格参数、constexpr 推导

这种设计使得 Inductor 可以共享大部分的调度和融合逻辑（在 ``SIMDScheduling`` 中），而代码生成细节由具体的子类实现。

Triton Kernel 的编译
=========================

Inductor 生成的 Triton 源码以 ``@triton.jit`` 装饰的函数呈现。编译发生在 ``PyCodeCache`` 中：

.. code-block:: text

   生成的 Triton kernel 源码字符串
       │
       ▼
   TritonScheduling.define_kernel()
       │
       ├─ 计算 src_code 的 content hash
       │
       ├─ 写入 .py 文件到磁盘缓存目录
       │      /tmp/torchinductor/xxx/kernel_name.py
       │
       ├─ 通过 PyCodeCache.load() 加载
       │      → Python 执行 .py 文件
       │      → @triton.jit 触发 Triton 编译器
       │      → Triton 编译器生成 PTX → cuBin
       │      → 返回可调用的 kernel object
       │
       └─ 返回 kernel_name 给 wrapper

Triton 的编译是**分层**的：

.. code-block:: text

   @triton.jit 函数 (Python AST)
       │
       ▼
   Triton 编译器前端
       │  Python AST → Triton IR → 类型推断 → 循环展开
       │
       ▼
   Triton 编译器后端
       │  Triton IR → PTX (NVIDIA 中间表示)
       │
       ▼
   ptxas（NVIDIA 工具链）
       │  PTX → SASS (GPU 机器码)
       │
       ▼
   cubin（编译后的 kernel 二进制）

第一次调用一个 kernel 时会触发编译，后续调用直接使用缓存的编译结果。

缓存机制
============

Inductor 对 Triton kernel 的缓存分为两层：

**源码级别**：``wrapper.src_to_kernel`` 字典确保同一份源码只被提交一次：

.. code-block:: python

   # triton.py
   def define_kernel(self, src_code, node_schedule, kernel):
       if src_code in wrapper.src_to_kernel:
           kernel_name = wrapper.src_to_kernel[src_code]
           return kernel_name  # 跳过重复编译
       ...

**编译结果级别**：Triton 编译器自带磁盘缓存（位于 ``~/.triton/cache/``）。相同的 kernel 源码、GPU 架构、Triton 版本组合只编译一次。

这两个缓存协同工作：

.. code-block:: text

   进程内调用 define_kernel()
       │
       ├─ src_code 在 wrapper.src_to_kernel 中?
       │   └─ 是 → 复用 kernel_name
       │
       ├─ Triton 编译
       │   └─ 生成的 cubin 在 ~/.triton/cache/ 中?
       │       ├─ 是 → 跳过编译，直接加载
       │       └─ 否 → 编译 → 写缓存 → 加载
       │
       └─ 返回可调用的 kernel

Triton kernel 持久化缓存详解
-----------------------------------

Triton 的磁盘缓存使用 content-addressed 策略。cache key 的生成逻辑如下：

.. code-block:: python

   # Triton 编译器内部（简化）
   def _get_cache_key(src_code, gpu_type, triton_version):
       # 对 kernel 源码做 hash
       src_hash = hashlib.sha256(src_code.encode()).hexdigest()
       # 加上 GPU 架构和编译器版本
       key = f"{src_hash}_{gpu_type}_{triton_version}"
       return key

   # 缓存目录结构
   # ~/.triton/cache/
   #   ├── a1b2c3d4..._sm80_3.0.0/   ← cache key
   #   │   ├── kernel.cubin           ← 编译后的二进制
   #   │   ├── kernel.ptx             ← 中间 PTX 文件（调试用）
   #   │   └── kernel.json            ← 元数据（参数信息等）
   #   ├── e5f6g7h8..._sm80_3.0.0/
   #   └── ...

缓存失效（cache miss）的条件包括：

1. **kernel 源码变化**：任何对 ``@triton.jit`` 函数的修改
2. **GPU 架构变化**：从 A100（sm_80）换到 H100（sm_90）
3. **Triton 版本变化**：升级或降级 ``triton`` 包
4. **PTX 版本变化**：Triton 编译器升级了 PTX 后端版本

.. tip::

   **缓存预热（Cache Warming）。**
   在模型部署的 CI 流水线中，可以在构建阶段通过 ``triton.compile`` 预编译所有 kernel，并将 ``~/.triton/cache/`` 目录打包到容器镜像中。这样在运行阶段，所有 kernel 的编译已提前完成，首次启动时间从分钟级降低到秒级。

Triton autotune 在 Inductor 中的实现
==========================================

第 5.10 节提到了 Inductor 的 autotune 机制。在 Triton 后端，autotune 通过枚举 ``constexpr`` 参数实现：

.. code-block:: python

   # Inductor 的 autotune 过程（简化）
   configs = [
       {"BLOCK_SIZE": 1024, "num_warps": 4},
       {"BLOCK_SIZE": 2048, "num_warps": 4},
       {"BLOCK_SIZE": 1024, "num_warps": 8},
       {"BLOCK_SIZE": 2048, "num_warps": 8},
   ]

   best_config = None
   best_time = float("inf")
   
   for config in configs:
       # 为每组 config 生成并编译 kernel
       src_code = generate_kernel_with_config(config)
       kernel = compile_triton_kernel(src_code)
       
       # 基准测试
       time = benchmark_kernel(kernel, example_inputs)
       
       if time < best_config:
           best_config = config
           best_time = time

这个 autotune 过程在 ``autotune_process.py`` 的子进程中异步执行，不阻塞主进程。

对于 ``max-autotune`` 模式，Inductor 会枚举更多配置组合，包括不同的 ``num_stages`` 和 ``num_warps``。对于 ``default`` 模式，使用启发式规则选择配置，跳过基准测试。

Autotune 策略详解
-----------------------

Inductor 的 Triton autotune 包含多种搜索策略，根据模式不同使用不同的策略：

坐标下降（Coordinate Descent）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

坐标下降是 ``max-autotune`` 模式下的核心搜索策略。它通过每次调整一个参数维度的方式，逐步逼近最优配置：

.. code-block:: text

   搜索过程示例：
   Step 1: 固定 num_warps=4, num_stages=2
           枚举 BLOCK_SIZE ∈ {64, 128, 256, 512}
           找到最优 BLOCK_SIZE=128
   
   Step 2: 固定 BLOCK_SIZE=128, num_stages=2
           枚举 num_warps ∈ {2, 4, 8, 16}
           找到最优 num_warps=8
   
   Step 3: 固定 BLOCK_SIZE=128, num_warps=8
           枚举 num_stages ∈ {1, 2, 3, 4}
           找到最优 num_stages=2

这种策略的优势在于搜索空间从 O(N^3) 降低到 O(3N)。对于 4 个候选值，全枚举需要 64 种组合，坐标下降只需 12 种组合。

配置枚举（Config Enumeration）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于简单 kernel（如逐元素操作），Inductor 使用预定义的配置列表，不进行坐标下降搜索：

.. code-block:: python

   # Inductor 中预定义的 Triton config 列表（简化）
   _triton_configs = {
       "persistent_reduction": [
           triton_configs.persistent_reduction(size_hints=[1024]),
           triton_configs.persistent_reduction(size_hints=[2048]),
       ],
       "pointwise": [
           triton_configs.pointwise(size_hints=[1024]),
           triton_configs.pointwise(size_hints=[2048]),
       ],
       "reduction": [
           triton_configs.reduction(size_hints=[1024]),
           triton_configs.reduction(size_hints=[2048]),
       ],
   }

这些预定义配置基于 kernel 的类型（pointwise、reduction、persistent_reduction）和大小提示（size hints）来选择。

启发式选择（Heuristic Selection）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 ``default`` 模式下，Inductor 完全跳过基准测试，直接使用启发式规则选择配置：

.. code-block:: python

   # Inductor 的启发式配置选择（简化）
   def heuristic_config(size_hints, kernel_type):
       # 根据数据大小和 kernel 类型推断合适的配置
       if kernel_type == "pointwise":
           if size_hints[0] > 1_000_000:
               return {"BLOCK_SIZE": 2048, "num_warps": 4}
           else:
               return {"BLOCK_SIZE": 1024, "num_warps": 4}
       elif kernel_type == "reduction":
           return {"BLOCK_SIZE": 512, "num_warps": 4}
       ...

启发式选择的优势在于**零编译开销**——不需要为多组配置生成和编译 kernel。对于训练场景，单次准确率 80-90% 的启发式选择通常已经足够。

Autotune 的异步执行
------------------------

Inductor 的 autotune 在 ``AutoTuneProcess`` 中异步执行。主进程继续执行代码生成和编译，autotune 子进程在后台进行基准测试：

.. mermaid::

   sequenceDiagram
       participant Main as 主进程
       participant AT as AutoTuneProcess
       participant Triton as Triton Compiler

       Main->>AT: 提交 autotune 请求（kernel 模板 + 输入信息）
       Note over Main: 主进程继续执行<br/>其他 kernel 的代码生成
       
       AT->>Triton: 生成 config 组合列表
       loop 每个 config
           AT->>Triton: 编译 kernel（带特定 config）
           Triton-->>AT: cubin
           AT->>Triton: 执行基准测试
           Triton-->>AT: 延迟数据
       end
       
       AT-->>Main: 返回最优 config
       Note over Main: 主进程使用最优 config<br/>更新 kernel 代码

这种异步设计避免了 autotune 阻塞整个编译流程。但是，最终选择的结果仍然需要在主进程中触发一次 re-compile（如果最优 config 与初始 config 不同）。

.. note::

   **autotune 的冷启动问题。**
   第一个 batch 通常使用默认配置执行（因为 autotune 子进程可能还没有返回结果）。从第二个 batch 开始，才会使用 autotune 确定的最优配置。对于推理场景，这意味着前几个 token 的延迟会偏高。

Inductor 中动态形状的处理
=================================

动态形状（dynamic shapes）是 Inductor 处理 Triton 代码生成时的核心挑战。不同形状的张量可能导致不同的分块策略和 kernel 代码。

动态形状的挑战
--------------------

考虑一个简单的逐元素 kernel，其 BLOCK_SIZE 的选择依赖于张量大小：

.. code-block:: python

   @triton.jit
   def dynamic_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)  # BLOCK_SIZE 在编译时固定
       mask = offsets < n_elements          # n_elements 在运行时确定
       x = tl.load(x_ptr + offsets, mask=mask)
       tl.store(output_ptr + offsets, x, mask=mask)

如果 ``n_elements`` 在不同调用间变化，固定 ``BLOCK_SIZE`` 可能导致两种低效：

- BLOCK_SIZE 太大（> n_elements）：大量线程闲置（mask 为 false），浪费计算资源
- BLOCK_SIZE 太小（<< n_elements）：网格过大，kernel launch 开销增加

Inductor 的策略
--------------------

Inductor 通过以下策略处理动态形状：

**策略 1：sympy 表达式计算**。Inductor 使用 ``sympy`` 符号表达式来表示动态形状。在代码生成阶段，它通过 ``sympy`` 表达式推导出 kernel 参数的值。

.. code-block:: python

   # Inductor 内部 - 使用 sympy 表达式
   from sympy import Symbol
   
   n_elements = Symbol("n_elements")  # 动态形状的符号表示
   BLOCK_SIZE = 1024  # 固定 block size
   grid_size = n_elements / BLOCK_SIZE  # 网格大小也是符号表达式

在 kernel launch 时，sympy 表达式被求值为具体的数值。

**策略 2：统一 kernel 代码生成**。Inductor 生成的 Triton kernel 使用相同的源码结构，无论动态形状如何变化。关键的约束条件是 ``tl.constexpr`` 参数在首次编译后固定：

.. code-block:: text

   动态形状下的 kernel 生成流程：
   
   第一次调用 (n_elements=1024):
       → 生成 kernel 源码（BLOCK_SIZE 由启发式规则决定）
       → 编译并缓存
       → 执行（mask 处理多余的线程）
   
   第二次调用 (n_elements=2048):
       → 查看缓存，kernel 源码相同
       → 使用同一个编译结果
       → 只需更新 grid 大小：grid = (2,)  # 2048/1024 = 2
       → 执行（新增的 block 处理额外元素）

这种设计的精妙之处在于：**kernel 的二进制代码不需要改变，只需要调整 launch 参数（grid size 和运行时参数）**。这避免了动态形状下的频繁重新编译。

**策略 3：Size Assertions 和 Guarding**。当动态形状的变化超出了 kernel 的适应范围时，Inductor 会插入 size assertion 来确保运行时形状匹配：

.. code-block:: python

   # Inductor 生成的代码中的 size guard（简化）
   def forward(self, x):
       assert x.size(0) <= MAX_SIZE, f"Input size {x.size(0)} exceeds max {MAX_SIZE}"
       # 使用缓存的 kernel 执行
       kernel[(grid,), ...](x, ...)

如果 guard 失败（形状超出预期范围），Inductor 会触发重新编译，生成适应新形状范围的 kernel。

sympy 表达式在 Triton codegen 中的应用
------------------------------------------------

Inductor 使用 ``sympy`` 来处理 kernel 参数中的动态维度。以下是几个典型应用：

**步幅计算**。对于非连续张量，Inductor 使用 sympy 表达式表示步幅：

.. code-block:: python

   # Inductor 生成的代码中的 sympy stride 表达式
   stride = sympy.Symbol("stride_0")  # 运行时确定
   ptr_offset = row_idx * stride + col_idx  # 符号化地址计算

在运行时，这些符号表达式被替换为具体的整数值。

**边界检查**。动态形状下的边界检查使用运行时变量：

.. code-block:: text

   kernel 中的 mask 生成:
       offsets = tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements  # n_elements 是运行时参数
       tl.load(ptr + offsets, mask=mask)

这里 ``n_elements`` 是 kernel 的运行时参数（不是 ``tl.constexpr``），在每次 launch 时传入不同的值。

.. tip::

   **动态形状的性能权衡。**
   动态形状的代价是：kernel 代码是为最坏情况（最大可能的 BLOCK_SIZE）生成的，当实际数据小于这个最大值时，部分线程处于空闲状态。如果形状的波动范围很大（如 1 到 65536），建议使用多个 kernel 变体，各自针对不同的形状范围进行优化。Inductor 的 ``torch.compile(dynamic=False)`` 实际上禁用了动态形状支持，让编译器为特定形状生成完全优化的代码。

与 Triton 编译器的交互
===============================

Inductor 通过 ``triton.compile`` 或 ``triton.jit`` 与 Triton 编译器交互。``@triton.jit`` 装饰的函数在首次调用时被编译：

.. code-block:: python

   # Triton 内部（简化）
   @triton.jit
   def kernel(x_ptr, ...):
       ...

   # 首次调用触发编译
   kernel[(grid,)](x, ...)
   # → Triton 编译器检查是否有缓存
   # → 无缓存 → 编译 → 缓存 cubin
   # → 执行编译后的 kernel

编译过程中，Triton 编译器会进行：

1. **类型推断**：根据 ``tl.constexpr`` 和其他参数类型推断所有中间变量的类型
2. **循环展开**：将 ``tl.arange`` 和 ``for`` 循环展开为具体指令序列
3. **内存合并分析**：分析块内访问模式，生成合并的内存加载/存储
4. **PTX 生成**：将 Triton IR 翻译为 PTX 指令，包括使用 Tensor Core 指令

这些步骤对 Inductor 完全透明——Inductor 只需要生成正确的 Triton 源码字符串，编译细节由 Triton 编译器处理。
