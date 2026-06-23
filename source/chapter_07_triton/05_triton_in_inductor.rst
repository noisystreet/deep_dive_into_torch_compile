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

Triton autotune 在 Inductor 中的实现
==========================================

第 5.8 节提到了 Inductor 的 autotune 机制。在 Triton 后端，autotune 通过枚举 ``constexpr`` 参数实现：

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
       
       if time < best_time:
           best_config = config
           best_time = time

这个 autotune 过程在 ``autotune_process.py`` 的子进程中异步执行，不阻塞主进程。

对于 ``max-autotune`` 模式，Inductor 会枚举更多配置组合，包括不同的 ``num_stages`` 和 ``num_warps``。对于 ``default`` 模式，使用启发式规则选择配置，跳过基准测试。

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
