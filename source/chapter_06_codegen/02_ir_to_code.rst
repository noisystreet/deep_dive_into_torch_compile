.. _ir-to-code:

=========================
IR 到代码的变换机制
=========================

第 5 章介绍过，Scheduler 完成融合后将 ``FusedSchedulerNode`` 分发给对应后端的 ``codegen()`` 方法。这一节深入这个分发过程，看一个融合的 IRNode 组如何被翻译为 Triton 或 C++ 的源码字符串。

入口：从 Scheduler 到 Scheduling
=======================================

Scheduler 遍历所有节点，调用每个节点的 ``codegen()``：

.. code-block:: python
   :caption: pytorch/torch/_inductor/scheduler.py（简化示意）

   class Scheduler:
       def codegen(self):
           for node in self.nodes:   # 遍历 SchedulerNode / FusedSchedulerNode
               backend = get_scheduling_for_device(node.device.type)
               backend.codegen(node)

对于 GPU 节点，``TritonScheduling.codegen()`` 被调用。``TritonScheduling``（继承自 ``SIMDScheduling``）接收到节点后，执行以下步骤：

.. code-block:: text

   TritonScheduling.codegen(node)
       │
       ├─ 1. 确定 tiling 参数
       │      get_tiling_and_scores()
       │      基于 numel / reduction_numel 计算
       │      BLOCK_SIZE、num_warps 等
       │
       ├─ 2. 创建 kernel 实例
       │      kernel = TritonKernel(...)
       │
       ├─ 3. codegen_node_schedule_with_kernel()
       │      两遍扫描，生成循环体
       │
       ├─ 4. codegen_kernel()
       │      生成完整的 Triton 源码
       │      （包括 @triton.jit 装饰器、参数签名等）
       │
       └─ 5. call_kernel()
              注册 kernel launch 到 wrapper

两遍扫描（Two-Pass Codegen）
================================

``codegen_node_schedule_with_kernel``（在 ``simd.py`` 中）是代码生成的核心——它使用**两遍扫描**策略处理融合后的节点列表：

.. code-block:: python
   :caption: pytorch/torch/_inductor/codegen/simd.py（简化示意）

   def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
       with kernel:
           # 第一遍：收集索引信息
           for node in node_schedule:
               if node is DisableReduction:
                   kernel.disable_reduction()
               elif node is EnableReduction:
                   kernel.enable_reduction()
               else:
                   node.decide_inplace_update()
                   index_vars = kernel.split_and_set_ranges(node.get_ranges())
                   # 记录所有索引表达式
                   ...

           # 第二遍：生成代码
           for node in node_schedule:
               if node is DisableReduction:
                   kernel.disable_reduction()
               elif node is EnableReduction:
                   kernel.enable_reduction()
               else:
                   index_vars = kernel.split_and_set_ranges(node.get_ranges())
                   node.codegen(index_vars)   # ← 核心调用

两遍扫描的必要性：第一遍收集索引信息是为了确定 loop 的边界和 tiling 参数，第二遍才实际生成加载/计算/存储代码。其中 ``split_and_set_ranges`` 将 IRNode 的语义范围（如 ``[M, N]``）映射为 Triton 的并行索引（如 ``pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)``）。

split_and_set_ranges：循环切分
==========================================

``split_and_set_ranges`` 方法（在 ``simd.py`` 的 ``SIMDKernel`` 中）是索引变换的核心。它接收 IRNode 的 ``ranges``（如 ``[1024, 768]``），根据 tiling 策略将其切分为多个级别的循环：

.. code-block:: text

   IRNode.ranges = [1024, 768]

   tiling 策略:
       x_block = 1024  → TRITON_BLOCK_SIZE = 1024
       y_block = 768   → TRITON_BLOCK_SIZE = 768

   split_and_set_ranges 输出:
       xpid = tl.program_id(0)    ← 跨 program 并行
       ypid = tl.program_id(1)
       x = xpid * 1024 + tl.arange(0, 1024)
       y = ypid * 768  + tl.arange(0, 768)

这些索引变量在后续的 ``node.codegen(index_vars)`` 中被传递给 IRNode 的 ``inner_fn``，从而实现从语义索引到硬件索引的映射。

node.codegen：触发 inner_fn
=========================================

每个 ``SchedulerNode`` 的 ``codegen(index_vars)`` 方法调用其包含的 IRNode 的 codegen。对于 ``Pointwise`` 节点，核心逻辑在 ``Loops.codegen`` 中：

.. code-block:: text

   node.codegen(index_vars)
       │
       ├─ 1. kernel.loads.write()
       │      调用 ops.load 加载输入
       │      生成: x = tl.load(x_ptr + offsets, mask=mask)
       │
       ├─ 2. kernel.compute.write()
       │      调用 IRNode.inner_fn(index_vars)
       │      执行: ops.sin(ops.load(...))
       │      生成: sin_x = tl.sin(x)
       │
       └─ 3. kernel.stores.write()
               调用 ops.store 存储结果
               生成: tl.store(output_ptr + offsets, result, mask=mask)

这个过程中的关键机制是 **CSE（公共子表达式消除）**。``SIMDKernel`` 维护了一个 CSE 缓存，当多个 IRNode 使用相同的索引表达式或中间值时，自动复用已有的计算结果：

.. code-block:: text

   两个 Pointwise 节点融合后的 CSE 效果:
       # 节点 A: output = sin(x) + cos(x)
       # 节点 B: output2 = sin(x) * cos(x)

       生成的代码:
       x_val = tl.load(x_ptr + offsets)   ← 一次加载
       sin_x = tl.sin(x_val)              ← 一次 sin
       cos_x = tl.cos(x_val)              ← 一次 cos
       tl.store(out1_ptr, sin_x + cos_x)  ← 复用 sin_x, cos_x
       tl.store(out2_ptr, sin_x * cos_x)  ← 复用 sin_x, cos_x

codegen_kernel：组装完整源码
===========================================

当循环体生成完毕后，``codegen_kernel()`` 方法（在 ``triton.py`` 的 ``TritonKernel`` 中）将之前分散生成的代码片段组装为完整的 Triton kernel：

.. code-block:: text

   TritonKernel.codegen_kernel()
       │
       ├─ 组装 imports
       │      import triton
       │      import triton.language as tl
       │
       ├─ 组装 kernel 签名
       │      @triton.jit
       │      def kernel_name(
       │          x_ptr, y_ptr, output_ptr,
       │          n_elements,
       │          BLOCK_SIZE: tl.constexpr,
       │      ):
       │
       ├─ 组装循环体
       │      pid = tl.program_id(axis=0)
       │      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       │      mask = offsets < n_elements
       │      x = tl.load(x_ptr + offsets, mask=mask)
       │      ...
       │      tl.store(output_ptr + offsets, result, mask=mask)
       │
       └─ 组装 benchmark 代码
              if config.benchmark_kernel:
                  插入基准测试代码

生成的完整源码作为字符串返回，随后被 ``TritonCodeCache`` 或 ``AsyncCompile`` 提交给 Triton 编译器编译。

CPU 后端的对应机制
========================

CPU 端（CPPScheduling）的流程本质相同，但输出是 C++/OpenMP 代码：

.. code-block:: text

   CPPScheduling.codegen(node)
       │
       ├─ 确定 tiling
       │      基于缓存行（64 bytes）对齐
       │
       ├─ 生成循环
       │      #pragma omp parallel for
       │      for (int i = 0; i < N; i++) {
       │
       ├─ 生成循环体（C++ 代码）
       │      float x = input[i];
       │      float sin_x = std::sin(x);
       │      float cos_x = std::cos(x);
       │      output[i] = sin_x + cos_x;
       │
       └─ 注册到 wrapper
              CppWrapperCpu 接收代码字符串

CSE 在 CPU 后端同样有效。CPPKernel 维护自己的 CSE 缓存，避免在循环体中生成冗余的加载和计算。

小结
======

这一节介绍了 IRNode 到代码的变换机制：

- **入口**：Scheduler 分发节点给 ``TritonScheduling.codegen()`` 或 ``CPPScheduling.codegen()``
- **两遍扫描**：第一遍收集索引信息，第二遍生成加载/计算/存储
- **split_and_set_ranges**：将 IR 语义索引映射为硬件循环索引
- **CSE**：自动消除公共子表达式，避免冗余加载和计算
- **codegen_kernel**：将分散的代码片段组装为完整的 kernel 源码
