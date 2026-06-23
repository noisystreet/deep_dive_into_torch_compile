.. _codegen-overview:

==================
代码生成概览
==================

代码生成（Codegen）是 Inductor 编译流水线的最后一步。当 Scheduler 完成 IRNode 的融合和调度后，它将每个 ``FusedSchedulerNode`` 分发给对应设备的代码生成器，后者将 IRNode 翻译为可编译的源代码。

这一节从整体上了解代码生成器的架构和它在整个流程中的位置。

代码生成器架构
====================

Inductor 的代码生成器位于 ``pytorch/torch/_inductor/codegen/`` 目录，采用**双层架构**：

1. **Scheduling 层**：负责将 Scheduler 输出的节点翻译为源代码字符串。不同类型的设备有不同的 Scheduling 实现。
2. **Wrapper 层**：负责生成调用 kernel 的 Python 包装器代码，包括 kernel launch 参数、grid 计算、stream 管理等。

.. code-block:: text

   codegen/
   ├── common.py              # 基类 BaseScheduling、设备注册
   ├── simd.py                # SIMDScheduling：GPU/CPU 共享的 SIMD 调度逻辑
   ├── triton.py              # TritonScheduling + TritonKernel：GPU 代码生成
   ├── cpp.py                 # CPPScheduling + CPPKernel：CPU 代码生成
   ├── wrapper.py             # PythonWrapperCodegen：kernel launch 包装器
   ├── cpp_wrapper_cpu.py     # CPU C++ wrapper
   ├── cpp_wrapper_gpu.py     # GPU C++ wrapper
   ├── cuda_combined_scheduling.py  # CUDA 组合调度
   ├── halide.py              # Halide 后端（实验性）
   └── mps.py                 # Apple Metal 后端

设备注册与后端选择
========================

Scheduler 在调用 ``codegen()`` 时，通过 ``get_scheduling_for_device()``（定义在 ``common.py`` 中）根据设备类型选择对应的 Scheduling 实现：

.. code-block:: python
   :caption: pytorch/torch/_inductor/codegen/common.py

   device_codegens: dict[str, DeviceCodegen] = {}

   def get_scheduling_for_device(device: str) -> SchedulingConstructor | None:
       return device_codegens[device].scheduling if device in device_codegens else None

   def init_backend_registration():
       """注册所有设备对应的后端"""
       from .cpp import CppScheduling
       from .cuda_combined_scheduling import CUDACombinedScheduling
       from .triton import TritonScheduling
       from .mps import MetalScheduling
       ...

设备注册在 ``init_backend_registration()`` 中完成。当 Inductor 导入时，此函数被调用，将调度器和包装器注册到 ``device_codegens`` 字典中。此后，Scheduler 遍历节点时，每个节点的设备类型决定了使用哪个后端。

从 Scheduler 到代码生成
=============================

Scheduler 的 ``codegen()`` 方法触发代码生成过程：

.. code-block:: text

   scheduler.codegen()
       │
       for node in self.nodes:      # 遍历所有 SchedulerNode
           │
           ├─ 根据 node 的设备获取对应后端
           │      backend = get_scheduling_for_device(node.device.type)
           │
           └─ 调用后端的 codegen 方法
                  backend.codegen(node)
                      │
                      ├─ GPU 设备:
                      │    TritonScheduling.codegen(node)
                      │    → 生成 Triton 代码字符串
                      │    → 调用 wrapper 注册 kernel launch
                      │
                      └─ CPU 设备:
                           CPPScheduling.codegen(node)
                           → 生成 C++/OpenMP 代码字符串
                           → 调用 wrapper 注册 kernel launch

对于融合后的 ``FusedSchedulerNode``，codegen 需要将多个 IRNode 的 ``inner_fn`` 组合到同一个 kernel 的循环体中。这是代码生成器最复杂的工作——它必须正确地合并多个循环体、处理不同 IRNode 之间的数据依赖、并消除中间结果的显存访问。

Wrapper 代码生成
====================

``wrapper.py`` 中的 ``PythonWrapperCodegen`` 负责生成 Python 级别的 wrapper 代码。它生成的代码结构如下：

.. code-block:: python

   # Wrapper 生成的 Python 代码示例
   def compiled_function(x, y):
       # 分配输出 buffer
       buf0 = torch.empty(...)
       
       # 调用 Triton kernel 1
       triton_kernel_1[(grid_x, grid_y)](
           x, y, buf0,
           BLOCK_SIZE=1024,
       )
       
       # 调用 Triton kernel 2
       triton_kernel_2[(grid_x,)](
           buf0,
           BLOCK_SIZE=512,
       )
       
       return buf0

Wrapper 代码在 CPU 上运行，它负责编排 GPU kernel 的 launch。对于 CPU 后端，Wrapper 直接调用 C++ 编译好的函数。

生成的代码被 ``codecache.py`` 编译为 ``.so`` 或 ``.py`` 文件，然后加载为 Python callable。

ops 原语的映射
====================

代码生成的关键一步是将 IR 层面的 ``ops.*`` 原语映射为具体的硬件指令。不同的后端有不同的映射方式：

.. list-table::
   :header-rows: 1

   * - ops 原语
     - Triton 后端
     - C++ 后端
   * - ``ops.load(name, index)``
     - ``tl.load(ptr + offsets, mask=mask)``
     - ``x[index]``
   * - ``ops.store(name, index, value)``
     - ``tl.store(ptr + offsets, value, mask=mask)``
     - ``x[index] = value``
   * - ``ops.sin(value)``
     - ``tl.sin(value)``
     - ``std::sin(value)``
   * - ``ops.add(a, b)``
     - ``a + b``
     - ``a + b``
   * - ``ops.reduction("sum", name, index)``
     - ``tl.sum(value, axis)``
     - 循环累加

这种映射通过 ``ops_handler.py`` 中的 ``OpsHandler`` 接口实现。每个后端实现自己的 ``OpsHandler``，在 codegen 时替换全局的 ``V.ops``：

.. code-block:: python

   # Triton 后端设置自己的 ops handler
   with V.set_ops_handler(TritonOverrides()):
       # 此范围内的 inner_fn 调用使用 Triton 语义
       value = inner_fn(index)

小结
======

这一节从整体上了解了代码生成器的架构：

- **双层架构**：Scheduling 层生成 kernel 代码，Wrapper 层生成 launch 代码
- **设备注册**：通过 ``device_codegens`` 字典选择对应后端的 Scheduling 实现
- **ops 映射**：IR 层面的 ``ops.*`` 原语被映射为具体硬件指令
- **Wrapper 代码**：Python 代码，负责编排 kernel launch
