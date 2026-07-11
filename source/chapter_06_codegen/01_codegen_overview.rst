.. _codegen-overview:

==================
代码生成概览
==================

.. note::

   **Inductor 的代码生成器用 Python 来生成 Python 代码。 **
   这听起来有点"套娃"——Inductor 本身是用 Python 写的，它生成的 Triton kernel 也是 Python（只是带 ``@triton.jit`` 装饰器）。这意味着 Inductor 可以用 Python 的字符串模板、AST 操作、甚至 ``eval()`` 来动态构造代码。这种"自举"方式让 Inductor 的开发效率远高于传统编译器（如 LLVM 需要处理各种后端描述文件）。当然，代价是代码生成的性能开销——但这只在编译时发生，不影响运行时的性能。

代码生成的设计思想
==========================

Codegen 是 Inductor 流水线的最后一环：Scheduler 已经决定 **算什么、怎么融合**，Codegen 负责 **怎么写成可执行的源码**。

**为什么用 Python 写 Codegen**。Inductor 团队选择「Python 生成 Python/Triton」，而不是 TableGen/LLVM 式的外部 DSL，是一个 deliberate 的工程权衡：

- **与 PyTorch 同栈**：lowering 逻辑和 codegen 模板可以共享工具链、快速迭代；新算子从 ATen 到 Triton 的路径短。
- **编译期成本可接受**：字符串拼接、模板展开发生在编译时，符合第 2.1 节 **编译重、运行轻** 原则。
- **放弃的是**：codegen 模块本身难以形式化验证；生成代码的质量高度依赖模板作者的工程经验。

**生成 vs 模板：两条路径** 。并非所有 kernel 都值得「逐行生成」：

.. list-table::
   :header-rows: 1

   * - 路径
     - 适用
     - 原因
   * - **逐行生成** （ ``TritonKernel`` ）
     - pointwise、reduction 等结构规则的操作
     - 模式统一，参数主要是 tiling / mask
   * - **预定义模板** （ ``TemplateBuffer`` ）
     - GEMM、conv 等计算密集型 op
     - 手写模板已达 Tensor Core 上限，生成器拼不出更好版本

第 6.4 节的 note 已点明：**逐元素归编译器生成，矩阵乘交给模板**——这是 Inductor codegen 最核心的分工 invariant。违反它（例如用 generic pointwise 逻辑拼 mm）通常意味着数量级的性能损失。

**Wrapper 层存在的理由** 。Codegen 产出的是 kernel**源码字符串** ，还不能直接跑。Wrapper（ ``wrapper.py`` / ``cpp_wrapper_*.py`` ）负责：

- 计算 grid/block、分配临时 buffer、管理 CUDA stream
- 把多个 kernel 的 launch 串成一次 Python/C++ 调用
- AOTInductor 路径下切换为 C++ wrapper，减少 Python 解释器开销（第 9.5 节）

可以把 Scheduler 的输出想象成「施工图纸」，Codegen 是「工厂生产预制件」，Wrapper 是「现场吊装」——三层分离，便于 GPU/CPU/AOT 部署形态各自替换最后一环。

代码生成（Codegen）是 Inductor 编译流水线的最后一步。当 Scheduler 完成 IRNode 的融合和调度后，它将每个 ``FusedSchedulerNode`` 分发给对应设备的代码生成器，后者将 IRNode 翻译为可编译的源代码。

这一节从整体上了解代码生成器的架构和它在整个流程中的位置。

代码生成器架构
====================

Inductor 的代码生成器位于 ``pytorch/torch/_inductor/codegen/`` 目录，采用 **双层架构** ：

1.**Scheduling 层** ：负责将 Scheduler 输出的节点翻译为源代码字符串。不同类型的设备有不同的 Scheduling 实现。
2.**Wrapper 层** ：负责生成调用 kernel 的 Python 包装器代码，包括 kernel launch 参数、grid 计算、stream 管理等。

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

Scheduler 在调用 ``codegen()`` 时，通过 ``get_scheduling_for_device()`` （定义在 ``common.py`` 中）根据设备类型选择对应的 Scheduling 实现：

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

对于融合后的 ``FusedSchedulerNode`` ，codegen 需要将多个 IRNode 的 ``inner_fn`` 组合到同一个 kernel 的循环体中。这是代码生成器最复杂的工作——它必须正确地合并多个循环体、处理不同 IRNode 之间的数据依赖、并消除中间结果的显存访问。

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

这种映射通过 ``ops_handler.py`` 中的 ``OpsHandler`` 接口实现。每个后端实现自己的 ``OpsHandler`` ，在 codegen 时替换全局的 ``V.ops`` ：

.. code-block:: python

   # Triton 后端设置自己的 ops handler
   with V.set_ops_handler(TritonOverrides()):
       # 此范围内的 inner_fn 调用使用 Triton 语义
       value = inner_fn(index)

小结
======

这一节从整体上了解了代码生成器的架构：

- **双层架构** ：Scheduling 层生成 kernel 代码，Wrapper 层生成 launch 代码
- **设备注册** ：通过 ``device_codegens`` 字典选择对应后端的 Scheduling 实现
- **ops 映射** ：IR 层面的 ``ops.*`` 原语被映射为具体硬件指令
- **Wrapper 代码** ：Python 代码，负责编排 kernel launch
