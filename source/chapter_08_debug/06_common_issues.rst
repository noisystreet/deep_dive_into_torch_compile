.. _common-issues:

============
常见问题
============

.. tip::

   **近 20% 的 Inductor 提交是在修 Bug——这说明踩坑是常态。 **
   在 Inductor 的 8,787 次提交中，明确标注为 bug fix 的约 1,709 次（19.5%）。再加上 1,012 次 revert（11.5%），超过 30% 的提交是在"解决问题"。这反映了一个现实：torch.compile 是一个复杂度极高的编译器项目，你不一定每次都能顺利跑通。遇到问题时，日志系统和 minimizer 是你最强的盟友——它们被设计出来就是为了应对这些 bug 的。团队在 PyTorch 2.2 中统一了 ``TORCH_LOGS`` 日志系统，在 2.5 中引入了区域编译以减少重新编译时的 bug 触发概率。但编译器 bug 永远不会消失——唯一能做的是让调试工具更好用。

这一节汇总使用 torch.compile 时最常见的错误和问题，以及对应的解决方法。

编译失败
============

**"Unsupported: call_function ..."**

原因：Dynamo 无法追踪某个 Python 操作。通常是因为操作超出了 Dynamo 的字节码覆盖范围。

解决方法：

.. code-block:: python

   # 方式一：降级到 eager
   @torch.compiler.disable
   def unsupported_fn(x):
       return custom_operation(x)

   # 方式二：注册 fallback
   from torch._dynamo import allow_in_graph
   allow_in_graph(custom_function)

   # 方式三：使用 torch.compiler.compilation_options
   @torch.compile(backend="eager")
   def fn(x):
       return custom_operation(x)

**"MissingOperatorWithoutDecomp"**

原因：Inductor 找不到某个 ATen 算子的 lowering 函数，也无法通过 decomposition 展开。

解决方法：

.. code-block:: python

   # 方式一：注册 decomposition
   from torch._decomp import register_decomposition
   
   @register_decomposition(aten.custom_op)
   def custom_op_decomp(x):
       return aten.sin(x) + aten.cos(x)

   # 方式二：注册 fallback（回退到 eager）
   from torch._inductor.lowering import make_fallback
   make_fallback(aten.custom_op)

   # 方式三：在 config 中启用隐式 fallback
   torch._inductor.config.implicit_fallbacks = True

结果不一致
============

**编译结果与 eager 模式结果不同**

当编译后的输出与 eager 模式的输出存在差异时（精度问题）：

.. code-block:: bash

   TORCHDYNAMO_REPRO_AFTER="aot" python train.py

这会生成最小复现脚本。常见原因：

1.**浮点精度差异**：Triton 和 CUDA 的归约顺序可能不同，导致微小差异。这是正常的，通常差异在 1e-6 级别。如果差异过大，检查是否误用了 ``tf32`` 或 ``fp16`` 。

2.**随机数生成差异**：编译后的 dropout 等随机操作可能与 eager 模式顺序不同。设置相同的随机种子：

   .. code-block:: python

      torch.manual_seed(42)
      torch.cuda.manual_seed_all(42)

3.**In-place 操作语义** ：功能化（functionalization）可能改变了 in-place 操作的语义。检查是否在编译图中正确处理了 in-place 操作的副作用。

性能比 Eager 还差
====================

**首次运行很慢** （正常）。编译本身有开销。首轮训练通常比 eager 慢，后续轮次会更快。

**Graph break 过多** 。如果模型中有大量 graph break，编译后的性能可能不如 eager：

.. code-block:: bash

   TORCH_LOGS="+perf_hints" python train.py

如果看到类似以下的日志，说明 graph break 过多：

.. code-block:: text

   [perf_hints] Graph break 10 次，产生了 11 个子图
   [perf_hints] 建议合并子图以避免 Python 解释器开销

解决方法：

- 使用 ``fullgraph=True`` 强制无 graph break
- 将 graph break 的代码用 ``torch.compiler.disable`` 隔离

**Kernel launch 开销过大** 。如果 Inductor 生成了大量小 kernel：

.. code-block:: bash

   TORCH_LOGS="+inductor" python train.py

看日志中生成的 kernel 数量。如果超过 100 个 kernel，而模型本身不大，说明融合不充分。尝试：

.. code-block:: python

   torch._inductor.config.max_fusion_size = 10

**内存不足（OOM）** 。编译后可能使用更多显存（因为保存了中间结果用于反向）。尝试：

.. code-block:: python

   # 减少保存的中间结果（增加重计算）
   torch._inductor.config.recompute_threshold = 100

与 DataLoader 配合
====================

如果 DataLoader 产出的 tensor 形状不稳定：

.. code-block:: python

   # 确保 DataLoader 产出固定形状
   def collate_fn(batch):
       # padding 到固定长度
       max_len = max(x.size(0) for x, _ in batch)
       padded = torch.stack([
           torch.nn.functional.pad(x, (0, max_len - x.size(0)))
           for x, _ in batch
       ])
       return padded
   
   dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn,
                            persistent_workers=True)

如果形状必须变化，使用动态形状（见 8.5 节）。

与分布式训练配合
====================

使用 DDP（Distributed Data Parallel）时：

.. code-block:: python

   # DDP 和 torch.compile 配合使用
   model = MyModel()
   model = torch.compile(model)
   model = torch.nn.parallel.DistributedDataParallel(model)

   # 注意：DDP 包装的顺序很重要
   # DDP 放在 compile 外面

使用 FSDP（Fully Sharded Data Parallel）时：

.. code-block:: python

   # FSDP + torch.compile
   model = MyModel()
   model = torch.compile(model)
   model = FullyShardedDataParallel(model)

   # 同样，compile 在 FSDP 之前

更多关于分布式训练和 torch.compile 配合使用的细节，见第 10 章的实战案例。

问题排查决策树
====================

遇到 torch.compile 相关问题时，可以按以下决策树快速定位：

.. mermaid::

   graph TD
       A["遇到 torch.compile 问题"] --> B{"问题类型"}
       B --> C["编译失败"]
       B --> D["结果不一致"]
       B --> E["性能差"]
       B --> F["显存不足"]

       C --> C1{"错误信息"}
       C1 -->|"Unsupported"| C2["禁用不支持操作<br/>或注册 fallback"]
       C1 -->|"MissingOperator"| C3["注册 decomposition<br/>或启用 implicit_fallbacks"]
       C1 -->|"CUDA Graph 失败"| C4["禁用 CUDA Graph<br/>或检查图捕获要求"]

       D --> D1{"差异类型"}
       D1 -->|"精度差异"| D2["检查浮点精度设置<br/>设置随机种子"]
       D1 -->|"结果错误"| D3["使用 TORCHDYNAMO_REPRO_AFTER<br/>生成 minimizer 脚本"]

       E --> E1{"可能原因"}
       E1 -->|"Graph break"| E2["使用 fullgraph=True<br/>检查 TORCH_LOGS=+perf_hints"]
       E1 -->|"Kernel launch 开销"| E3["使用 reduce-overhead<br/>或 CUDA Graph"]
       E1 -->|"动态形状重编译"| E4["标记动态维度<br/>或使用 padding"]

       F --> F1{"排查方向"}
       F1 -->|"中间结果过多"| F2["增加 recompute_threshold"]
       F1 -->|"编译缓存过大"| F3["减小 cache_size_limit"]
       F1 -->|"内存泄漏"| F4["检查 tensor 引用<br/>使用 memory snapshot"]

Eager vs Compiled 显存使用模式对比
==============================================

编译后的模型和 eager 模式的显存使用模式有本质区别。下图展示了两种模式的差异：

.. mermaid::

   graph LR
       subgraph EagerMem["Eager 模式显存使用"]
           E1["Forward: 逐层分配显存"]
           E2["Backward: 逐层释放 Forward 中间结果"]
           E3["显存峰值发生在<br/>forward 结束时"]
           E1 --> E2
       end

       subgraph CompiledMem["Compiled 模式显存使用"]
           C1["编译图保持所有中间结果"]
           C2["Kernel 执行时分配和释放<br/>但生命周期管理更复杂"]
           C3["显存峰值可能更高<br/>因为融合保留了更多中间 tensor"]
           C4["CUDA Graph 额外占用<br/>重放缓冲区"]
           C1 --> C2 --> C3 --> C4
       end

Eager 模式下，每个 ATen 操作立即执行，中间结果的分配和释放是即时且局部的。Compiled 模式下，编译器将多个操作融合为一个 kernel，需要同时保留所有融合操作的中间结果，导致显存峰值可能更高。此外，CUDA Graph 模式会预分配重放缓冲区，进一步增加显存占用。

CUDA Graph 捕获失败
==========================

``mode="reduce-overhead"`` 内部使用 CUDA Graph 来减少 kernel launch 开销。但 CUDA Graph 对捕获的 kernel 有严格的限制，某些情况下会捕获失败。

常见失败原因
----------------

.. list-table:: CUDA Graph 捕获失败常见原因
   :header-rows: 1

   * - 原因
     - 表现
     - 解决方案
   * - CPU 同步操作
     - ``.item()`` 、 ``.cpu()`` 、 ``.numpy()``
     - 移出编译图
   * - 动态控制流
     - ``if tensor.item() > 0:``
     - 使用 ``torch.cond`` 或移出图
   * - 内存分配/释放
     - ``torch.cuda.empty_cache()``
     - 禁用 CUDA Graph
   * - 非确定性操作
     - ``torch.cuda.synchronize()``
     - 移出编译区域
   * - 设备间传输
     - ``tensor.cuda()`` 、 ``tensor.cpu()``
     - 统一设备

诊断 CUDA Graph 失败
-------------------------

使用 ``TORCH_LOGS="+cuda_graphs"`` 查看详细的 CUDA Graph 捕获日志：

.. code-block:: bash

   TORCH_LOGS="+cuda_graphs" python train.py

日志输出示例：

.. code-block:: text

   [cuda_graphs] 尝试捕获 CUDA Graph...
   [cuda_graphs] 捕获失败: 检测到 CPU 同步操作
   [cuda_graphs]   - 位置: forward() 中的 .item() 调用
   [cuda_graphs] 回退到 eager 模式

禁用 CUDA Graph
-------------------

如果 CUDA Graph 反复失败，可以显式禁用它：

.. code-block:: python

   import torch._inductor.config as inductor_config

   # 全局禁用 CUDA Graph
   inductor_config.triton.cudagraphs = False

   # 或者对特定函数禁用
   @torch.compile(options={"triton.cudagraphs": False})
   def fn(x):
       return torch.sin(x)

mode="reduce-overhead" 与 CUDA Graph 的关系
-------------------------------------------------------

``mode="reduce-overhead"`` 等价于启用 CUDA Graph + 其他 launch 优化：

.. code-block:: python

   # 这两者等价
   torch.compile(model, mode="reduce-overhead")

   # 等价于
   torch.compile(model, options={
       "triton.cudagraphs": True,
       "max_autotune": False,
   })

当 CUDA Graph 捕获失败时， ``reduce-overhead`` 模式会自动回退到常规 kernel launch，优化效果会打折扣。

.. note::

   **CUDA Graph 回退是透明的 ** 。
   即使 ``mode="reduce-overhead"`` 下的 CUDA Graph 捕获失败，模型仍然能正常运行——只是退回到常规的 kernel launch 路径，性能提升幅度会减小。你不会看到明显的错误信息，但性能可能不如预期。此时检查 ``cuda_graphs`` 日志就能发现问题。

Export 模式下的常见问题
==============================

使用 ``torch.export`` 导出编译后的模型时，会遇到一些特有的问题。

基本用法回顾
----------------

.. code-block:: python

   import torch

   class MyModel(torch.nn.Module):
       def forward(self, x):
           return torch.sin(x) + torch.cos(x)

   model = MyModel()
   exported = torch.export.export(model, (torch.randn(4, 4),))

常见 Export 错误及其修复
------------------------------

**错误 1：Unsupported operator in export**

.. code-block:: text

   torch.export.export(): 不支持的操作 aten::_unsafe_view
   在 FX graph 中发现了非 export 安全的操作

解决方案：使用 decomposition 替换不支持的操作

.. code-block:: python

   from torch._decomp import get_decompositions

   # 获取指定操作的 decomposition
   decompositions = get_decompositions([aten._unsafe_view])
   exported = torch.export.export(model, args, decompositions=decompositions)

**错误 2：动态形状导致的 export 失败**

.. code-block:: text

   Export 失败：检测到动态形状，但未提供 dynamic_shapes 参数

解决方案：为 export 指定动态形状约束

.. code-block:: python

   from torch._export import DynamicShapes

   dynamic_shapes = {
       "x": {0: torch.export.Dim("batch_size", min=1, max=128)},
   }
   exported = torch.export.export(model, (x,), dynamic_shapes=dynamic_shapes)

**错误 3：Control flow 不支持**

.. code-block:: text

   Export 不支持动态控制流：if tensor.item() > 0

解决方案：使用 ``torch.cond`` 替代动态控制流

.. code-block:: python

   import torch._higher_order_ops.cond as cond_ops

   def true_fn(x):
       return x * 2

   def false_fn(x):
       return x / 2

   def forward(self, x, condition):
       return torch.cond(condition, true_fn, false_fn, [x])

Export 与 Dynamic Shapes 的协同
--------------------------------------

Export 模式的动态形状处理与 ``torch.compile`` 不同：

.. list-table:: Export vs torch.compile 动态形状对比
   :header-rows: 1

   * - 特性
     - torch.compile dynamic=True
     - torch.export
   * - 动态维度声明
     - 隐式推断
     - 显式声明
   * - 约束检查
     - 运行时 guard
     - 编译时验证
   * - Shape 范围
     - 自动推断 min/max
     - 用户指定范围
   * - 适用场景
     - 训练、推理
     - 部署、序列化

.. warning::

   导出的模型在部署时如果遇到超出声明范围的形状，会产生运行时错误。因此 export 时声明的动态形状范围应该覆盖所有可能的输入形状。

编译缓存相关的问题
==========================

Cache Poisoning（缓存中毒）
------------------------------

缓存中毒是指 Dynamo 的编译缓存返回了与当前输入不匹配的编译结果。

.. code-block:: python

   # 缓存中毒的典型场景
   @torch.compile
   def fn(x, training=True):
       if training:
           return torch.dropout(x, 0.5)
       else:
           return x

   fn(x, training=True)   # 编译，缓存 training=True 的版本
   fn(x, training=False)  # Guard failed，重新编译
   fn(x, training=True)   # 命中缓存，正确
   fn(x, training=False)  # 命中缓存，正确（第二次）

缓存中毒很少见，但可能在以下情况发生：

1. **全局状态被修改** ：编译时假设的全局状态在运行时被改变
2.**张量的元数据变化但 guard 没有捕获** ：如 ``requires_grad`` 变化
3.**自定义 Python 对象的相等性判断异常** ： ``__eq__`` 实现不正确

如何清除和失效缓存
---------------------------

.. code-block:: python

   import torch._dynamo.config as config

   # 方法一：清空所有编译缓存
   torch._dynamo.reset()

   # 方法二：清空指定函数的缓存
   # （无法直接清空单个函数的缓存，
   # 因为缓存是全局的）

   # 方法三：禁用缓存
   config.cache_size_limit = 0
   # 注意：这会导致每次调用都重新编译

   # 方法四：增加缓存的 key 维度
   # 确保不同状态的模型有不同的 cache key
   # 例如使用不同的模型实例

磁盘缓存 vs 内存缓存
--------------------------

.. list-table:: 磁盘缓存 vs 内存缓存
   :header-rows: 1

   * - 特性
     - 内存缓存 (Dynamo)
     - 磁盘缓存 (Inductor)
   * - 存储位置
     - RAM
     - ``~/.cache/torch/inductor/``
   * - 缓存内容
     - FX Graph + Guard
     - 编译好的 Triton kernel (so 文件)
   * - 生命周期
     - 进程生命周期
     - 永久（可手动删除）
   * - 失效方式
     - ``torch._dynamo.reset()``
     - 删除缓存目录
   * - 大小限制
     - ``cache_size_limit`` (默认 64)
     - 无限制（使用 LRU 驱逐）

磁盘缓存可以通过环境变量控制位置和禁用：

.. code-block:: bash

   # 修改磁盘缓存位置
   TORCHINDUCTOR_CACHE_DIR=/path/to/cache python train.py

   # 禁用磁盘缓存
   TORCHINDUCTOR_CACHE_FORCE_FINGERPRINT_MATCH=0 python train.py

   # 清除所有磁盘缓存
   rm -rf ~/.cache/torch/inductor/

.. tip::

   **何时需要清除磁盘缓存？**
   当你升级 PyTorch 或 Triton 版本后，旧版本的编译缓存可能不兼容。此时应该清除磁盘缓存，否则 Inductor 可能会加载不兼容的 .so 文件导致段错误（segfault）。升级后运行一次 ``rm -rf ~/.cache/torch/inductor/`` 是个好习惯。

跨设备和跨精度问题
==========================

CPU vs GPU 编译差异
------------------------

torch.compile 的主要优化目标是 GPU，CPU 后端的优化程度远不如 GPU：

.. list-table:: CPU vs GPU 编译差异
   :header-rows: 1

   * - 方面
     - GPU 编译
     - CPU 编译
   * - 默认后端
     - Inductor (Triton)
     - Inductor (C++) 或 Eager
   * - Kernel 生成
     - Triton kernel
     - C++ kernel (通过 C++ codegen)
   * - 优化程度
     - 高度优化
     - 基础优化
   * - 常见问题
     - CUDA Graph 失败、OOM
     - C++ 编译错误、性能无提升

CPU 编译的典型问题：

.. code-block:: python

   # CPU 编译常见问题：C++ 编译错误
   @torch.compile(backend="inductor")
   def fn(x):
       return x.unique()  # unique 在 CPU 后端可能不支持

   # 解决方案：回退到 eager
   @torch.compile(backend="eager")
   def fn(x):
       return x.unique()

AMP (Automatic Mixed Precision) 与 compile 的交互
------------------------------------------------------

AMP 和 torch.compile 的配合需要特别注意顺序：

.. code-block:: python

   import torch
   from torch.cuda.amp import autocast

   model = torch.compile(model)

   # 正确用法：autocast 在外面，compile 在里面
   with autocast(dtype=torch.float16):
       output = model(x)  # model 是 compiled 的

   # 错误用法：compile 不会自动处理 autocast 上下文
   # （但不会报错，只是精度可能不符合预期）

AMP 与 compile 配合时的常见问题：

1. **精度不匹配** ：某些操作在 FP16 下精度不足，编译后可能放大误差
2.**Loss scaling 失效** ：编译后的 autograd 可能改变梯度 scale 的行为
3.**Dynamic shapes + AMP** ：同时启用时，Triton 生成的 kernel 需要同时处理数据类型转换和符号形状，可能降低融合效率

.. code-block:: python

   # AMP + compile 的最佳实践
   import torch._inductor.config as inductor_config

   # 确保 triton kernel 使用正确的数据类型
   inductor_config.force_fp16 = True  # 强制所有 kernel 使用 FP16

   # 或者手动控制精度
   @torch.compile
   def forward(x):
       with autocast(dtype=torch.float16):
           return model(x)

BF16 vs FP16 vs FP32 的考量
-----------------------------------

不同的浮点精度对编译后的模型性能有显著影响：

.. list-table:: BF16 vs FP16 vs FP32 for Compiled Models
   :header-rows: 1

   * - 精度
     - 数值范围
     - 精度
     - Compile 优势
     - 适用场景
   * - FP32
     - 最大
     - 高
     - 最稳定，融合最安全
     - 精度敏感型任务
   * - FP16
     - 中
     - 低
     - Triton kernel 速度最快
     - 对精度不敏感的推理
   * - BF16
     - 最大
     - 中
     - 数值范围与 FP32 相同
     - 训练（推荐）

.. code-block:: python

   # 精度配置示例
   @torch.compile
   def fn(x):
       # FP32 (默认)
       return torch.sin(x) + torch.cos(x)

   @torch.compile
   def fn_fp16(x):
       # FP16 - Triton kernel 使用 tl.float16
       return torch.sin(x.half()) + torch.cos(x.half())

   @torch.compile
   def fn_bf16(x):
       # BF16 - Triton kernel 使用 tl.bfloat16
       return torch.sin(x.bfloat16()) + torch.cos(x.bfloat16())

.. warning::

   **FP16 下的梯度下溢 ** 。
   使用 FP16 + torch.compile 时，如果损失函数很小（如 < 1e-3），梯度可能在 FP16 下溢为零。此时应该使用 BF16 或开启 loss scaling。BF16 保留了与 FP32 相同的指数位，因此不存在下溢问题。

Inductor 特定错误
=========================

Inductor 作为 torch.compile 的默认 GPU 后端，有一些特定的错误和调试方法。

常见 Lowering 错误
-----------------------

Lowering（降级）是指将 ATen IR 转换为 Triton 代码的过程。常见的 lowering 错误：

.. code-block:: text

   # 错误 1：不支持的算子
   [inductor] Lowering failed for aten::_native_batch_norm_legit
   [inductor] 无法找到对应的 Triton kernel 实现

   # 错误 2：类型不匹配
   [inductor] Type mismatch: expected float32, got float64
   [inductor] 算子 aten::mm 的输入类型不一致

   # 错误 3：形状推断失败
   [inductor] Shape inference failed for aten::_unsafe_view
   [inductor] 无法确定输出的符号形状

解决方案：

.. code-block:: python

   # 启用隐式 fallback（回退到 eager）
   torch._inductor.config.implicit_fallbacks = True

   # 注册自定义 decomposition
   from torch._decomp import register_decomposition

   @register_decomposition(aten.custom_op)
   def custom_op_decomp(x):
       return aten.sin(x) + aten.cos(x)

   # 注册 fallback 算子
   from torch._inductor.lowering import make_fallback
   make_fallback(aten.custom_op)

使用 ``TORCH_LOGS="+inductor"`` 查看详细的 lowering 过程：

.. code-block:: bash

   TORCH_LOGS="+inductor,+lowering" python train.py

MaxAutotune 失败
--------------------

``mode="max-autotune"`` 会在编译时对每个 kernel 进行基准测试以选择最佳 tiling 参数。这个过程可能失败：

.. code-block:: text

   [inductor] Max-autotune 失败: CUDA 内存不足
   [inductor]   当前显存: 4.2GB/8.0GB
   [inductor]   Kernel: triton_poi_fused_0
   [inductor]   自动回退到 Heuristic 选择

常见原因：

1.**显存不足** ：max-autotune 需要额外显存来并行基准测试多个 kernel 变体
2.**编译超时** ：某些 kernel 的 autotune 空间过大
3.**Triton 编译错误** ：autotune 生成的某些 kernel 变体无法通过 Triton 编译

解决方案：

.. code-block:: python

   # 减小 autotune 的搜索空间
   torch._inductor.config.max_autotune = True
   torch._inductor.config.max_autotune_search_space = 'small'  # 'small', 'medium', 'large'

   # 限制 autotune 的 kernel 数量
   torch._inductor.config.max_autotune_pointwise = 10
   torch._inductor.config.max_autotune_gemm = 10

   # 对特定 kernel 禁用 autotune
   torch._inductor.config.autotune_in_subproc = False  # 在子进程中 autotune

.. note::

   **MaxAutotune 失败是透明的** 。
   如果某个 kernel 的 autotune 失败，Inductor 会回退到 heuristic 的 tiling 选择。模型仍然可以正常运行，只是该 kernel 的性能可能未达最优。你不会看到红色错误，而是在日志中看到一条回退消息。

Pattern Matcher 失败
-------------------------

Inductor 使用 Pattern Matcher 将多个 ATen 操作匹配为特定的融合模式。当匹配失败时，融合效果会下降。

.. code-block:: bash

   # 查看 Pattern Matcher 的匹配日志
   TORCH_LOGS="+pattern_matcher" python train.py

   # 输出示例
   # [pattern_matcher] 尝试匹配 pattern: addmm
   # [pattern_matcher] 匹配失败: 形状不匹配
   # [pattern_matcher] 节点 12 (aten::mm) 的输出形状与节点 15 (aten::add) 不兼容

常见的 Pattern Matcher 失败原因：

.. list-table:: Pattern Matcher 失败原因
   :header-rows: 1

   * - 失败原因
     - 表现
     - 解决方案
   * - 形状不兼容
     - 相邻算子的 tensor 形状无法对齐
     - 检查模型中是否有 reshape/transpose 打乱了形状
   * - 数据类型不一致
     - 两个算子的输入输出 dtype 不同
     - 统一数据类型
   * - Device 不一致
     - 算子在 CPU 和 GPU 之间切换
     - 确保所有操作在同一设备
   * - Pattern 未被注册
     - Inductor 不知道某种操作组合可以融合
     - 使用 ``@torch.compile(fullgraph=True)`` 强制融合

调试 Pattern Matcher 的最佳方式：

.. code-block:: python

   # 启用 pattern matcher 的详细日志
   import logging
   logging.getLogger("torch._inductor.pattern_matcher").setLevel(logging.DEBUG)

   # 或者通过环境变量
   # TORCH_LOGS="+pattern_matcher" python train.py

   # 查看生成的 FX graph 是否包含可融合的模式
   @torch.compile
   def fn(x):
       return torch.sin(torch.cos(x))

   # 如果 pattern matcher 工作正常，sin 和 cos 会融合为一个 kernel
   # 如果失败，你会看到两个独立的 kernel

.. seealso::

   关于 Inductor 的 Pattern Matcher 内部实现细节和融合策略，详见第 5 章的 Scheduler 相关章节。
