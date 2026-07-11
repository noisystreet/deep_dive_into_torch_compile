.. _aot-debugging:

=========================
AOTAutograd 调试与排查
=========================

在开发和优化 ``torch.compile`` 模型时，AOTAutograd 是整个编译流水线中承上启下的关键环节。本节介绍如何观察、调试和排查 AOTAutograd 的行为——从日志输出到图分区决策，从工具函数到生命周期钩子。

.. tip::

   **调试 AOTAutograd 的价值。** 很多 ``torch.compile`` 的编译错误（如"joint graph 构建失败"、"分区后梯度不匹配"）的根因都在 AOTAutograd 层。学会调试 AOTAutograd，等于掌握了整个编译栈中"最难定位问题"的排查能力。相比之下，Inductor 的错误通常表现为 CUDA error 或 kernel 编译失败，Dynamo 的错误通常表现为 Python 级回溯——AOTAutograd 的错误介于两者之间，既不像 Dynamo 那样容易复现，也不像 Inductor 那样有明确的错误信息，最需要系统化的调试方法。


使用 TORCH_LOGS 观察 AOTAutograd
=========================================

``TORCH_LOGS`` 是调试 ``torch.compile`` 最方便的工具。设置 ``TORCH_LOGS="+aot"`` 可以输出 AOTAutograd 的完整信息，包括联合图（joint graph）、分区后的前向/反向子图以及编译过程中的关键日志。

基本用法

.. code-block:: bash

   TORCH_LOGS="+aot" python -c "
   import torch

   def fn(x, y):
       return (x * y).sum()

   compiled_fn = torch.compile(fn, fullgraph=True)
   x = torch.randn(3, requires_grad=True)
   y = torch.randn(3)
   result = compiled_fn(x, y)
   result.backward()
   "

运行后会得到类似以下的输出（为便于阅读，已简化并添加注释）：

.. code-block:: text

   [INFO] 图模式训练: aot_dispatch_autograd_graph
   [INFO] 创建联合图...
   [INFO] ==== 联合图（Joint Graph）开始 ====
   graph():
       %primals_1 : [num_users=1] = placeholder[target=primals_1]
       %primals_2 : [num_users=1] = placeholder[target=primals_2]
       %tangents_1 : [num_users=1] = placeholder[target=tangents_1]
       %mul : [num_users=1] = aten.mul.Tensor(%primals_1, %primals_2)
       %sum_1 : [num_users=1] = aten.sum.default(%mul)
       %grad_output : [num_users=1] = aten.expand(%tangents_1, ...)
       %grad_mul : [num_users=1] = aten.mul.Tensor(%primals_2, %grad_output)
       %grad_primals_1 : [num_users=1] = aten.sum.default(%grad_mul)
       return (%sum_1, %grad_primals_1)
   [INFO] ==== 联合图结束 ====
   [INFO] 使用默认分区策略...
   [INFO] 前向子图:
   graph():
       %primals_1 : [num_users=1] = placeholder[target=primals_1]
       %primals_2 : [num_users=1] = placeholder[target=primals_2]
       %mul : [num_users=1] = aten.mul.Tensor(%primals_1, %primals_2)
       %sum_1 : [num_users=1] = aten.sum.default(%mul)
       return (%sum_1, %mul)
   [INFO]
   [INFO] 反向子图:
   graph():
       %mul : [num_users=1] = placeholder[target=mul]
       %tangents_1 : [num_users=1] = placeholder[target=tangents_1]
       %primals_2 : [num_users=1] = placeholder[target=primals_2]
       %grad_output : [num_users=1] = aten.expand(...)
       %grad_mul : [num_users=1] = aten.mul.Tensor(%primals_2, %grad_output)
       %grad_primals_1 : [num_users=1] = aten.sum.default(%grad_mul)
       return (%grad_primals_1,)

.. tip::

   如果只想看 AOTAutograd 相关的日志而不想看 Dynamo 的 guard 和 trace 信息，可以用 ``TORCH_LOGS="aot_graphs"`` 只输出图信息： ``TORCH_LOGS="aot_graphs" python script.py`` 。

日志内容的解读

- **联合图部分** 同时包含前向节点（ ``aten.mul`` 、 ``aten.sum`` ）和反向节点（ ``aten.expand`` 、 ``aten.mul`` 、 ``aten.sum`` 中的梯度部分）。占位符 ``primals_*`` 是前向输入和中间结果， ``tangents_*`` 是反向输入的梯度。
- **前向子图部分** 是分区后的前向部分，其输出除了最终结果外，还包含反向需要的 saved tensors（此处为 ``%mul`` ）。
- **反向子图部分** 通过占位符接收前向保存的 tensor 和反向输入，计算出梯度。

AOTAutograd 日志输出的整体流程：

.. mermaid::

   graph TD
       Start["设置 TORCH_LOGS='+aot'"] --> Dynamo["Dynamo 捕获 FX Graph"]
       Dynamo --> AOT["AOTAutograd 开始"]
       AOT --> Joint["创建联合图<br/>（Joint Graph）"]
       Joint -->|日志输出| ShowJoint["打印联合图内容"]
       ShowJoint --> Partition["图分区"]
       Partition -->|日志输出| ShowPart["打印分区策略"]
       ShowPart --> Split["分割为前向/反向子图"]
       Split -->|日志输出| ShowFwd["打印前向子图"]
       ShowFwd --> ShowBwd["打印反向子图"]
       ShowBwd --> Compile["Inductor 编译"]

更细粒度的日志控制

``TORCH_LOGS`` 除了 ``+aot`` 外，还可以组合使用更细粒度的选项，只关注 AOTAutograd 特定阶段的输出：

.. code-block:: bash

   # 只看 joint graph 的创建过程
   TORCH_LOGS="aot_joint_graph" python script.py

   # 看分区决策的详细日志
   TORCH_LOGS="aot_partition" python script.py

   # 看 functionalization 的转换过程
   TORCH_LOGS="aot_functionalization" python script.py

   # 同时观察多个阶段
   TORCH_LOGS="aot_joint_graph,aot_partition" python script.py

不同日志级别的输出内容：

.. list-table::
   :header-rows: 1

   * - 日志选项
     - 输出内容
     - 适用场景
   * - ``+aot``
     - 完整的 AOTAutograd 流程日志
     - 初次调试，了解全貌
   * - ``aot_graphs``
     - 联合图与前/反向子图
     - 只关心图结构
   * - ``aot_joint_graph``
     - 联合图的创建过程
     - 排查 joint trace 失败
   * - ``aot_partition``
     - 分区决策细节
     - 排查 saved tensor 不匹配
   * - ``aot_functionalization``
     - functionalization 转换细节
     - 排查 in-place 操作问题


使用 torch._functorch 工具函数
======================================

除了日志输出，PyTorch 还提供了一组位于 ``torch._functorch`` 下的工具函数，可以直接在 Python 代码中调用，用于检查 AOTAutograd 的中间产物。

log_compilation_event

``torch._functorch.aot_autograd.log_compilation_event`` 是 AOTAutograd 内置的日志记录函数，用于记录每个编译事件。当你在自定义后端中包装 AOTAutograd 时，可以用它来输出结构化日志：

.. code-block:: python

   import torch._functorch.aot_autograd as aot

   # AOTAutograd 在每次编译时会调用这个函数
   # 记录编译的输入、输出、耗时等信息
   aot.log_compilation_event({
       "event": "aot_compile_start",
       "num_params": 10,
       "num_ops_in_graph": 50,
   })

这个函数在 AOTAutograd 内部被用于记录编译事件到 ``TORCH_LOGS`` ，也可以被用户的自定义代码调用，用于扩展日志记录。

nop 编译器

``torch._functorch.compilers.nop`` 是一个"空操作"编译器，它接收 FX Graph 后不做任何实际的 lowering，仅仅返回原图。这个函数在调试中非常有用——当你怀疑 Inductor 的 lowering 有问题时，可以用 ``nop`` 替换 Inductor 作为后端，验证 AOTAutograd 自身的输出是否正确：

.. code-block:: python

   from torch._functorch.compilers import nop

   def fn(x, y):
       return (x * y).sum()

   # 使用 nop 编译器——AOTAutograd 正常执行，
   # 但分区后的子图不会被 Inductor 进一步编译
   compiled_fn = torch.compile(fn, backend=nop, fullgraph=True)

   x = torch.randn(3, requires_grad=True)
   y = torch.randn(3)
   result = compiled_fn(x, y)      # 前向使用原始 FX Graph 执行
   result.backward()               # 反向使用原始 FX Graph 执行

使用 ``nop`` 编译器时，AOTAutograd 完整的 joint trace 和分区流程都会执行，你可以通过 ``TORCH_LOGS="+aot"`` 看到所有中间图。如果此时模型可以正常运行但使用 Inductor 时报错，问题就定位在 Inductor 侧。

调试模式下也有等效的 ``"aot_eager"`` 后端可用：

.. code-block:: python

   # aot_eager 等价于 "AOTAutograd + eager 执行"，
   # 即只做 joint trace 和分区，但子图交给 eager 模式执行
   compiled_fn = torch.compile(fn, backend="aot_eager")

二者的区别在于： ``nop`` 直接返回 FX Graph 作为可调用对象， ``aot_eager`` 则使用 eager 模式执行分区后的子图。两者都跳过了 Inductor，但 ``aot_eager`` 更贴近真实执行路径。

.. warning::

   ``nop`` 返回的 FX Graph 不是标准的 ``torch.fx.GraphModule`` 的 ``forward`` 方法，它是直接由 ``fx.Graph`` 的可执行闭包包装而成。在生产代码中，请优先使用 ``"aot_eager"`` 后端而非 ``nop`` ，因为 ``"aot_eager"`` 的兼容性更好。

自定义回调函数检查中间图

编写一个自定义回调函数，在 AOTAutograd 的流水线中插入检查点，可以更精细地控制调试过程。AOTAutograd 的 ``aot_function`` 和 ``aot_export_module`` 都接受可选的 ``fw_compiler`` 和 ``bw_compiler`` 参数，我们可以用包装函数在其中插入打印或断言：

.. code-block:: python

   import torch
   from torch._functorch.aot_autograd import aot_function

   def debug_fw_compiler(gm, example_inputs):
       """自定义前向编译器：打印图结构后，再用 eager 执行"""
       print("=== 前向子图 ===")
       gm.graph.print_tabular()
       print(f"输入个数: {len(example_inputs)}")
       # 验证图结构：前向子图不应包含反向相关节点
       for node in gm.graph.nodes:
           if node.op == "call_function":
               assert "grad" not in node.name, \
                   f"前向图中发现了反向节点: {node.name}"
       # 返回 eager 模式的可调用对象
       return gm.forward

   def debug_bw_compiler(gm, example_inputs):
       """自定义反向编译器：打印图结构"""
       print("=== 反向子图 ===")
       gm.graph.print_tabular()
       return gm.forward

   def fn(x, y):
       return (x * y).sum()

   # 直接调用 aot_function，传入调试编译器
   compiled_fn = aot_function(
       fn,
       fw_compiler=debug_fw_compiler,
       bw_compiler=debug_bw_compiler,
   )

   x = torch.randn(3, requires_grad=True)
   y = torch.randn(3)
   result = compiled_fn(x, y)
   result.backward()

``gm.graph.print_tabular()`` 的输出格式是表格形式，包含每个节点的操作类型、目标、输入等信息，比 ``print(gm.graph)`` 更易读：

.. code-block:: text

   opcode         name           target                  args                   kwargs
   -------------  -------------  ----------------------  ---------------------  --------
   placeholder    primals_1      primals_1               ()                     {}
   placeholder    primals_2      primals_2               ()                     {}
   call_function  mul            aten.mul.Tensor         (primals_1, primals_2) {}
   call_function  sum_1          aten.sum.default        (mul,)                 {}
   output         output         output                  (sum_1, mul)           {}

这个表格比原始的 ``print(gm.graph)`` 输出多了列对齐和操作类型标注，更适合快速浏览图结构。


调试图分区决策
======================

图分区是 AOTAutograd 的核心步骤之一，也是最容易引入 bug 的环节。我们需要确保分区的边界是正确的——前向保存了反向需要且计算代价足够高的 tensor，而反向也正确地接收了这些 tensor。

检查 saved tensors

分区决策最直接的体现是前向子图的输出。前向子图返回的内容中，除了最终的 loss 外，其余都是供反向使用的 saved tensors。

可以通过 ``TORCH_LOGS="aot_partition"`` 查看分区细节：

.. code-block:: bash

   TORCH_LOGS="aot_partition" python -c "
   import torch

   class Model(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.w1 = torch.nn.Parameter(torch.randn(64, 64))
           self.w2 = torch.nn.Parameter(torch.randn(64, 64))

       def forward(self, x):
           h = torch.relu(x @ self.w1)
           return (h @ self.w2).sum()

   model = Model()
   compiled = torch.compile(model, fullgraph=True)
   x = torch.randn(32, 64)
   out = compiled(x)
   out.backward()
   "

日志中会输出类似以下的内容：

.. code-block:: text

   [INFO] 分区策略: min_cut_rematerialization_partition
   [INFO] 联合图节点数: 15
   [INFO] 前向子图输出:
       - output (loss)
       - relu_out (保存供反向使用)
       - mm_result (重计算，不在前向保存)
   [INFO] 反向子图的 placeholder:
       - tangents_1 (反向输入梯度)
       - relu_out (从保存中读取)
   [INFO] 决策: 保存 relu 输出，重计算 mm 结果

通过观察前向子图的输出列表，可以判断哪些 tensor 被保存了：所有不是最终 loss 的输出都是 saved tensors。

使用自定义分区函数

如果默认的分区器（朴素分区或 min-cut 分区）行为不符合预期，可以编写自定义分区函数来调试：

.. code-block:: python

   import torch
   from torch._functorch.partitioners import default_partition

   def debug_partition(joint_graph, joint_inputs, num_fwd_outputs):
       """调试分区器：打印关键信息后调用默认分区"""
       print(f"联合图节点数: {len(joint_graph.nodes)}")
       print(f"联合输入数: {len(joint_inputs)}")
       print(f"前向输出数: {num_fwd_outputs}")

       # 调用默认分区器
       fwd_gm, bwd_gm = default_partition(
           joint_graph, joint_inputs, num_fwd_outputs
       )

       print(f"前向子图节点数: {len(fwd_gm.nodes)}")
       print(f"反向子图节点数: {len(bwd_gm.nodes)}")
       return fwd_gm, bwd_gm

   # 通过 aot_config 传入自定义分区器
   from torch._functorch.aot_autograd import aot_config

   # 更常用的方法：使用 torch.compile 的 backend 参数
   def debug_backend(gm, example_inputs):
       from torch._functorch import aot_function
       from torch._functorch.partitioners import default_partition

       def comp_fn(gm, inputs):
           return gm

       return aot_function(
           gm, fw_compiler=comp_fn, bw_compiler=comp_fn,
           partition_fn=debug_partition,
       )

   fn = lambda x, y: (x * y).sum()
   compiled_fn = torch.compile(fn, backend=debug_backend, fullgraph=True)

在 ``partition_fn`` 参数中传入自定义分区函数，可以在分区前后插入断点或日志，精确观察分区器的决策过程。

分区决策的流程

.. mermaid::

   graph TD
       Joint["联合图（Joint Graph）"] --> Classify["节点分类"]
       Classify -->|"分析节点间的数据依赖"| Analyze["计算每个节点的<br/>reuse/cost 比"]
       Analyze --> Decision{"执行哪种分区?"}
       Decision -->|"默认（朴素）"| Naive["朴素分区<br/>保存所有被反向引用的 tensor"]
       Decision -->|"min-cut"| MinCut["min-cut 重计算分区<br/>计算最小割"]
       Decision -->|"自定义"| Custom["自定义分区函数"]
       Naive --> FwdOut["确定前向输出集合"]
       MinCut --> FwdOut
       Custom --> FwdOut
       FwdOut --> Split["分割 out graph"]
       Split --> Validate{"验证分区正确性"}
       Validate -->|"前向/反向输入输出匹配"| Done["输出前向子图 + 反向子图"]
       Validate -->|"不匹配"| Error["抛出异常"]

这个流程展示了分区器从接收到联合图到输出子图的全过程。理解这个流程有助于定位分区环节的问题。

.. tip::

   如果怀疑 min-cut 分区导致了精度问题，可以临时强制使用默认（朴素）分区器来对比结果：在 ``aot_config`` 中设置 ``partition_fn="default"`` 。如果朴素分区下结果正确而 min-cut 分区下错误，说明 min-cut 的重计算决策引入了精度损失，需要进一步查看哪些操作被重计算了。


常见问题与排查
=========================

AOTAutograd 编译失败

这是最常见的错误类型之一，表现为 ``torch.compile`` 抛出类似 ``RuntimeError: AOTAutograd failed to trace`` 的错误信息。

**常见原因 1：图中包含不可追踪的操作**

AOTAutograd 基于 ``make_fx`` 的 proxy tensor 系统追踪，遇到以下情况会失败：

- 使用了 ``torch.Tensor.numpy()`` 或 ``.item()`` 等将 tensor 转为标量的操作
- 使用了 ``if tensor > 0`` 这样的数据依赖控制流
- 使用了 ``torch.einsum`` 的某些非标准模式

排查方法：

.. code-block:: python

   import torch

   def problematic_fn(x):
       # AOTAutograd 无法追踪 .item() 操作
       threshold = x[0].item()  # 这会抛出异常
       return x * threshold

   # 先用 eager 模式测试，再开启编译
   # 如果 eager 正常而编译失败，基本可定位为 AOTAutograd 问题
   x = torch.randn(3, requires_grad=True)

   try:
       compiled_fn = torch.compile(problematic_fn, fullgraph=True)
       compiled_fn(x)
   except Exception as e:
       print(f"AOTAutograd 追踪失败: {e}")

**常见原因 2：不支持高阶 autograd**

AOTAutograd 默认不支持"在 ``backward()`` 中再次求导"的场景。如果你的模型自定义了 ``backward`` 且其中包含了需要梯度的操作，AOTAutograd 可能失败。

.. warning::

   如果你的模型使用了 ``create_graph=True`` 的 ``torch.autograd.grad`` ，AOTAutograd 不仅需要追踪前向 + 反向，还要追踪"反向的反向"（即二阶梯度）。这种情况超出了 AOTAutograd 的默认能力范围，需要考虑使用 ``torch.compile`` 的 ``dynamic=False`` 或者退回到 eager 模式。

Joint graph 过于庞大

当模型非常深，或者用了大量高层算子（如 ``torch.linalg.*`` 、 ``torch.fft.*`` ）时，joint graph 的节点数可能达到数千甚至上万。这不仅拖慢编译速度，还可能耗尽内存。

**处理方法：**

.. code-block:: python

   import torch

   # 1. 减少 joint graph 的大小：减小输入 batch size
   #    较小的输入 → 较少的分解 → 较少的节点
   x_small = torch.randn(1, 64, requires_grad=True)

   # 2. 减少 fusion 的粒度：通过 torch._dynamo.config 控制
   torch._dynamo.config.suppress_errors = True  # 跳过无法编译的子图

   # 3. 在分解层面控制：跳过某些算子的分解
   #    通过自定义 decomposition 表，保留某些高层算子
   from torch._inductor.decomposition import select_decomp_table
   decomps = select_decomp_table()
   # 移除某些分解来减少图膨胀
   # del decomps[aten.native_layer_norm]  # 示例

   # 4. 使用 partitioner 的 min-cut 策略，让分区器处理"保存 vs 重计算"
   #    从而减少反向子图的节点（某些节点被重计算而非保存）

   # 5. 检查是否无意中 trace 了不需要的路径
   #    使用 torch._dynamo.exc 捕获异常详细日志
   from torch._dynamo.exc import format_error_msg

In-place 操作导致 joint trace 失败

in-place 操作在 eager 模式下可以正常工作，但在 AOTAutograd 的 joint trace 中可能导致问题。这是因为 functionalization（第 4.4 节）将 in-place 操作转换为 out-of-place 操作，但这个转换在某些边界情况下会失败。

典型场景：

.. code-block:: python

   import torch

   def fn_with_inplace(x):
       # AOTAutograd 会通过 functionalization 将下面的 add_ 转换为 add
       # 但如果 x 在后续操作中被多次使用，转换可能复杂化
       x = x.clone()  # 确保有独立的存储
       x.add_(1.0)    # in-place 操作
       return x.sum()

   # 如果 functionalization 支持这个操作，它会被转换为：
   # x = x.clone()
   # x = x + 1.0
   # return x.sum()

   compiled_fn = torch.compile(fn_with_inplace, fullgraph=True)
   x = torch.randn(3, requires_grad=True)
   result = compiled_fn(x)
   result.backward()

排查 in-place 问题的建议：

1.**先用 ``backend="aot_eager"`` 测试**：如果 ``aot_eager`` 后端成功但 Inductor 后端失败，问题可能不在 AOTAutograd 而是在 Inductor 对 functionalization 结果的处理上。
2.**观察 functionalization 日志**：使用 ``TORCH_LOGS="aot_functionalization"`` 查看 functionalization 的每一步转换。
3.**分步排查**：在函数开始处添加 ``.clone()`` 确保输入不被其他操作共享，排除别名问题。

Eager 与编译后的梯度不匹配

这是最隐蔽的问题：模型在 eager 模式下 loss 正常下降，但编译后梯度出现 NaN 或不收敛。

**排查步骤：**

.. code-block:: python

   import torch

   def compare_gradients(fn, x, y):
       """对比 eager 和 compiled 模式的梯度"""

       # Eager 模式
       x_eager = x.clone().detach().requires_grad_(True)
       y_eager = y.clone().detach()
       loss_eager = fn(x_eager, y_eager)
       loss_eager.backward()
       grad_eager = x_eager.grad.clone()

       # Compiled 模式
       x_comp = x.clone().detach().requires_grad_(True)
       y_comp = y.clone().detach()
       compiled_fn = torch.compile(fn, fullgraph=True)
       loss_comp = compiled_fn(x_comp, y_comp)
       loss_comp.backward()
       grad_comp = x_comp.grad.clone()

       # 对比
       diff = (grad_eager - grad_comp).abs().max()
       print(f"梯度最大差异: {diff:.6e}")

       if diff > 1e-5:
           print("=== 梯度不匹配！===")
           print(f"Eager grad[:5]: {grad_eager.flatten()[:5]}")
           print(f"Compiled grad[:5]: {grad_comp.flatten()[:5]}")
       else:
           print("梯度一致")

       return diff

   def fn(x, y):
       return (x.cos() * y.sin()).sum()

   x = torch.randn(10, requires_grad=True)
   y = torch.randn(10)
   compare_gradients(fn, x, y)

如果发现梯度不匹配，使用 ``TORCH_LOGS="+aot"`` 观察分区后的前向/反向子图，与 eager 模式下的 autograd 图对比，定位是哪个环节的图不同。

.. tip::

   一个实用的调试策略是"二分法"：如果编译后的梯度在某个中间节点开始出现 NaN，在该节点前后分别禁用 ``torch.compile`` 的子图捕获（使用 ``torch._dynamo.disable`` ），逐步缩小问题范围。


AOTAutograd 的生命周期钩子
=======================================

AOTAutograd 提供了一组生命周期钩子，允许用户在编译流程的关键节点插入自定义操作。这些钩子主要用于调试和性能分析。

aot_graphs 钩子

``aot_graphs`` 是 AOTAutograd 在生成联合图后触发的钩子。通过设置这个钩子，可以截获联合图并对其进行分析或修改：

.. code-block:: python

   import torch
   from torch._functorch import aot_autograd

   def joint_graph_hook(joint_graph, inputs, num_fwd_outputs):
       """联合图钩子：在分区前检查 joint graph"""
       print(f"联合图节点数: {len(joint_graph.nodes)}")
       print(f"联合输入个数: {len(inputs)}")

       # 分析图的拓扑结构
       fwd_nodes = []
       bwd_nodes = []
       for node in joint_graph.nodes:
           tag = node.meta.get("partitioner_tag", "unknown")
           if tag == "is_forward":
               fwd_nodes.append(node)
           elif tag == "is_backward":
               bwd_nodes.append(node)

       print(f"前向节点: {len(fwd_nodes)}, 反向节点: {len(bwd_nodes)}")

       # 找出所有被保存的中间 tensor
       saved_tensors = [
           n for n in joint_graph.nodes
           if n.op == "call_function" and n.users
           and any(u.meta.get("partitioner_tag") == "is_backward" for u in n.users)
       ]
       print(f"被反向引用的前向节点数: {len(saved_tensors)}")
       return joint_graph  # 返回（可修改后的）图

   # 注册钩子
   aot_autograd.register_hook("aot_graphs", joint_graph_hook)

   def fn(x, y):
       return (x * y).sum()

   compiled_fn = torch.compile(fn, fullgraph=True)
   x = torch.randn(3, requires_grad=True)
   y = torch.randn(3)
   result = compiled_fn(x, y)
   result.backward()

partitioner 钩子

``partitioner`` 钩子在图分区完成后触发，接收分区后的前向和反向子图：

.. code-block:: python

   import torch
   from torch._functorch import aot_autograd

   def partition_hook(fwd_gm, bwd_gm):
       """分区钩子：在分区完成后检查子图"""
       print("=== 分区完成 ===")
       print(f"前向子图: {len(fwd_gm.nodes)} 个节点")
       print(f"反向子图: {len(bwd_gm.nodes)} 个节点")

       # 验证前向子图的输出
       fwd_outputs = list(fwd_gm.graph.nodes)[-1].args[0]
       if isinstance(fwd_outputs, (list, tuple)):
           saved = fwd_outputs[1:]  # 第一个输出是 loss，其余是 saved tensors
           print(f"Saved tensors 数量: {len(saved)}")
           for i, t in enumerate(saved):
               print(f"  saved[{i}]: {t.meta.get('val', 'unknown')}")

       # 检查反向子图的输入是否是 saved tensors
       bwd_inputs = [n for n in bwd_gm.graph.nodes if n.op == "placeholder"]
       print(f"反向子图输入（不包括 tangents）: {len(bwd_inputs) - 1}")

       return fwd_gm, bwd_gm

   aot_autograd.register_hook("partitioner", partition_hook)

   def fn(x, y):
       return (x * y).sum()

   compiled_fn = torch.compile(fn, fullgraph=True)
   x = torch.randn(3, requires_grad=True)
   y = torch.randn(3)
   result = compiled_fn(x, y)
   result.backward()

注册自定义钩子的注意事项

.. warning::

   ``register_hook`` 是一个 **全局操作** ，注册的钩子会影响该进程中所有的 AOTAutograd 编译调用。在生产环境部署或进行基准测试之前，务必取消钩子的注册。建议在调试代码块中使用 ``try/finally`` 或上下文管理器模式来管理钩子的生命周期。

.. code-block:: python

   from torch._functorch import aot_autograd

   class DebugHook:
       """上下文管理器风格的调试钩子"""
       def __init__(self):
           self.hooks = []

       def register(self, name, hook_fn):
           aot_autograd.register_hook(name, hook_fn)
           self.hooks.append((name, hook_fn))

       def unregister_all(self):
           for name, hook_fn in self.hooks:
               aot_autograd.unregister_hook(name, hook_fn)
           self.hooks.clear()

       def __enter__(self):
           return self

       def __exit__(self, *args):
           self.unregister_all()

   # 使用方式
   with DebugHook() as dh:
       dh.register("aot_graphs", joint_graph_hook)
       dh.register("partitioner", partition_hook)
       # 在此范围内执行的 torch.compile 会触发上述钩子
       compiled_fn = torch.compile(fn, fullgraph=True)
       result = compiled_fn(x, y)

可用的钩子列表

AOTAutograd 支持的钩子包括以下阶段：

.. list-table::
   :header-rows: 1

   * - 钩子名称
     - 触发时机
     - 回调参数
     - 常见用途
   * - ``aot_graphs``
     - 联合图生成后，分区前
     - ``(joint_graph, inputs, num_fwd_outputs)``
     - 分析图结构、修改图
   * - ``partitioner``
     - 分区完成后
     - ``(fwd_gm, bwd_gm)``
     - 检查分区结果、验证 saved tensors
   * - ``compilation``
     - AOTAutograd 编译完成
     - ``(fwd_gm, bwd_gm, fwd_compiled, bwd_compiled)``
     - 统计编译时间、缓存结果
   * - ``runtime``
     - 运行时调用前
     - ``(fwd_compiled, bwd_compiled, inputs)``
     - 运行时 profiling、输入验证

通过这些钩子，用户可以深入到 AOTAutograd 的编译流水线中，在任意阶段插入自己的分析和调试逻辑——而不需要修改 PyTorch 的源代码。这是 AOTAutograd 设计中最具可观测性的接口。

.. tip::

   钩子函数的参数对象是 FX Graph 的 ``GraphModule`` ，你可以利用 ``torch.fx`` 提供的所有图分析工具——例如 ``torch.fx.passes.shape_prop`` 来传播 shape 信息、 ``torch.fx.symbolic_trace`` 来做子图分析、或者自己写 ``torch.fx.Interpreter`` 来自定义执行语义。这些工具组合使用，可以在不修改源码的前提下对 AOTAutograd 的行为做全面的诊断。


小结
======

- 使用 ``TORCH_LOGS="+aot"`` 可以观察 AOTAutograd 的完整流程，从联合图创建到前向/反向子图输出
- ``torch._functorch.compilers.nop`` 和 ``backend="aot_eager"`` 可以跳过 Inductor，仅验证 AOTAutograd 自身的正确性
- 自定义 ``partition_fn`` 可以深入调试分区决策，检查 saved tensors 的选取
- 梯度不匹配时，使用逐元素对比和 ``TORCH_LOGS`` 定位问题环节
- 生命周期钩子（ ``aot_graphs`` 、 ``partitioner`` 、 ``compilation`` 、 ``runtime`` ）提供了在 AOTAutograd 流水线关键节点插入自定义逻辑的能力
