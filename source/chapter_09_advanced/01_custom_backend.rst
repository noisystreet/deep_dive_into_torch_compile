.. _custom-backend:

==================
自定义后端
==================

.. note::

   **Inductor 不是唯一的选择——社区已经有了 20+ 个后端。 **
   PyTorch 的 ``backend`` 参数目前支持超过 20 个后端，包括 ``ipex`` （Intel）、 ``xla`` （Google TPU）、 ``tensorrt`` （NVIDIA），以及社区贡献的 ``torch-mlir`` （LLVM 生态）等。其中最具戏剧性的是 ``torch-mlir``——它由几位 LLVM 编译器工程师在业余时间开发，却在某些场景下达到了与 Inductor 相当的性能。这说明"FX Graph 作为中间表示"的设计是成功的：只要你能消费 FX Graph，就能成为 torch.compile 的后端。

torch.compile 支持自定义后端（custom backend），允许用户将编译后的计算图发送到自定义的编译器或运行时。这为实验性的编译器研究、专用硬件加速等场景提供了接口。

什么是后端
==============

在 torch.compile 的架构中，** 后端** 是接收 FX Graph 并返回可调用函数的组件。AOTAutograd 分区后的前向/反向子图被分别发送给后端：

.. code-block:: text

   FX Graph (前向子图)
       │
       ▼
   后端 compile_fn(fx_graph, example_inputs)
       │
       ├─ 分析 FX Graph
       ├─ 生成优化后的代码或调用 eager
       └─ 返回可调用的函数

PyTorch 内置了多个后端：

- ``inductor`` ：默认后端，生成 Triton 或 C++ 代码
- ``eager`` ：直接使用 PyTorch eager 模式执行
- ``aot_eager`` ：走完 AOTAutograd 流程但用 eager 执行
- ``ipex`` ：Intel 扩展

注册自定义后端
====================

通过 ``@torch.compiler.register_backend`` 装饰器注册：

.. code-block:: python

   import torch
   from torch import fx

   @torch.compiler.register_backend
   def my_backend(gm: fx.GraphModule, example_inputs):
       """自定义后端：打印图结构后用 eager 执行"""
       print("FX Graph 节点数:", len(gm.graph.nodes))
       for node in gm.graph.nodes:
           print(f"  {node.op}: {node.target}")
       return gm.forward  # 使用 eager 执行

   @torch.compile(backend="my_backend")
   def fn(x):
       return torch.sin(x) + torch.cos(x)

   result = fn(torch.randn(3))

运行输出：

.. code-block:: text

   FX Graph 节点数: 4
     placeholder: x
     call_function: aten.sin
     call_function: aten.cos
     call_function: aten.add
     output: output

自定义后端必须满足接口：
- 输入： ``fx.GraphModule`` 和 ``example_inputs`` （位置参数列表）
- 输出：一个可调用的函数，接收与原始函数相同的参数

更完整的后端示例
====================

以下是一个更完整的设计——将 FX Graph 序列化为 JSON 并发送到外部编译器：

.. code-block:: python

   import json
   import torch
   import torch.fx as fx

   class ExternalCompiler:
       """模拟外部编译器"""
       def compile(self, graph_json):
           print(f"接收到图: {len(graph_json['nodes'])} 个节点")
           # 这里连接到真实的外部编译器
           return lambda *args: None

   compiler = ExternalCompiler()

   @torch.compiler.register_backend
   def external_backend(gm: fx.GraphModule, example_inputs):
       # 将 FX Graph 序列化为可 JSON 序列化的格式
       graph_json = {"nodes": []}
       for node in gm.graph.nodes:
           graph_json["nodes"].append({
               "name": node.name,
               "op": node.op,
               "target": str(node.target),
               "args": [str(a) for a in node.args],
           })
       
       # 发送到外部编译器
       compiled_fn = compiler.compile(graph_json)
       return compiled_fn

   @torch.compile(backend="external_backend")
   def fn(x):
       return torch.sin(x) + torch.cos(x)

   result = fn(torch.randn(3))

与 AOTAutograd 的交互
==========================

如果后端需要处理 joint graph（即前向和反向未分区的图），可以通过 ``aot_autograd`` 的接口注册：

.. code-block:: python

   from functorch.compile import min_cut_rematerialization_partition

   def my_compiler(gm, example_inputs):
       """处理 AOTAutograd 分区后的子图"""
       print(f"编译子图: {len(gm.graph.nodes)} 个节点")
       return gm.forward

   @torch.compile(backend=my_compiler)
   def fn(x):
       return torch.sin(x) + torch.cos(x)

   result = fn(torch.randn(3))

对于更复杂的场景，可以自定义 AOTAutograd 的分区策略：

.. code-block:: python

   def my_aot_backend(gm, example_inputs):
       # 自定义分区策略
       from functorch.compile import default_partition
       fwd_compiler = my_compiler
       bwd_compiler = my_compiler
       return aot_function(gm, fw_compiler=fwd_compiler, 
                           bw_compiler=bwd_compiler,
                           partition_fn=min_cut_rematerialization_partition)

调试自定义后端
====================

使用 ``TORCH_LOGS`` 观察后端调用：

.. code-block:: bash

   TORCH_LOGS="+dynamo" python custom_backend.py

日志中会显示后端被调用的时机和传入的图。

对自定义后端的限制
========================

- 自定义后端 **没有自动微分功能 ** 。传入后端的图是 AOTAutograd 分区后的，前向子图不包含反向信息。如果后端需要自己处理微分，需要使用 AOTAutograd 级别的接口。
- 自定义后端**不参与算子分解 ** 。传入的图中可能包含高层算子（如 ``layer_norm`` ）。后端需要自己处理这些算子。
- 自定义后端生成的代码**不受 PyTorch 的 guard 保护** 。如果输入形状变化导致重新编译，后端会再次被调用。
