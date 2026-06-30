.. _torch-compile-debug:

====================
torch.compile debug
====================

PyTorch 提供了 ``torch.compiler.debug`` 上下文管理器，用于生成**结构化的编译调试报告**。它比原始的日志系统更易读，以 HTML 格式呈现完整的编译过程。

基本用法
============

.. code-block:: python

   import torch
   
   @torch.compile
   def fn(x):
       return (torch.sin(x) + torch.cos(x)).sum()

   x = torch.randn(10)
   
   with torch.compiler.debug():
       result = fn(x)

运行这段代码后，会在当前目录生成一个 ``torch_compile_debug`` 目录，里面包含 HTML 格式的调试报告。

调试报告的生成流程如下图所示：

.. mermaid::

   sequenceDiagram
       participant User as 用户代码
       participant DC as torch.compiler.debug()
       participant Dynamo as TorchDynamo
       participant Inductor as TorchInductor
       participant Report as 调试报告生成器
       participant Files as 输出文件

       User->>DC: 进入 debug() 上下文
       DC->>Dynamo: 启用详细记录模式
       DC->>Inductor: 启用详细记录模式
       Dynamo->>Report: 记录 Graph Break 位置
       Dynamo->>Report: 记录 Guard 表达式
       Dynamo->>Report: 记录子图 FX Graph
       Inductor->>Report: 记录 Lowering 过程
       Inductor->>Report: 记录融合决策
       Inductor->>Report: 记录生成的 Kernel 代码
       Report->>Files: 生成 torchdynamo_debug.html
       Report->>Files: 生成 inductor.html
       Report->>Files: 生成 fx_graph_readable.txt
       Report->>Files: 生成 fx_graph_runnable.py
       Report->>Files: 生成 replay.py
       User->>DC: 退出 debug() 上下文
       DC->>Files: 最终写入磁盘

调试报告的目录结构
=========================

.. code-block:: text

   torch_compile_debug/
   └── run_2025-01-01_00-00-00-xxxxxx/
       ├── torchdynamo_debug.html       # 主报告（Dynamo 视角）
       ├── inductor.html                # Inductor 报告
       ├── fx_graph_readable.txt        # 可读的 FX Graph
       ├── fx_graph_runnable.py         # 可运行的 FX Graph
       └── replay.py                    # 编译过程回放脚本

报告的主要部分
===================

**Dynamo 报告** （``torchdynamo_debug.html``）包含了：

1. **Graph Break 摘要**：列出所有触发 graph break 的位置和原因
2. **Guard 列表**：显示生成的所有 guard 表达式
3. **子图列表**：每个子图的 FX Graph 展示
4. **编译统计**：编译时间、节点数、参数大小

**Inductor 报告** （``inductor.html``）包含了：

1. **Lowering 记录**：每个 FX 节点如何降级为 IRNode
2. **融合结果**：哪些 IRNode 被融合在一起
3. **生成的 kernel 列表**：每个 kernel 的 Triton 或 C++ 源代码
4. **性能估算**：每个 kernel 的估算 FLOPs 和内存带宽

读取 FX Graph
==================

调试报告中的 ``fx_graph_readable.txt`` 文件可以快速查看计算图结构：

.. code-block:: text

   // fx_graph_readable.txt 示例
   opcode       name     target              args               kwargs
   --------    ------   --------            ------              ------
   placeholder x        x                   ()                  {}
   call_function sin_1  aten.sin            (x,)                {}
   call_function cos_1  aten.cos            (x,)                {}
   call_function add_1  aten.add            (sin_1, cos_1)      {}
   call_function sum_1  aten.sum            (add_1,)            {}
   output       output  output              (sum_1,)            {}

``fx_graph_runnable.py`` 是一个可以直接运行的 Python 脚本，用于测试图的正确性。

利用调试报告分析 Graph Break
====================================

当模型因 graph break 导致性能下降时，调试报告可以帮助定位原因。以下流程图展示了完整的分析路径：

.. mermaid::

   graph TD
       START["打开 torchdynamo_debug.html"] --> GB_SECTION["查看 Graph Break 摘要<br/>列出所有 graph break 位置"]
       GB_SECTION --> GB_DECIDE{"存在 Graph Break?"}
       GB_DECIDE -->|"是"| GB_ANALYZE["逐条分析每个 Break:<br/>1. 触发位置(文件名:行号)<br/>2. 触发原因<br/>3. 前后子图分界"]
       GB_DECIDE -->|"否"| GUARD_SECTION["查看 Guard 列表<br/>确认 guard 生成是否正确"]
       GB_ANALYZE --> SUBGRAPH["查看子图列表<br/>了解计算图被拆分成几块"]
       SUBGRAPH --> FIX["定位问题代码<br/>进行修复"]
       GUARD_SECTION --> STATS["查看编译统计<br/>编译时间/节点数/参数大小"]
       STATS --> VERIFY["修复后重新生成报告<br/>验证 Graph Break 是否消失"]
       FIX --> VERIFY
       VERIFY --> DONE["性能恢复正常"]

每个 graph break 在报告中都会显示：

.. list-table::
   :header-rows: 1

   * - 信息字段
     - 说明
     - 示例
   * - 触发位置
     - 导致 graph break 的 Python 代码位置
     - ``example.py:10 in forward``
   * - 触发原因
     - Dynamo 无法捕获的具体原因
     - ``Unsupported: call_function print``
   * - 涉及的操作
     - 导致 break 的具体算子或函数
     - ``torch.Tensor.item``
   * - 子图分界
     - 前后子图的节点范围
     - ``Subgraph #0 (3 nodes) → Subgraph #1 (5 nodes)``

调试报告详解
===================

调试报告目录下包含 5 个文件，每个文件提供了编译过程的不同视角。

torchdynamo_debug.html
------------------------------

这是主报告文件，以 HTML 格式呈现 Dynamo 捕获阶段的完整信息。关键部分包括：

**Graph Break 摘要**

报告首页会列出所有 graph break 的位置和原因。每个 break 都是一个可展开的卡片，点击可以查看详细信息。对于没有 graph break 的模型，这里会显示 "No graph breaks detected"。

.. tip::

   如果一个模型显示大量 graph break，建议从第一个 break 开始排查。因为后面的 break 可能是第一个 break 的连锁反应——图被拆开后，后续的融合机会减少，可能导致更多的 break。

**Guard 列表**

显示 Dynamo 为每个子图生成的所有 guard 表达式。Guard 是 Dynamo 用于验证缓存是否有效的条件：

.. code-block:: text

   Guard 1: ___check_type_id(L['x'], 2)        # 检查 x 的类型
   Guard 2: ___check_obj_id(L['x'], 139...)     # 检查 x 的对象 ID
   Guard 3: ___check_size(L['x'], (3, 3))       # 检查 x 的形状

如果 guard 失败，Dynamo 会重新编译对应的子图。频繁的 guard 失败意味着编译缓存无法生效，会导致反复编译的性能开销。

**子图列表**

每个子图都包含完整的 FX Graph 展示。你可以清晰地看到图在何处被切分，以及每个子图包含哪些操作。

**编译统计**

底部区域展示编译性能数据，包括：

- 编译耗时（Dynamo 和 Inductor 分别统计）
- 节点总数和子图数量
- 参数大小和梯度设置
- 缓存命中率

inductor.html
--------------------

Inductor 报告展示了从 FX Graph 到 Triton/C++ 代码的完整 lowering 过程：

**Lowering 记录**

每个 FX 节点是如何被降级（lower）为 Inductor 的 IRNode 的：

.. code-block:: text

   [0]   aten.sin     → Pointwise    sin_1: {x}
   [1]   aten.cos     → Pointwise    cos_1: {x}
   [2]   aten.add     → Pointwise    add_1: {sin_1, cos_1}

每一行左侧显示 FX 节点编号和 ATen 算子名，中间显示 IRNode 类型（Pointwise / Reduction / Template），右侧显示节点间的数据依赖关系。

**融合结果**

Inductor 会尝试将多个相邻的 Pointwise 节点融合为单个 kernel。融合结果部分会显示：

.. code-block:: text

   Fused Scheduler Node #0:
     Nodes: [sin_1, cos_1, add_1]
     Type: Pointwise
     Kernel: triton_poi_fused_add_cos_sin_0

这意味着 ``sin``、``cos``、``add`` 三个操作被融合到了同一个 Triton kernel 中，而不是生成三个独立的 kernel。

**生成的 Kernel 列表**

列出所有最终生成的 kernel，每个 kernel 包含完整的源代码。可以通过查看 kernel 数量来判断融合效率——理想情况下，一个简单的模型应该只生成 1-2 个 kernel。

**性能估算**

对于每个 kernel，Inductor 会估算其计算量和内存带宽：

.. code-block:: text

   Kernel: triton_poi_fused_add_cos_sin_0
   FLOPs: 15
   Memory: 12 bytes (read: 4, write: 8)
   Estimated latency: 0.02 ms

fx_graph_readable.txt
-----------------------------

这个文件以文本表格形式展示 FX Graph，比 HTML 报告中的图结构更易于快速浏览。对于复杂的模型，FX Graph 可能包含数百个节点，文本格式便于使用 ``grep`` 等工具搜索特定操作。

以下是一个包含分支结构的复杂示例：

.. code-block:: text

   // fx_graph_readable.txt — 复杂示例
   opcode       name         target              args                 kwargs
   --------    ------       --------            ------               ------
   placeholder x            x                   ()                   {}
   call_module conv1        Conv2d(...)         (x,)                 {}
   call_module bn1          BatchNorm2d(...)    (conv1,)             {}
   call_function relu_1     aten.relu           (bn1,)               {}
   call_function split_1    aten.split          (relu_1, [16, 16])  {}
   output       output      output              ([split_1],)         {}

fx_graph_runnable.py
----------------------------

这个文件是一个**可直接运行的 Python 脚本**，包含了与原始模型相同的计算图，但不依赖原始模型代码。它的主要用途：

- **隔离测试**：在脱离原始模型代码的环境中复现编译问题
- **对比测试**：比较 eager 模式和 compiled 模式的结果是否一致
- **简化调试**：当原始模型很大时，使用这个简化的脚本来验证修复

.. code-block:: python

   # fx_graph_runnable.py 示例结构
   import torch
   import torch.fx

   # 从 FX Graph 重建的计算图
   class GraphModule(torch.nn.Module):
       def forward(self, x):
           sin_1 = torch.sin(x)
           cos_1 = torch.cos(x)
           add_1 = torch.add(sin_1, cos_1)
           sum_1 = torch.sum(add_1)
           return sum_1

   # 测试代码
   gm = GraphModule()
   result_eager = gm(torch.randn(10))
   result_compiled = torch.compile(gm)(torch.randn(10))
   print(f"结果一致: {torch.allclose(result_eager, result_compiled)}")

.. note::

   默认情况下，``fx_graph_runnable.py`` 生成的模块不包含 ``torch.compile`` 调用。
   你可以手动添加 ``@torch.compile`` 装饰器来测试编译后的行为。

replay.py
--------------

这个文件记录了编译过程的完整上下文，用于**回放调试**：

.. code-block:: python

   # replay.py 示例结构
   import torch
   import torch._dynamo as dynamo

   # 记录编译时的所有配置
   dynamo.config.replay_record_enabled = True

   # 复现编译过程
   def replay():
       # 重新创建输入
       x = torch.randn(10)
       
       # 使用记录时的配置重新编译
       with torch.compiler.debug():
           result = torch.compile(fn)(x)
   
   if __name__ == "__main__":
       replay()

当你在不同版本的 PyTorch 之间迁移或需要向开发者报告问题时，``replay.py`` 可以确保复现环境与原始编译环境一致。

使用调试报告分析融合效率
====================================

调试报告的另一个重要用途是分析 Inductor 的**融合效率**——即 Inductor 能否将多个计算节点合并为单个 kernel，以减少 kernel launch 开销。

判断融合效率的指标
--------------------------

主要看两个关键数字：

1. **FX 节点数**：计算图中的操作总数（``torchdynamo_debug.html`` 中的编译统计）
2. **生成的 Kernel 数**：Inductor 实际生成的 GPU kernel 数量（``inductor.html`` 中的 kernel 列表）

理想情况下，**kernel 数应远小于 FX 节点数**。如果两者接近，说明融合效果不佳。

融合良好的示例
-----------------------

以下是一个融合良好的例子。模型包含多个逐点操作，但 Inductor 将它们融合为单个 kernel：

.. code-block:: python

   @torch.compile
   def well_fused(x):
       # 五个逐点操作
       a = torch.sin(x)
       b = torch.cos(a)
       c = torch.add(b, a)
       d = torch.mul(c, 1.5)
       e = torch.sub(d, 0.5)
       return e

   # FX 节点数: 5 (sin, cos, add, mul, sub)
   # 生成的 Kernel 数: 1 (所有操作被融合为一个 Pointwise kernel)
   # 融合效率: 5/1 = 5x 减少

在 ``inductor.html`` 的融合结果中，你会看到类似：

.. code-block:: text

   Fused Scheduler Node #0:
     Nodes: [sin_1, cos_1, add_1, mul_1, sub_1]
     Type: Pointwise
     Kernel: triton_poi_fused_add_cos_mul_sin_sub_0

融合不佳的示例
-----------------------

以下示例包含一个阻碍融合的操作，导致生成了多个 kernel：

.. code-block:: python

   @torch.compile
   def poorly_fused(x):
       a = torch.sin(x)        # Kernel 1
       b = a.sum()             # Reduction — 无法与 Pointwise 融合
       c = torch.cos(a)        # Kernel 2 (需要等待 b 完成后)
       d = c * b               # Kernel 3
       return d

   # FX 节点数: 4 (sin, sum, cos, mul)
   # 生成的 Kernel 数: 3 (一个 Pointwise + 一个 Reduction + 一个 Pointwise)
   # 融合效率: 4/3 ≈ 1.33x — 几乎没融合

``sum`` 是一个 reduction 操作，它改变了数据的维度结构。由于后续的 ``cos`` 和 ``mul`` 都依赖 ``sum`` 的输出，Inductor 无法将它们与前面的 ``sin`` 融合。

.. seealso::

   关于融合的更多技术细节，请参见第 6 章（Inductor 代码生成）中有关调度器和融合决策的讨论。

调试报告的实际案例分析
====================================

下面通过一个完整的案例，演示从发现问题到修复验证的完整流程。

初始模型
----------------

假设我们有一个简单的 MLP 风格的模型：

.. code-block:: python

   import torch
   import torch.nn as nn

   class MyModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(256, 128)
           self.fc2 = nn.Linear(128, 64)
           self.dropout = nn.Dropout(0.5)  # Dropout 在训练模式下可能导致 graph break
           
       def forward(self, x):
           x = self.fc1(x)
           x = torch.relu(x)
           x = self.dropout(x)  # 这里可能有 graph break
           x = self.fc2(x)
           return x

   model = MyModel()
   x = torch.randn(32, 256)
   
   # 使用调试模式编译
   with torch.compiler.debug():
       result = torch.compile(model)(x)

生成调试报告并分析
------------------------

运行后，在 ``torch_compile_debug/`` 目录下生成报告。打开 ``torchdynamo_debug.html``，定位到 Graph Break 摘要部分，发现类似以下信息：

.. code-block:: text

   Graph Break #1:
     位置: my_model.py:10 in forward → self.dropout(x)
     原因: Unsupported: torch.nn.Dropout.forward (training mode)
     影响: 图在 dropout 处被切分为两个子图
           Subgraph #0: [fc1, relu] → Subgraph #1: [fc2]

问题分析
----------------

Dropout 在训练模式下引入随机性（``torch.rand``），Dynamo 默认会在遇到随机操作时产生 graph break。这是因为 Dynamo 需要保证追踪出的计算图是确定性的。这个问题的严重程度取决于使用场景：

- **推理模式**：Dropout 是恒等映射，不应该有 graph break
- **训练模式**：Dropout 需要随机性，graph break 导致了性能下降

修复方案
----------------

最简单的修复方法是使用 ``torch.no_grad()`` 或设置模型为评估模式：

.. code-block:: python

   # 修复方案 1: 推理模式下使用 eval()
   model.eval()
   with torch.compiler.debug():
       result = torch.compile(model)(x)

   # 修复方案 2: 如果确实需要在训练时编译，替换为确定性实现
   class MyModelFixed(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(256, 128)
           self.fc2 = nn.Linear(128, 64)
           
       def forward(self, x):
           x = self.fc1(x)
           x = torch.relu(x)
           x = self.fc2(x)
           return x  # 在外部添加 dropout

修复验证
----------------

修复后，重新生成调试报告并对比：

.. list-table::
   :header-rows: 1

   * - 指标
     - 修复前
     - 修复后
   * - Graph Break 数量
     - 1
     - 0
   * - 子图数量
     - 2
     - 1
   * - 生成的 Kernel 数
     - 3（每个子图至少 1 个 kernel）
     - 1
   * - 编译时间
     - 较慢（需要编译多个子图）
     - 较快（单图编译）

.. warning::

   上例中的 Graph Break 只影响训练模式。如果你的模型在推理时也有同样的 graph break，
   那可能不是 Dropout 的问题，而是其他操作导致的——此时需要重新检查调试报告，
   从第一个 graph break 开始排查。

使用 ``torch.compiler.reset()``
=====================================

编译缓存可能导致调试信息不完整。在调试前重置缓存：

.. code-block:: python

   torch.compiler.reset()
   
   with torch.compiler.debug():
       result = fn(x)

这样确保每次调试都是一次全新的编译。

``compile_context`` 的更多选项
========================================

除了 ``torch.compiler.debug()``，还可以使用 ``torch._dynamo.config`` 获取更细粒度的控制：

.. code-block:: python

   import torch._dynamo.config as config
   
   # 保存每次编译的 FX Graph
   config.debug_dir_root = "/tmp/torch_debug"
   
   # 写入每个编译步骤的中间状态
   config.replay_record_enabled = True
   config.record_guard_failure = True

这些配置在 ``torch/_dynamo/config.py`` 中定义，提供了比 ``TORCH_LOGS`` 更细粒度的控制。
