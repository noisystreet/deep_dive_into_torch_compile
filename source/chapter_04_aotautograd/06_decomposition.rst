.. _decomposition:

===========================
算子分解（Decomposition）
===========================

在 AOTAutograd 创建 joint graph 的过程中，有一道重要的预处理工序：**算子分解（Decomposition）** 。它将高层算子（如 ``layer_norm``、``softmax``）展开为基本算子（``mean``、``rsqrt``、``mul`` 等），使得下游组件（Inductor 或其他后端）只需要为基本算子实现 lowering。

什么是 Decomposition？
========================

PyTorch 有许多高层算子（如 ``aten.native_layer_norm``、``aten._softmax``、``aten.convolution``），它们内部由多个基本算子组合而成。Decomposition 就是将这些高层算子展开为基本算子的过程。

例如，``layer_norm`` 的 decomposition 展开了以下操作：

.. code-block:: python

   # aten.native_layer_norm 被分解为
   mean = aten.mean(x, dim=(-1,), keepdim=True)
   var = aten.mean((x - mean) ** 2, dim=(-1,), keepdim=True)
   rstd = aten.rsqrt(var + eps)
   output = (x - mean) * rstd * weight + bias

分解前后的图结构变化：

.. mermaid::

   flowchart LR
       subgraph before["分解前"]
           ln["aten.native_layer_norm(x, weight, bias)"]
       end
       subgraph after["分解后"]
           mean["mean(x, dim=-1)"]
           sub1["x - mean"]
           sq["(x - mean) ** 2"]
           var["mean(sq, dim=-1)"]
           add_eps["var + eps"]
           rstd["rsqrt(add_eps)"]
           sub2["x - mean"]
           normed["sub2 * rstd"]
           scaled["normed * weight"]
           out["scaled + bias"]
           sub1 --> sq
           sub2 --> normed
       end
       before --> after

分解后的图中，7 个基本算子全部是 **pointwise 操作或 reduction 操作**，这些操作在 Inductor 中可以被高效地融合。

为什么要分解？
==================

**降低 lowering 的复杂度** 。Inductor（或其他后端）只需要为几十个基本算子写 lowering 函数，而不是几百个高层算子。每个高层算子写一段 decomposition 的 Python 代码即可，不需要为它单独实现 lowering。

**暴露融合机会** 。分解后，基本算子之间的中间结果对 Scheduler 可见。例如 ``layer_norm`` 分解后，``x - mean`` 这个逐元素操作可以和后面的 ``rsqrt`` 融合。

**自动获得新算子支持** 。当 PyTorch 新增算子时，只需注册 decomposition，所有后端都可以自动编译它——不需要每个后端为它单独写 lowering。

Decomposition 的执行机制
==============================

AOTAutograd 通过 ``decompositions`` 参数接收一个字典，映射需要分解的算子到对应的分解函数。在 joint graph 追踪时，每当遇到字典中的算子，AOTAutograd 就调用分解函数将其展开。

在 ``graph_capture.py`` 的 ``aot_dispatch_autograd_graph`` 中，decomposition 作为参数传入了 ``make_fx``：

.. code-block:: python
   :caption: pytorch/torch/_functorch/_aot_autograd/graph_capture.py（简化示意）

   def aot_dispatch_autograd_graph(flat_fn, flat_args, ..., aot_config):
       # ...
       fx_g, _ = _create_graph_and_save_traced_inputs(
           joint_fn_to_trace,
           updated_joint_inputs,
           updated_joint_inputs_descs,
           aot_config=aot_config,
       )
       # 此时 joint graph 中已经是分解后的基本算子
       # ...

在 ``make_fx`` 内部，当追踪到 ``aten.native_layer_norm`` 时，proxy tensor 系统检查这个算子是否在 decomposition 表中。如果在，则调用对应的分解函数，将 ``layer_norm`` 替换为 ``mean + var + rsqrt`` 的子图。

常见算子的分解展开
=========================

不同高层算子的分解方式各不相同，但都遵循"将复杂算子拆解为基本算子"的原则。以下以 ``softmax`` 和 ``gelu`` 为例展示常见的分解模式。

**Softmax 的分解。** ``aten._softmax`` 被分解为 exp、sum、div 三步：

.. code-block:: python

   # aten._softmax(x, dim) 被分解为
   exp_x = aten.exp(x)
   sum_exp = aten.sum(exp_x, dim, keepdim=True)
   result = aten.div(exp_x, sum_exp)

分解后的计算图：

.. mermaid::

   flowchart LR
       softmax["aten._softmax(x, dim)"]
       exp["aten.exp(x)"]
       sum_exp["aten.sum(exp, dim, keepdim=True)"]
       div["aten.div(exp, sum_exp)"]
       result["输出"]

       softmax --> exp
       exp --> sum_exp
       exp --> div
       sum_exp --> div
       div --> result

分解后，``exp`` 和 ``div`` 是逐元素操作，``sum`` 是 reduction 操作。如果 softmax 的输入是 ``x`` 而 ``x`` 本身也是来自其他逐元素操作的结果，Inductor 可以将它们融合成一个 kernel。

**GELU 的分解。** GELU 激活函数有两种近似形式。精确形式使用 ``erf`` 函数，Tanh 近似形式则展开为多项式：

.. code-block:: python

   # aten.gelu(x, approximate='tanh') 被分解为
   x3 = x * x * x
   inner = 0.044715 * x3
   inner = x * (0.79788456 + inner)  # 0.79788456 = sqrt(2/pi)
   tanh_inner = aten.tanh(inner)
   result = x * (1.0 + tanh_inner) * 0.5

分解后的计算图：

.. mermaid::

   flowchart TD
       gelu["aten.gelu(x, approximate='tanh')"]
       pointwise1["x * x * x"]
       pointwise2["0.044715 * x3"]
       pointwise3["0.79788456 + pw2"]
       pointwise4["x * pw3"]
       tanh["aten.tanh(pw4)"]
       pointwise5["1.0 + tanh"]
       pointwise6["x * pw5 * 0.5"]
       result["输出"]

       gelu --> pointwise1
       pointwise1 --> pointwise2
       pointwise2 --> pointwise3
       pointwise3 --> pointwise4
       pointwise4 --> tanh
       tanh --> pointwise5
       pointwise5 --> pointwise6
       pointwise6 --> result

GELU 分解后全部是逐元素操作，包含一次 ``tanh``。这些操作可以融合成一个 pointwise kernel，不需要额外 kernel launch。

.. tip::

   **分解与融合是对立统一的。** 分解（Decomposition）将大算子拆开暴露中间结果，融合（Fusion）再将相邻的逐元素操作合并。两者看似方向相反，实则目标一致——让编译器能够更灵活地选择融合边界。分解提供了更细粒度的 IR，融合在此基础上做合并。

决定权与执行权的分离
===========================

这是 PyTorch 编译器中一个重要的架构设计：**策略与机制分离（Strategy vs Mechanism）** 。

.. mermaid::

   flowchart LR
       subgraph strategy["策略层（谁决定）"]
           inductor["Inductor
           decomposition.py
           select_decomp_table()"]
           other_backend["FooBackend
           自定义 decompositions"]
       end

       subgraph mechanism["机制层（谁执行）"]
           aot["AOTAutograd
           joint trace
           接受 decompositions 参数"]
           execution["在追踪时遇到目标算子
           调用分解函数
           展开为基本算子子图"]
       end

       subgraph output["输出"]
           decomposed["分解后的基本算子图
           mean, add, mul, exp, ..."]
       end

       inductor -->|"select_decomp_table()"| aot
       other_backend -->|"自定义 decompositions"| aot
       aot --> execution
       execution --> decomposed

   style strategy fill:#e1f5fe
   style mechanism fill:#f3e5f5
   style output fill:#e8f5e9

图中左侧是**策略层**——决定"分解哪些算子"；右侧是**机制层**——执行"如何分解"。

为什么这样拆分？

**不同的后端需要不同的分解策略** 。假设出现一个后端 "FooBackend"，它原生支持 ``aten.native_layer_norm`` （不需要分解）。FooBackend 只需要：

.. code-block:: python

   def foo_compile(gm):
       decompositions = {
           # 不分解 layer_norm，后端原生支持
           aten.elu: elu_decomposition,
       }
       aot_autograd(gm, decompositions=decompositions, ...)

不需要修改 AOTAutograd 一行代码。``decompositions`` 参数是一个**策略注入点** ，每个后端通过它表达自己的分解需求。

**AOTAutograd 保持后端无关**。如果 decomposition 配置硬编码在 AOTAutograd 中，AOTAutograd 就和 Inductor 耦合了——换后端就必须改 AOTAutograd。现在的设计让 AOTAutograd 接受 ``decompositions`` 参数，保持了一个纯净的接口：AOTAutograd 只负责"执行分解"，不关心"分解哪些算子"。

**决定权归消费者** 。谁最终消费分解后的图（Inductor 负责 lowering），谁就应该决定怎么分解。这是编译器设计中的常见模式——类似 LLVM 中每个后端（X86、ARM、NVPTX）各自决定自己的 target lowering 策略，而不是让中间层（LLVM IR）替后端做决定。

Inductor 的 Decomposition 配置
====================================

Inductor 的 decomposition 配置在 ``pytorch/torch/_inductor/decomposition.py`` 中：

.. code-block:: python
   :caption: pytorch/torch/_inductor/decomposition.py（简化示意）

   inductor_decompositions = get_decompositions([
       aten.native_layer_norm,
       aten._softmax,
       aten.elu,
       aten.leaky_relu,
       aten.gelu,
       aten.hardtanh,
       aten.flip,
       aten.arange,
       aten.addmv,
       ...
   ])

   def select_decomp_table():
       decomps = inductor_decompositions.copy()
       if not config.max_autotune:
           # 非 autotune 模式下保留某些高层算子
           # 让 Inductor 调用 cuBLAS/Triton GEMM 模板
           remove_decompositions(decomps, [aten.mm, aten.convolution])
       return decomps

注意 ``aten.mm`` 和 ``aten.convolution`` 在非 autotune 模式下被从 decomposition 表中移除。这是因为矩阵乘法和卷积有高度优化的 cuBLAS/Triton 实现，保持不分解可以让 Inductor 直接生成 ``TemplateBuffer`` 调用这些优化库，而不是用通用 pointwise 逻辑去处理它们。

除了 Inductor 自选的 decomposition 外，PyTorch 还有一个 ``core_aten_decompositions``（在 ``torch/_decomp/__init__.py`` 中），它定义了算子规范层面的标准分解方法，所有编译器后端都可以依赖它们。

整个流水线中的位置
=========================

.. code-block:: text

   compile_fx_inner(gm, ...)
       │
       ├─ pre_grad_passes()            ← Inductor
       │
       ├─ aot_autograd(
       │       gm,
       │       decompositions=select_decomp_table(),  ← Inductor 配置，AOTAutograd 执行
       │   )
       │   输出: 分解后的前向子图 + 反向子图
       │
       ├─ post_grad_passes(fwd_gm)     ← Inductor
       │  post_grad_passes(bwd_gm)
       │
       └─ Lowering → Scheduler → Codegen

小结
======

- **算子分解（Decomposition）**  将高层算子展开为基本算子，降低后端实现成本
- **配置在 Inductor** （``decomposition.py`` / ``select_decomp_table()``），**执行在 AOTAutograd** （joint trace 中的 decompositions 参数）
- 这种 **策略与机制分离**  的设计让 AOTAutograd 保持后端无关，同时让每个后端自主控制分解策略

AOTAutograd 章节的图变换到此结束。Inductor 侧的 FX Passes（``pre_grad_passes`` / ``post_grad_passes``）将在第 5.2 节展开。
