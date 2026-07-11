.. _fx-passes:

=================
FX Passes：图优化
=================

第 4 章完成了 AOTAutograd 侧的图变换（联合求导、分区、decomposition）。从本节开始，我们进入 Inductor 后端——而 **FX Passes** 正是 Inductor 编译流程中、在 lowering 之前对 FX Graph 做优化的第一道工序。

虽然 AOTAutograd 和 FX Passes 在 ``compile_fx_inner`` 中交替出现，但 FX Passes 的代码全部位于 ``torch/_inductor/fx_passes/`` ，属于 Inductor 职责。因此本书将其放在第 5 章而非第 4 章，避免读者在 AOTAutograd 章节中遇到 Inductor 专有逻辑。

为什么要在多个阶段执行 Pass？
======================================

读者第一次看到 ``compile_fx_inner`` 的流水线，常会问： **既然都是改 FX Graph，为什么不把所有 pass 攒到最后、对 lowering 前的那一张图统一跑一遍？**

答案是： **编译过程中间的图「形状」和「语义」在变**——同一张图在 Dynamo 出口、joint graph 内部、分区后的前向/反向子图上，能安全做的优化完全不同。FX Passes 的分阶段设计，本质是 **在正确的抽象层、正确的时机做正确的变换**。

编译过程中图的四次「变形」
----------------------------------

对照 PyTorch v2.12.1 源码，Inductor 侧的 FX Pass 并非只有 pre/post 两档，中间还有 **joint graph** 这一插入点。 ``compile_fx`` 在 `compile_fx.py <https://github.com/pytorch/pytorch/blob/v2.12.1/torch/_inductor/compile_fx.py>`__ 的 ``_compile_fx_main`` 文档字符串里把整条链路概括为四步（2837–2844 行）：

.. code-block:: text

   (1) pre-grad passes
   (2) 构造 fw_compiler / bw_compiler
   (3) aot_autograd：
       (3a) 用 decompositions 追踪 joint graph
       (3b) partition_fn 分区（ **分区前** 跑 joint-graph passes）
       (3c) fw/bw_compiler 各自编译（ **lowering 前** 跑 post-grad passes）
   (4) 组装前向/反向编译结果

用「图形态」重新表述，读者会看到 **四次变形** ：

.. code-block:: text

   阶段 A：Dynamo 出口
       一张前向 FX Graph（可能含子图模块）
       节点可能是 aten.layer_norm、SDPA 等高层算子
       │
       ▼  pre_grad_passes（AOTAutograd 之前）
       │
   阶段 B：AOTAutograd 内部
       joint trace + decomposition → 一张 joint graph
       图膨胀：出现反向节点；高层算子可能被展开
       │
       ▼  joint_graph_passes（分区 **之前** ，仍在 joint 上）
       │
       ▼  partition_fn → 前向子图 + 反向子图
       │
   阶段 C：Inductor 编译入口（_compile_fx_inner）
       各子图经 view_to_reshape、FakeTensorProp 后
       │
       ▼  post_grad_passes × N（前向 + 反向 **分别** 调用）
       │
   阶段 D：Lowering 入口
       基本算子为主的 FX Graph → IRNode
       │
       ▼  Scheduler 融合（第 5.6 节，IR 层）

**关键 invariant** ：pass 只能作用于 **当前已经存在** 的图结构。阶段 A 还没有反向子图；阶段 B 的 joint 上可以做 **跨前/反向** 的常量折叠、 ``pad_mm`` 等，但尚未分区；阶段 C 才在 **已分区、已 functionalize** 的子图上做 conv+relu、SDPA→Flash 等替换；阶段 D 之后 FX 节点语义名消失，只能做 IR 级融合。

**推理路径的差异** ：训练时 ``joint_graph_passes`` 由 Inductor 的 ``partition_fn`` （ ``compile_fx.py`` 2255–2270 行）在 ``min_cut_rematerialization_partition``**之前** 调用；纯推理（ ``is_inference=True`` ）不走分区，改在 ``compile_fx_forward`` （2413–2439 行）里对 **单张前向图** 调用 ``_recursive_joint_graph_passes``——函数名带 joint，但此时图里通常还没有反向节点，pass 规则会按推理语境生效。

分阶段的设计逻辑
--------------------

.. list-table::
   :header-rows: 1
   :widths: 16 20 26 28

   * - 阶段
     - 时机（源码锚点）
     - 此时图长什么样
     - 为什么在这里做
   * - ``pre_grad_passes``
     - ``aot_module_simplified`` 内、joint trace **之前** （ ``aot_autograd.py`` 1161–1220 行；Inductor 回调为 ``run_pre_grad_passes`` ）
     - 单张前向图，Dynamo 刚捕获；IR**尚未** functionalize
     - 减轻 joint trace 负担；在高层算子还在时做 BN folding 等
   * - decomposition + joint trace
     - AOTAutograd **内部** （第 4 章）
     - 一张 joint graph，含前向+反向节点
     - 不是 ``fx_passes/`` 里的 pass，但会 **改变节点集合**
   * - ``joint_graph_passes``
     - ``partition_fn`` **之前** （ ``compile_fx.py`` 2266–2270 行；实现见 ``joint_graph.py`` 619–690 行）
     - 仍是 joint graph（推理则为单张前向图）
     - ``pad_mm`` 、常量折叠、RNG 替换等需 **分区前** 看到完整数据流
   * - ``post_grad_passes``
     - ``_compile_fx_inner`` 内、lowering **之前** （ ``compile_fx.py`` 1338–1369 行）
     - 分区后的前向/反向 **各一张** 子图；IR 已 functionalize
     - conv+relu、SDPA→Flash 等模式在 decomp 后稳定；前反向 **分别** 优化

用第 2.1 节的话说：这是 **阶段专精** 在图优化层的体现——**autograd 负责造图与分区，Inductor FX pass 在造图前、分区前、Lowering 前各收拾一次** 。

为什么不合并成「一次 pass 跑到底」？
------------------------------------------

假设只在 lowering 前跑一次大 pass，会遇到三类硬问题：

**1. Joint trace 成本**

AOTAutograd 要对前向代码做一次 **假反向** 追踪，生成 joint graph。Dynamo 捕获的图若充满 ``x + 0`` 、重复子表达式，joint graph 会 **同比膨胀**。 ``pre_grad_passes`` 在 trace 前做 CSE、常量折叠、恒等替换，是在 **降低 autograd 追踪的输入规模**——这是编译时间优化，不是运行时优化。

**2. 模式匹配的可见性**

许多 ``post_grad`` 规则匹配 **decomposition 之后** 的基本算子组合。例如 ``fuse_attention.py`` 匹配的是 SDPA 展开后的子图形态；若在 decomposition 之前跑，模式对不上。反之， ``pre_grad`` 里的 BN folding 需要在 **高层算子还在** 时识别。 ``joint_graph`` 里的 ``pad_mm`` 则必须在 **分区前** 看到 joint 上的 matmul 链——挪到 post 分区后，部分跨前/反向的 padding 决策信息已丢失。

**3. 前向与反向的不同优化空间**

分区之后， ``post_grad_passes(fwd_gm)`` 与 ``post_grad_passes(bwd_gm)``**各跑一遍** （ ``post_grad.py`` 116–117 行注释）。反向图常有 distinct 模式（重计算节点、梯度累积、FSDP bucketing），与前向共享同一套 pass**函数**，但应用在 **不同图** 上。若在 joint graph 上统一优化再分区，要么规则无法区分前/反向语境，要么需要 partition-aware 规则，复杂度爆炸——这正是 ``joint_graph_passes`` 与 ``post_grad_passes``**分工** 而非合并的原因。

因此流水线是 deliberate 的 **「pre → autograd 变形 → joint → partition → post × N」**，而不是疏忽导致的重复劳动。

源码中的编排入口
--------------------

``compile_fx`` 把 ``run_pre_grad_passes`` 作为回调传给 ``dynamo_common.aot_autograd`` （ ``compile_fx.py`` 3013–3024 行）。Inductor 自己 **不直接** 在 ``_compile_fx_main`` 里调用 pre_grad——时机由 AOTAutograd 的 ``aot_module_simplified`` 决定，且与 **Autograd 缓存** 挂钩：

- **early** ：缓存查找 **之前** 跑 pre_grad（自定义 pass 无 ``uuid()`` 时必须走此路径，否则缓存键无法反映图变化）
- **late** （默认）：缓存未命中 **之后** 再跑，避免缓存命中时重复做 pass

``pre_grad_passes`` 与 ``post_grad_passes`` 都会对 **嵌套子图** 递归处理（ ``_recursive_pre_grad_passes`` / ``_recursive_post_grad_passes`` ，505–581 行），以支持 ``cond`` 、 ``invoke_subgraph`` 等 Higher-Order Op 内的独立子图模块。

进入 ``_compile_fx_inner`` 后，post_grad 之前还有一步 **``view_to_reshape``** （1322–1338 行）：layout 优化可能把 contiguous 张量变成 channels_last，原先合法的 ``view`` 在编译期会失败，因此在 FakeTensorProp 之前统一改成 ``reshape``——这是 **post_grad 的前置卫生步骤** ，不属于 pattern 融合本身。

与 IR 层融合的关系
--------------------

FX Passes 和 Scheduler 融合（第 5.6–5.7 节）是 **互补的两层** ，不是重复：

.. code-block:: text

   FX Passes（图级代数）     Scheduler（IR 级内存/并行）
   ─────────────────────     ───────────────────────────
   conv + relu → 一个算子     两个 Pointwise → 一个 kernel
   SDPA → flash_attention    逐元素链 → 融合读写
   pad_mm 对齐 Tensor Core   决定 tile 与 launch 顺序

Pattern Matcher（第 5.8 节）大多挂在 ``post_grad_passes`` 里，因为它需要 **FX 节点的语义名字** （ ``aten.convolution`` ）。Scheduler 看不到这些名字，只看到 IRNode 类型。第 5.1 节的四层分工在此体现： **FX pass 改「算什么」，Scheduler 改「怎么算、怎么融」** 。

FX Passes 分为 **三个阶段** （若把 decomposition 算作 AOTAutograd 内部变形，则读者常概括为 pre / joint / post）。 ``compile_fx_inner`` 只负责最后一段；前面两段在 ``compile_fx`` → ``aot_autograd`` 路径上完成。

.. code-block:: text

   compile_fx(model_, example_inputs_)
       │
       ├─ aot_autograd(..., pre_grad_passes=run_pre_grad_passes)
       │       │
       │       ├─ run_pre_grad_passes → pre_grad_passes()     ← 阶段 1
       │       │
       │       ├─ joint trace + decomposition                 ← 第 4 章
       │       │
       │       ├─ partition_fn(gm, ...)
       │       │       └─ _recursive_joint_graph_passes()     ← 阶段 2（分区内）
       │       │           └─ joint_graph_passes()
       │       │
       │       ├─ fw_compiler(gm_fwd)  ──► _compile_fx_inner
       │       └─ bw_compiler(gm_bwd)  ──► _compile_fx_inner
       │
       └─ _compile_fx_inner(gm, ...)
               ├─ view_to_reshape(gm)
               ├─ fake_tensor_prop(...)
               ├─ post_grad_passes(gm)   ← 阶段 3（fwd/bwd 各一次）
               └─ Lowering → Scheduler → Codegen

pre_grad_passes
====================

``pre_grad_passes`` 定义在 `pre_grad.py <https://github.com/pytorch/pytorch/blob/v2.12.1/torch/_inductor/fx_passes/pre_grad.py>`__ （286 行起）。经 ``run_pre_grad_passes`` （ ``compile_fx.py`` 2587–2634 行）包装后，作为回调注入 AOTAutograd，在 joint trace**之前** 运行。输入是 Dynamo 捕获的原始 FX Graph，尚未进行自动微分。

源码在函数文档字符串里明确警告（292–302 行）：**grad 之前的 IR 不是 functional、也未 normalization** ，写 pass 更难——必须正确处理 alias/mutation 与各种 arg schema。因此官方建议：能放到 ``post_grad.py`` 或 ``joint_graph.py`` 的规则尽量后移。

**设计目标** ：在 joint trace 发生前 **瘦身**——让 AOTAutograd 追踪更少的节点，生成的 joint graph 更小。这里的优化偏 **结构性、与梯度无关**：

.. code-block:: text

   pre_grad_passes(gm)   # 经 _recursive_pre_grad_passes 递归子图
       │
       ├─ fuse_fx：permute 融合、cat 下沉等（390 行起）
       ├─ PRE_GRAD_PATTERNS：normalization、group_batch_fusion 等
       ├─ efficient_conv_bn_eval、quant_lift_up 等专用 pass
       └─ 末尾 stable_topological_sort + lint + recompile

若 ``config.is_predispatch`` 为真，则走 ``_run_pre_dispatch_passes`` （198 行起）——在 Predispatch ATen IR 上按固定顺序跑另一套 pass 列表，与默认 Dynamo 出口路径分离。

典型场景：Dynamo 捕获的图里常有 Python 语义遗留的恒等操作；若不提前消掉，autograd 会为每个 ``+ 0`` 多追踪一条 backward 边。

joint_graph_passes
======================

``joint_graph_passes`` （`joint_graph.py <https://github.com/pytorch/pytorch/blob/v2.12.1/torch/_inductor/fx_passes/joint_graph.py>`__ 619–690 行）在 **尚未分区** 的 joint graph 上运行（推理时为单张前向图）。Inductor 的 ``partition_fn`` 在调用 ``min_cut_rematerialization_partition``**之前** 先调用 ``_recursive_joint_graph_passes`` （ ``compile_fx.py`` 2266–2270 行）。

**设计目标**：利用 **分区前仍连在一起的** 前向+反向数据流，做 pre/post 都不合适的变换：

.. code-block:: text

   joint_graph_passes(graph)
       │
       ├─ canonicalize_aten_ir_passes     # 必须最先
       ├─ remove_noop_ops
       ├─ constant_fold_uniform_value     # 可选
       ├─ early_patterns（pattern_matcher）
       ├─ auto_chunker（可选）
       ├─ pass_patterns（含 pad_mm 等）
       └─ replace_random_passes           # 非 fallback_random 时

``run_joint_graph_passes_on_hops`` （`graph_compile.py` 764–790 行）对 ``invoke_subgraph`` 等高阶算子的 **内嵌子图** 单独跑 joint pass 再缝回主图——与 ``_recursive_joint_graph_passes`` 对 FX 子模块的递归是同一设计思想的两层实现。

日常阅读源码时，**抓住 pre → autograd/joint → partition → post(fwd) + post(bwd) 这条主线即可**； ``joint_graph.py`` 里的 pass 数量少于 pre/post，但对理解「为何不能全部挪到 post_grad」至关重要。

post_grad_passes
=====================

``post_grad_passes`` （`post_grad.py <https://github.com/pytorch/pytorch/blob/v2.12.1/torch/_inductor/fx_passes/post_grad.py>`__ 114 行起）在 AOTAutograd 分区与 decomposition**之后**、lowering**之前** 运行。文档字符串写明（116–119 行）：**此时的 IR 已经 normalization 且 functionalize**。 ``_compile_fx_inner`` 通过 ``_recursive_post_grad_passes`` 对每个子图调用它（ ``compile_fx.py`` 1369 行）；训练时前向、反向 **各编译一次**，故 post_grad**各跑一遍**。

**设计目标**：在 **基本算子粒度** 上做 **语义级** 替换与布局类优化。此时：

- decomposition 已展开高层算子，pattern 的 **匹配目标稳定**
- 前向/反向已分离，可对 backward 做专门规则（重计算、FSDP 相关）
- 尚未 lowering，改 FX 节点仍比改 IRNode 便宜

.. code-block:: text

   post_grad_passes(gm, is_inference)
       │
       ├─ eliminate_dead_code（config.dce）
       ├─ reorder_for_locality（仅 inference）
       ├─ post_grad_custom_pre_pass
       ├─ remove_profiler_ops
       ├─ group_batch_fusion（pre_grad=False）
       ├─ remove_noop_ops / remove_assert_ops
       ├─ pass_patterns + POST_GRAD_PATTERNS
       ├─ fuse_ddp_communication、bucketing 等分布式 pass
       └─ stable_topological_sort → FakeTensor 增量更新

``is_inference`` 参数会改变 pass 行为（例如 locality 重排、自定义 pass 的分支），与训练路径共享同一入口函数。

关键 FX Pass 文件
======================

这些 pass 的实现分布在 ``pytorch/torch/_inductor/fx_passes/`` 目录中：

.. code-block:: text

   fx_passes/
   ├── pre_grad.py            # autograd 之前：常量折叠、模式替换
   ├── post_grad.py           # lowering 之前：综合优化、CSE、DCE
   ├── fuse_attention.py      # 将 SDPA 匹配为 Flash Attention
   ├── pad_mm.py              # 将非对齐 mm padding 到对齐尺寸
   ├── binary_folding.py      # batchnorm + 后续操作的融合
   ├── joint_graph.py         # joint graph 级别的 pass
   ├── fusion_regions.py      # FX 级别的融合区域规划
   ├── group_batch_fusion.py  # 分组批处理融合
   ├── decompositions         # 分解相关 pass
   └── ...

关于 pattern matching 的具体机制（ ``@register_graph_pattern`` ），我们会在第 5.8 节 Pattern Matcher 中详细讨论。

小结
======

- **分阶段原因**：图经历「前向 → joint/decomp → joint pass → 分区 → 前向+反向子图 → lowering」多次变形，pass 必须在 **对应形态与 IR 约束** 上运行
- **pre_grad_passes** ：joint trace**之前** 瘦身；IR**未**functionalize，写 pass 成本高（ ``pre_grad.py`` 292–302 行）
- **joint_graph_passes** ： **分区之前** 在 joint（或推理前向）图上做 pad_mm、常量折叠等；由 ``partition_fn`` / ``compile_fx_forward`` 触发
- **decomposition** ：在 AOTAutograd**内部** ，改变节点集合（第 4.6 节）
- **post_grad_passes** ：lowering**之前** 对前向/反向 **分别** 做语义级模式匹配；IR 已 functionalize（ ``post_grad.py`` 116–119 行）
- **与 Scheduler 互补** ：FX pass 改「算什么」，Scheduler 在 IR 层做 kernel 融合（第 5.6–5.8 节）
- **子图递归** ：pre/joint/post 三套入口均通过 ``_recursive_*`` 处理嵌套 ``GraphModule`` ，与高阶算子子图 pass （ ``run_joint_graph_passes_on_hops`` ）配套
