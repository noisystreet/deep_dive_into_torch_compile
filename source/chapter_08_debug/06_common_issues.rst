.. _common-issues:

============
常见问题
============

.. tip::

   **近 20% 的 Inductor 提交是在修 Bug——这说明踩坑是常态。**
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

1. **浮点精度差异**：Triton 和 CUDA 的归约顺序可能不同，导致微小差异。这是正常的，通常差异在 1e-6 级别。如果差异过大，检查是否误用了 ``tf32`` 或 ``fp16``。

2. **随机数生成差异**：编译后的 dropout 等随机操作可能与 eager 模式顺序不同。设置相同的随机种子：

   .. code-block:: python

      torch.manual_seed(42)
      torch.cuda.manual_seed_all(42)

3. **In-place 操作语义**：功能化（functionalization）可能改变了 in-place 操作的语义。检查是否在编译图中正确处理了 in-place 操作的副作用。

性能比 Eager 还差
====================

**首次运行很慢** （正常）。编译本身有开销。首轮训练通常比 eager 慢，后续轮次会更快。

**Graph break 过多**。如果模型中有大量 graph break，编译后的性能可能不如 eager：

.. code-block:: bash

   TORCH_LOGS="+perf_hints" python train.py

如果看到类似以下的日志，说明 graph break 过多：

.. code-block:: text

   [perf_hints] Graph break 10 次，产生了 11 个子图
   [perf_hints] 建议合并子图以避免 Python 解释器开销

解决方法：

- 使用 ``fullgraph=True`` 强制无 graph break
- 将 graph break 的代码用 ``torch.compiler.disable`` 隔离

**Kernel launch 开销过大**。如果 Inductor 生成了大量小 kernel：

.. code-block:: bash

   TORCH_LOGS="+inductor" python train.py

看日志中生成的 kernel 数量。如果超过 100 个 kernel，而模型本身不大，说明 fusion 不充分。尝试：

.. code-block:: python

   torch._inductor.config.max_fusion_size = 10

**内存不足（OOM）**。编译后可能使用更多显存（因为保存了中间结果用于反向）。尝试：

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
