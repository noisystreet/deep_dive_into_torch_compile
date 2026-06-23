.. _guard-mechanism:

=================
Guard 机制
=================

上一节我们看到，Dynamo 符号执行完成后输出了两样东西：FX Graph 和 Guards。这一节我们来说 Guards——它是保证编译结果正确性的关键。

为什么需要 Guard？
=====================

Dynamo 的编译结果是**针对特定的输入特征**优化的。例如，第一次调用时输入是 ``shape=(32, 784), dtype=float32, device=cuda:0``，Inductor 为这个配置生成的 Triton kernel 假设了这些参数：

- block size 根据 32×784 计算
- 数据类型是 float32，kernel 中使用 ``tl.float32``
- device 是 CUDA，kernel 使用 ``@triton.jit``

如果下一次调用传入的不是 float32 而是 float16，这个 kernel 就不适用——它会读取错误数据类型的值。

**Guard 就是在运行时验证"当前输入是否与编译时一致"的检查条件。** 本质上是把编译时观察到的输入特征序列化为一组断言，在运行时逐条验证。

.. code-block:: text

   编译时观察到的:
       x.shape  = (32, 784)
       x.dtype  = torch.float32
       x.device = cuda:0
       model    = <MyModel at 0x1234>  (ID_MATCH)

   生成的 Guard:
       check x.shape  == (32, 784)
       check x.dtype  == torch.float32
       check x.device == cuda:0
       check model is <MyModel at 0x1234>

   运行时验证:
       全部通过 → 命中缓存，执行编译好的 kernel
       任意一个失败 → 未命中，重新编译

Guard 的类型
================

Dynamo 中有多种不同类型的 guard，分别对应输入特征的不同维度：

.. list-table::
   :header-rows: 1

   * - Guard 类型
     - 检查内容
     - 触发条件
   * - Shape Guard
     - ``x.shape == (32, 784)``
     - Tensor 的形状
   * - DType Guard
     - ``x.dtype == torch.float32``
     - Tensor 的数据类型
   * - Device Guard
     - ``x.device == cuda:0``
     - Tensor 所在的设备
   * - ID_MATCH Guard
     - ``model is <0x1234>``
     - Python 对象的 identity（常用于 nn.Module）
   * - Data Dependent Guard
     - ``x.sum() > 0`` 的符号化表示
     - 依赖于 Tensor 数据的控制流

不同类型的 guard 有不同的开销和精确度。Shape/DType/Device guard 的检查非常快（一次比较即可），ID_MATCH guard 也很快（指针比较）。Data dependent guard 则更复杂——它涉及到符号形状（symbolic shapes）的运行时重验证，见第 3.7 节。

GuardManager 的树形结构
=============================

Guards 不是平铺的列表，而是**树形结构**。Dynamo 使用 ``GuardManager``（定义在 ``pytorch/torch/_dynamo/guards.py``）来组织 guard 检查：

.. code-block:: text

   RootGuardManager
   │
   ├─ TensorGuardManager(x)
   │   ├─ ShapeGuard(shape=(32, 784))
   │   ├─ DTypeGuard(dtype=float32)
   │   └─ DeviceGuard(device=cuda:0)
   │
   ├─ TensorGuardManager(y)
   │   ├─ ShapeGuard(shape=(32, 784))
   │   ├─ DTypeGuard(dtype=float32)
   │   └─ DeviceGuard(device=cuda:0)
   │
   └─ ID_MATCH_Guard(model=<MyModel at 0x1234>)

这种树形结构的优势在于：

1. **短路求值**：如果 ``x`` 的第一个 guard 就失败了，不需要检查 ``y`` 的 guard
2. **快速定位差异**：当 guard 不匹配时，树形结构可以精确指出是哪个输入、哪个属性不一致
3. **增量更新**：树中的单个分支可以被替换，而不需要重建整个 guard

在运行时，``GuardManagerWrapper`` 持有根节点，遍历整棵树执行检查：

.. code-block:: python
   :caption: pytorch/torch/_dynamo/guards.py（简化示意）

   class GuardManagerWrapper:
       def __init__(self):
           self.root = RootGuardManager()

       def check(self, *args):
           """检查所有 guard，全部通过返回 True"""
           return self.root.check(args)

       def finalize(self):
           """冻结 guard 树，优化检查顺序"""
           self.root.finalize()

Guards 是在什么时候生成的？
=================================

Guards 不是在编译完成后一次性生成的，而是在**符号执行过程中逐步积累**的。

当 ``InstructionTranslator`` 在执行字节码时，每遇到一次对 Tensor 属性的访问（如 ``x.shape``、``x.dtype``），它都会记录对应的 guard。积累的 guard 存储在 ``OutputGraph.guards`` 列表中。

.. code-block:: text

   符号执行过程                          Guard 积累
   ──────────────────                   ──────────────────
   LOAD_FAST x
   LOAD_ATTR shape                      ← add ShapeGuard(x, (32, 784))
   ...
   LOAD_FAST model
   LOAD_ATTR forward                    ← add ID_MATCH_Guard(model)
   ...

但这里有一个重要的设计：**guard 不是在访问时立即完整生成的，而是在编译结束时统一优化和萃取**。符号执行过程中收集的是"raw guard"（原始条件），最后通过 ``CheckFunctionManager``（也在 ``guards.py`` 中）将其编译为高效的 guard 检查代码。

CheckFunctionManager 是做什么的？
==========================================

``CheckFunctionManager`` 的职责是对收集到的 raw guards 进行**编译和优化**：

.. code-block:: text

   OutputGraph.guards (raw)
       │
       ▼
   CheckFunctionManager
       │
       ├─ 1. 去重：移除冗余的 guard 条件
       ├─ 2. 合并：将多个 shape guard 合并为一个
       ├─ 3. 排序：将最可能失败的 guard 排在前面
       │     （短路优化）
       ├─ 4. 编译：生成 GuardManager 树
       │
       ▼
   GuardManagerWrapper (可执行的 guard 检查代码)

编译后的 guard 检查代码会被缓存到 code object 的 co_extra 中（第 2.4 节讨论过）。每次函数被调用时，Dynamo 遍历缓存链表，对每个 entry 执行其 ``GuardManagerWrapper.check()``。

Guard 失败的调试
=====================

当 guard 检查失败时，Dynamo 可以通过 ``TORCH_LOGS`` 输出详细的失败原因：

.. code-block:: bash

   TORCH_LOGS="+guards" python train.py

输出会类似：

.. code-block:: text

   [guards] 检查失败: x.shape
   [guards]   - 编译时: (32, 784)
   [guards]   - 运行时: (64, 784)
   [guards] 触发重新编译

这种粒度对于调试动态形状问题非常重要。如果发现训练过程中频繁出现 guard 失败（即频繁重新编译），很可能是因为输入形状变化过于频繁。这时可以考虑开启 ``dynamic=True``，让 Dynamo 使用 symbolic shapes 替代具体数值。

关于 symbolic shapes 的原理，见第 3.7 节；调试与重编译诊断见第 8.5 节。

小结
======

这一节我们介绍了 Dynamo 的 guard 机制——它保证了编译结果对新的输入仍然有效。

- Guard 是**运行时验证条件**，在编译时生成、运行时检查
- Guard 以 **树形结构** 组织，支持短路求值和增量更新
- Guard 在 **符号执行过程中逐步积累**，最后由 ``CheckFunctionManager`` 编译优化
- 不同类型的 guard 对应不同的输入特征维度

下一节我们来看 graph break——当 Dynamo 遇到无法捕获的操作时，它是如何优雅地"断图"的。
