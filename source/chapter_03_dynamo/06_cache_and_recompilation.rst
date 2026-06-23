.. _cache-and-recompilation:

=======================
缓存与重新编译
=======================

第 2.4 节介绍了 torch.compile 的三层缓存架构。这一节我们聚焦第一层——**Dynamo 自身的缓存机制**：code object 上的缓存链表是怎么组织的，什么样的条件会触发重新编译。

code object 缓存链表
==========================

Dynamo 将编译结果缓存到**每个 Python 函数的 code object** 上。具体来说，缓存存储在 ``co_extra`` 字段中——这是 CPython 为每个 code object 预留的一个 ``void*`` 扩展字段，可以存储任意附加数据。

.. code-block:: text

   code object (fn.__code__)
   ┌──────────────────────────────────────┐
   │  co_code:       字节码指令            │
   │  co_consts:     常量池                │
   │  co_filename:   source.py             │
   │  co_name:       fn                     │
   │  ...                                  │
   │  ┌──────────────────────────────────┐ │
   │  │  co_extra: Dynamo 缓存链表        │ │
   │  │                                   │ │
   │  │  ┌───────┐  ┌───────┐  ┌───────┐ │ │
   │  │  │entry1 │→→│entry2 │→→│entry3 │ │ │
   │  │  │       │  │       │  │       │ │ │
   │  │  │ guard │  │ guard │  │ guard │ │ │
   │  │  │ code  │  │ code  │  │ code  │ │ │
   │  │  │ next  │  │ next  │  │ next  │ │ │
   │  │  └───┬───┘  └───┬───┘  └───┬───┘ │ │
   │  │      │          │          │     │ │
   │  │      ▼          ▼          ▼     │ │
   │  │  可执行函数  可执行函数  可执行函数 │ │
   │  └──────────────────────────────────┘ │
   └──────────────────────────────────────┘

每个 cache entry 包含三部分：

.. code-block:: python
   :caption: 缓存条目结构（简化示意）

   @dataclass
   class CacheEntry:
       guard_manager: GuardManagerWrapper  # 运行时验证条件
       compiled_code: Callable              # 编译后的可执行函数
       next: CacheEntry | None             # 链表下一个条目

缓存查找过程
================

当 ``compiled_fn(x, y)`` 被调用时，Dynamo 依次执行以下步骤：

.. code-block:: text

   入口: compiled_fn(x, y)
       │
       ├─ 获取 fn 的 code object
       │
       ├─ 从 co_extra 读取缓存链表头
       │
       ├─ 遍历链表:
       │      current = head
       │      while current is not None:
       │          if current.guard_manager.check(x, y):
       │              return current.compiled_code(x, y)
       │          current = current.next
       │
       ├─ 未命中任何缓存:
       │      ├─ 调用编译流水线（Dynamo → AOTAutograd → Inductor）
       │      ├─ 生成新的 guard_manager + compiled_code
       │      ├─ 在链表头部插入新条目
       │      └─ 执行 compiled_code(x, y)
       │
       └─ 返回结果

这里有一个性能细节：**缓存链表是从头开始遍历的**。最近插入的条目（即最近一次编译的结果）被放在链表头部。这意味着如果输入模式高度稳定（始终是相同的形状），第一次 miss 后后续调用总能一次命中。

缓存大小限制与重新编译
==============================

在第 2.4 节中我们介绍了 Dynamo 的两层缓存限制：

- **recompile_limit** （默认 8）：限制单个 nn.Module 实例的缓存条目数
- **accumulated_recompile_limit** （默认 256）：限制同一个 code object 的总编译次数

当超过这些限制时，Dynamo 会触发 **CacheSizeRelevantForFrame** 逻辑（定义在 ``pytorch/torch/_dynamo/cache_size.py``），有两种可能的处理方式：

1. **如果设置了 ``error_on_recompile=True``** （由 ``FailOnRecompileLimitHit`` 处理）：直接抛出异常，让用户明确知道缓存已满
2. **默认行为**：fallback 到 eager 模式——不再尝试编译新的变体，后续调用直接用 Python 解释器执行

这个 fallback 行为是 Dynamo 安全设计的一部分：它宁愿性能回退到 eager，也不愿意无限地重新编译直到内存耗尽。

但 fallback 到 eager 是一个无声的性能陷阱——用户可能发现"torch.compile 怎么没效果了"而不知道为什么。可以通过日志来观察：

.. code-block:: bash

   TORCH_LOGS="+dynamo" python train.py

如果看到 ``[dynamo] 缓存满，fallback 到 eager`` 之类的输出，说明缓存限制已触发。

重新编译的触发条件
========================

重新编译不仅仅由 guard 失败触发。以下几种情况也会触发重新编译：

.. list-table::
   :header-rows: 1

   * - 触发条件
     - 原因
     - 频率
   * - Guard 失败
     - 输入形状/设备/dtype 变化
     - 常见
   * - 动态形状首次出现
     - 第一次遇到某个形状
     - 中等
   * - 模型代码发生变化
     - 新的图结构需要重新捕获
     - 罕见（通常稳定）
   * - ``torch.compiler.reset()``
     - 手动清空缓存
     - 用户触发

最常见的场景是训练循环中动态 batch size：

.. code-block:: text

   Step 1:  batch size = 32 → 编译 fn(x) → guard: shape=(32, *)
   Step 2:  batch size = 32 → guard 命中
   Step 3:  batch size = 32 → guard 命中
   ...
   Step 10: batch size = 64 → guard 失败 → 重新编译 → guard: shape=(64, *)
   Step 11: batch size = 64 → guard 命中
   ...

如果在训练循环中 batch size 频繁变化（比如数据集的最后一个 batch 余量不一），每次新形状都会触发重新编译。在这种情况下，建议设置 ``dynamic=True`` 或者确保数据加载器的 ``drop_last=True``。

``torch.compiler.reset()`` 的行为
=========================================

``torch.compiler.reset()``（等价于 ``torch._dynamo.reset()``）会清空所有 Dynamo 缓存：

.. code-block:: python

   import torch

   # 编译并执行
   compiled_fn = torch.compile(fn)
   compiled_fn(x, y)  # 第一次：编译，缓存命中

   # 清空所有缓存
   torch.compiler.reset()

   # 重新编译
   compiled_fn(x, y)  # 再次编译（缓存已被清空）

这个函数在基准测试中非常关键。如果不 reset，第二次测试同一段代码时会直接命中缓存，测到的是"纯执行时间"而不是"编译 + 执行时间"。

reset 内部做的事情：

1. 遍历所有已加载的 module，清空它们 code object 的 ``co_extra`` 字段
2. 重置全局缓存计数器
3. 重置 guard 状态

注意：reset 不会影响 Inductor 的磁盘缓存。即使调用了 ``reset()``，磁盘上已经编译好的 ``.so`` 文件仍然存在——下次编译同样的 IRNode 时仍然可以命中。

第 2.4 节用整节的篇幅讨论了三层缓存（Dynamo guard 缓存、AOTAutograd 缓存、Inductor 磁盘缓存），建议结合这一节一起阅读，获得完整的缓存全景。

小结
======

这一节我们聚焦 Dynamo 的缓存机制：

- 缓存以**链表形式**存储在 code object 的 ``co_extra`` 中
- 缓存查找遍历链表，**对每个条目执行 guard 检查**
- 超过缓存上限会 **fallback 到 eager** 模式
- ``torch.compiler.reset()`` **清空内存缓存**，但不会影响磁盘缓存

下一节我们深入 **符号形状** （第 3.7 节）——当输入维度不固定时，Dynamo 如何用 ``ShapeEnv`` 替代具体数值 guard，减少频繁重编译。
