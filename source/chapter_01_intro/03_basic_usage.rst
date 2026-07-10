.. _basic-usage:

=============
基本用法
=============

上一节我们用了 ``torch.compile(foo, backend="eager", fullgraph=True)`` 这种形式。这一节展开看看 ``torch.compile`` 的各种使用方式。

函数式 vs 装饰器
===================

``torch.compile`` 有两种等价的用法。

**函数式 ** （上一节用的）：

.. code-block:: python

   compiled_fn = torch.compile(fn)
   result = compiled_fn(x)

**装饰器式** ：

.. code-block:: python

   @torch.compile
   def fn(x, y):
       return torch.sin(x) + torch.cos(y)

   result = fn(x, y)  # 自动编译

两种方式完全等价。装饰器式更简洁，适合定义时就确定要编译的函数。函数式更灵活，适合需要条件编译的场景——比如只在推理时编译，训练时不编译。

作用于 nn.Module
====================

``torch.compile`` 同样可以直接作用于 ``nn.Module`` ：

.. code-block:: python

   class MyModel(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.layers = torch.nn.Sequential(
               torch.nn.Linear(784, 256),
               torch.nn.ReLU(),
               torch.nn.Linear(256, 10),
           )

       def forward(self, x):
           return self.layers(x)

   model = MyModel()
   compiled_model = torch.compile(model)
   x = torch.randn(32, 784)
   logits = compiled_model(x)  # 编译 forward

这并不需要 ``model`` 实现任何特殊接口。Dynamo 追踪的是 ``forward`` 方法的执行过程，和追踪普通函数一样。

核心参数
=================

``torch.compile`` 有几个关键参数，理解它们能帮你更好地控制编译行为。

.. list-table::
   :header-rows: 1

   * - 参数
     - 默认值
     - 作用
   * - ``backend``
     - ``"inductor"``
     - 选择编译器后端
   * - ``mode``
     - ``None``
     - 预设优化策略
   * - ``fullgraph``
     - ``False``
     - 是否要求完整图捕获
   * - ``dynamic``
     - ``None``
     - 是否支持动态形状

backend：选择后端
-----------------------

PyTorch 内置了多个后端：

.. code-block:: python

   # eager 后端：不做优化，只运行 FX Graph（调试用）
   compiled_fn = torch.compile(fn, backend="eager")

   # inductor 后端：默认，生成 Triton/C++ 代码
   compiled_fn = torch.compile(fn, backend="inductor")

   # aot_eager 后端：走 AOTAutograd 但不优化（调试用）
   compiled_fn = torch.compile(fn, backend="aot_eager")

通过 ``torch.compiler.list_backends()`` 可以查看所有可用的后端：

.. code-block:: python

   >>> torch.compiler.list_backends()
   ['aot_eager', 'aot_ts_nvfuser', 'eager', 'inductor', 'ipex', 'nnc', ...]

自定义后端的注册接口会在第 9 章介绍。

mode：预设优化策略
------------------------

.. code-block:: python

   # 默认模式：平衡编译时间与性能
   compiled_fn = torch.compile(fn, mode="default")

   # 减少编译时间，适合快速迭代
   compiled_fn = torch.compile(fn, mode="reduce-overhead")

   # 最大优化，编译时间最长
   compiled_fn = torch.compile(fn, mode="max-autotune")

这三个模式的区别在于 Inductor 后端的自动调优（autotuning）强度：

- ``default`` ：使用预配置的 Tiling 和 Heuristic，不做额外搜索
- ``reduce-overhead`` ：减少 kernel launch 开销的优化，但不做 exhaustive search
- ``max-autotune`` ：对每个 kernel 枚举多种配置（block size、num warps 等），选择最快的一种

一个简单的经验：开发阶段用 ``default`` 或 ``reduce-overhead`` ，部署阶段用 ``max-autotune`` 跑一次，把编译结果缓存住。

fullgraph：完整图约束
-------------------------------

.. code-block:: python

   # 如果有 graph break，直接报错
   compiled_fn = torch.compile(fn, fullgraph=True)

在第 1.1 节我们提到，Dynamo 遇到无法捕获的操作时会 graceful 地 graph break。但如果设置了 ``fullgraph=True`` ，有 graph break 时直接抛出异常。这在调试时很有用——可以帮助你找到代码中哪些地方导致了图断裂。

dynamic：动态形状支持
------------------------------

.. code-block:: python

   # 默认：先按静态形状编译，遇到新形状再重新编译
   compiled_fn = torch.compile(fn)

   # 开启动态形状支持：预编译适配多种形状
   compiled_fn = torch.compile(fn, dynamic=True)

默认情况下，Dynamo 假定输入形状是静态的。如果第一次收到 (32, 784) 的张量，它只缓存针对这个形状的编译结果。如果下一次收到 (64, 784)，guard 检查失败，重新编译。

设置 ``dynamic=True`` 后，Dynamo 会在编译时用符号形状（symbolic shapes）代替具体数值，生成的代码可以适配多种尺寸。符号形状的底层机制见第 3.8 节。代价是编译时间更长，且生成的代码可能不如专门为固定形状优化的代码高效。

一个常见的模式是：训练时关闭 dynamic（输入形状通常是固定的 batch size），推理服务中开启 dynamic（应对变长请求）。

编译缓存
============

torch.compile 会缓存编译结果，避免重复编译。缓存的 key 包括：

- 函数 / Module 的身份
- 输入张量的 ``.shape`` 、 ``.dtype`` 、 ``.device``
- 编译参数（ ``mode`` 、 ``dynamic`` 等）

缓存可以通过 ``torch.compiler.reset()`` 清空：

.. code-block:: python

   # 清空所有编译缓存
   torch.compiler.reset()

这在性能测试时很关键——如果不 reset，第二次跑同样的代码时会命中缓存，测到的是执行时间而不是编译时间。

缓存默认放在内存中，进程退出即释放。你也可以通过 ``TORCHINDUCTOR_CACHE_DIR`` 环境变量将缓存持久化到磁盘，这样多次运行可以复用编译结果。

.. code-block:: bash

   TORCHINDUCTOR_CACHE_DIR=/tmp/compile_cache python train.py

小结
======

这一节我们覆盖了 ``torch.compile`` 的几种使用方式：

- **函数式 vs 装饰器式 ** ：灵活选择应用范围
- **作用于 nn.Module** ：直接编译整个模型
- **四大参数 ** ： ``backend`` 、 ``mode`` 、 ``fullgraph`` 、 ``dynamic``
- **编译缓存** ：自动缓存，可清空、可持久化

下一节我们用真实的基准测试来看看 ``torch.compile`` 到底能带来多少性能提升。
