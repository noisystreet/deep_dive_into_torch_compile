.. _minimizer:

==========
Minimizer
==========

Minimizer 是 PyTorch 提供的自动调试工具，用于 **在复杂的编译过程中定位问题的最小复现** 。当 ``torch.compile`` 编译一个模型失败时，Minimizer 可以自动缩小问题范围，找到导致失败的最小子图和输入。

什么情况下使用 Minimizer？
================================

- ``torch.compile`` 编译时报错（而不是运行时）
- 生成的编译结果与 eager 模式结果不一致
- 不确定是哪个操作导致了 graph break 或编译失败

Minimizer 会自动执行二分搜索来定位问题。

基本用法
============

Minimizer 通过环境变量启用：

.. code-block:: bash

   TORCHDYNAMO_REPRO_AFTER="dynamo" python train.py

当编译失败时，它会生成一个可运行的复现脚本，保存在磁盘上：

.. code-block:: text

   Reproduced 后的输出:
   =========================
   复现脚本保存在:
   /tmp/torchinductor_xxx/repro.py
   
   运行方法:
   python /tmp/torchinductor_xxx/repro.py

复现脚本是一个自包含的 Python 文件，可以直接运行以重现问题。

两种模式
============

``TORCHDYNAMO_REPRO_AFTER`` 支持两种模式：

**dynamo 模式** ：在 Dynamo 捕获图之后触发复现。当 Dynamo 本身报错时使用：

.. code-block:: bash

   TORCHDYNAMO_REPRO_AFTER="dynamo" python train.py

**aot 模式** ：在 AOTAutograd 处理之后触发复现。当后端的 lowering 或代码生成报错时使用：

.. code-block:: bash

   TORCHDYNAMO_REPRO_AFTER="aot" python train.py

两种模式的区别在于定位的环节不同。以下决策树可以帮助你选择合适的模式：

.. figure:: /_static/figures/minimizer_decision.svg
   :align: center
   :alt: Minimizer 模式选择与二分搜索
   :figwidth: 90%

   上半部分为 dynamo/aot 模式选择决策树，下半部分为 bisect 算法的迭代过程。

Minimizer 的工作原理
===========================

Minimizer 的核心是一个 **二分搜索（bisect）** 算法：

这个流程图合并在上图中（下半部分的"二分搜索算法"）。

这个过程相当于一个自动化的 ``git bisect`` ，但作用于计算图节点而不是 git 提交。

手动复现
===========

Minimizer 生成的复现脚本格式如下：

.. code-block:: python

   # 由 Minimizer 自动生成的复现脚本
   import torch
   
   # 最小化的输入
   args = [torch.randn(3, 3, device='cuda')]
   kwargs = {}
   
   # 最小化的函数
   def fn(x):
       return torch.sin(torch.cos(x))
   
   # 触发编译
   compiled = torch.compile(fn, fullgraph=True)
   result = compiled(*args, **kwargs)

你可以直接运行这个脚本，或者在此基础上进一步简化。

Minimizer 实战：定位一个真实的编译错误
=============================================

下面通过一个完整的案例，演示 Minimizer 的实际工作流程。

场景描述
----------------

假设我们有一个模型，在 ``torch.compile`` 编译时发生了 graph break，但我们不确定是哪个操作导致的。这个模型包含了 ``sin``、``cos``、``print`` 和自定义函数等操作：

.. code-block:: python

   import torch

   def my_custom_function(x):
       # 这个函数内部包含了不支持的 Python 构造
       result = 0
       for i in range(x.shape[0]):  # Python for 循环会导致 graph break
           result += x[i].item()    # .item() 也不被 dynamo 支持
       return result

   @torch.compile
   def model(x):
       a = torch.sin(x)
       b = torch.cos(a)
       print(f"中间值: {b}")  # print 也会导致 graph break
       c = my_custom_function(b)
       return c

   x = torch.randn(3)
   model(x)

直接运行会看到类似这样的输出——模型确实能运行，但性能很差，而且我们不知道具体哪些部位导致了 graph break。

使用 Minimizer 定位问题
-------------------------------

第一步，启用 Minimizer 的 dynamo 模式：

.. code-block:: bash

   TORCHDYNAMO_REPRO_AFTER="dynamo" python model.py

运行后，Minimizer 会输出类似下面的信息：

.. code-block:: text

   [WARNING] torch._dynamo: Reproducer 已保存到:
   /tmp/torchinductor_xxx/repro.py
   =========================
   最小复现脚本:
   =========================
   ================================================================================
   import torch

   args = [torch.randn(3)]
   kwargs = {}

   def fn(x):
       print(f"中间值: {torch.cos(torch.sin(x))}")
       return None

   compiled = torch.compile(fn, fullgraph=True)
   compiled(*args,**kwargs)
   ================================================================================

.. note::

   注意 Minimizer 自动提取出了最小复现路径：它发现 ``print`` 是导致 graph break 的关键操作，
   并且将输入简化为单个 ``torch.randn(3)`` 。 ``fullgraph=True`` 是 Minimizer 自动添加的，
   用于强制要求完整的图捕获——这样只要有任何 graph break 就会直接报错。

第二步，查看生成的复现脚本，确认 root cause 是 ``print`` 操作。修复方法很简单——移除 ``print`` 或者将其移到编译区域之外。

第三步，修复后重新验证：

.. code-block:: python

   import torch

   @torch.compile
   def model(x):
       a = torch.sin(x)
       b = torch.cos(a)
       # print 已移除，或移到外部
       c = b.sum()
       return c

   x = torch.randn(3)
   model(x)  # 不再有 graph break

完整的调试工作流
------------------------

可以将 Minimizer 的使用整合进标准调试流程：

.. mermaid::

   graph LR
       COMPILE["torch.compile 报错"] --> MINIMIZER["设置 TORCHDYNAMO_REPRO_AFTER"]
       MINIMIZER --> REPRO["Minimizer 生成<br/>复现脚本"]
       REPRO --> ANALYZE["分析脚本<br/>定位 root cause"]
       ANALYZE --> FIX["修复代码"]
       FIX --> VERIFY["重新编译验证"]
       VERIFY -->|"仍有问题"| MINIMIZER
       VERIFY -->|"通过"| DONE["问题解决"]

这个流程的核心优势在于 Minimizer 自动完成了最耗时的部分——将大型模型简化为最小复现代码，让你可以直接聚焦于 root cause 本身。

Minimizer 的限制
======================

- Minimizer 只能定位 **确定性的编译错误**。如果错误是随机出现的（如数据竞争），Minimizer 可能无法稳定复现
- 对于 **性能问题** （编译太慢、运行太慢），Minimizer 不适用——它只用于错误定位
- Minimizer 的二分搜索假设错误是 **单调的**——即子集保留错误性质。如果错误只在特定组合下出现，二分搜索可能失败
