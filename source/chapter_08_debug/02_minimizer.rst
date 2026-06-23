.. _minimizer:

==========
Minimizer
==========

Minimizer 是 PyTorch 提供的自动调试工具，用于**在复杂的编译过程中定位问题的最小复现**。当 ``torch.compile`` 编译一个模型失败时，Minimizer 可以自动缩小问题范围，找到导致失败的最小子图和输入。

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

**dynamo 模式**：在 Dynamo 捕获图之后触发复现。当 Dynamo 本身报错时使用：

.. code-block:: bash

   TORCHDYNAMO_REPRO_AFTER="dynamo" python train.py

**aot 模式**：在 AOTAutograd 处理之后触发复现。当后端的 lowering 或代码生成报错时使用：

.. code-block:: bash

   TORCHDYNAMO_REPRO_AFTER="aot" python train.py

两种模式的区别在于定位的环节不同：

.. code-block:: text

   dynamo 模式: 错误发生在 Dynamo 图捕获阶段？
        ├─ 是 → 快速定位到错误的 Python 代码
        └─ 否 → 尝试 aot 模式

   aot 模式: 错误发生在 Inductor 编译阶段？
        ├─ 是 → 定位到导致编译失败的 FX 子图
        └─ 否 → 可能是后端问题

Minimizer 的工作原理
===========================

Minimizer 的核心是一个**二分搜索（bisect）**算法：

.. code-block:: text

   输入: 完整的 FX Graph（N 个节点）

   Step 1: 将图一分为二
           前半部分：用 eager 模式执行
           后半部分：用 compiled 模式执行
           测试是否有错误

   Step 2: 根据结果缩小范围
           如果错误消失 → 错误在后半部分
           如果错误仍在 → 错误在前半部分

   Step 3: 继续二分，直到找到单个有问题的节点

   Step 4: 输出最小复现代码

这个过程相当于一个自动化的 ``git bisect``，但作用于计算图节点而不是 git 提交。

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

Minimizer 的限制
======================

- Minimizer 只能定位**确定性的编译错误**。如果错误是随机出现的（如数据竞争），Minimizer 可能无法稳定复现
- 对于**性能问题**（编译太慢、运行太慢），Minimizer 不适用——它只用于错误定位
- Minimizer 的二分搜索假设错误是**单调的**——即子集保留错误性质。如果错误只在特定组合下出现，二分搜索可能失败
