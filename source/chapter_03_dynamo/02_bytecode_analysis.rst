.. _bytecode-analysis:

==================
字节码分析
==================

.. note::

   **CPython 字节码有 200+ 条指令，Dynamo 只需要处理其中约 30 条。 **
   大多数指令只与 Python 运行时相关（如 ``STORE_GLOBAL`` 、 ``DELETE_ATTR`` ）而不涉及 Tensor 操作。Dynamo 的策略是** 忽略无关指令，只处理涉及 Tensor 的调用 **。这类似于"鹰眼"——不分析代码中的每一行，只关注与计算相关的部分。在实际的模型中，Dynamo 通常能成功捕获 95% 以上的操作，剩下的 5% 通过 graph break 优雅回退。

上一节我们了解了 CPython 字节码的基本结构——指令、栈、值传递。这一节我们来看 Dynamo 具体是怎么分析这些字节码，从中识别出 Tensor 操作并构建 FX Graph 的。

Dynamo 的字节码分析器位于 ``pytorch/torch/_dynamo/bytecode_analysis.py`` ，它与符号执行引擎 ``InstructionTranslator`` （在 ``symbolic_convert.py`` 中）配合工作。字节码分析器负责静态分析（不执行代码），而 InstructionTranslator 负责动态模拟（"假装执行"的同时记录图）。

为什么要做字节码分析？
==========================

在 Dynamo 开始符号执行之前，它需要对函数帧的字节码做一次** 预分析 **。原因有二：

** 检测控制流和 liveness**。字节码分析器会扫描所有指令，识别出哪些变量在哪些指令之后不再被使用（dead code），哪些跳转是循环的回边（backedge）。这些信息帮助 Dynamo 在图捕获期间做出更好的决策——比如遇到 backedge 时知道这是个循环，而不是简单的条件分支。

** 优化转换策略 **。有些字节码模式可以被简化或消除。例如，连续的 ``LOAD_FAST`` + ``STORE_FAST`` 这种"搬移"操作在很多场景下可以被优化掉。分析器输出的是经过简化后的指令序列，减少了后续符号执行需要处理的工作量。

分析器的核心功能
======================

``bytecode_analysis.py`` 实现了几个关键的分析 Pass：

.. code-block:: text

   bytecode_analysis.py
   ├── remove_dead_code()       # 删除不可达代码
   ├── remove_pointless_jumps() # 删除无意义跳转
   ├── get_jump_targets()       # 提取所有跳转目标
   └── stack_analysis()         # 栈深度分析

移除死代码（remove_dead_code）
----------------------------------------

死代码是指永远不会被执行到的指令。例如：

.. code-block:: python

   def fn(x):
       return x.sin()
       x.cos()  # ← 永远不会执行

在字节码层面， ``x.cos()`` 对应的指令序列在 ``RETURN_VALUE`` 之后，永远不会被执行到。 ``remove_dead_code`` 会将这些指令从指令列表中移除。

不过在实际的 PyTorch 模型中，死代码很少见（编译器优化过的代码或者自动生成的代码中可能出现）。这个 Pass 更多的作用是** 清理输入 **，确保后续处理不会因为奇怪的字节码而出错。

分析器与 InstructionTranslator 的分工
===========================================

分析器和 InstructionTranslator 的分工可以概括为：

.. code-block:: text

   Python 帧到达
       │
       ▼
   ┌─────────────────────────────────────┐
   │   字节码分析器（静态分析）           │
   │   bytecode_analysis.py              │
   │                                     │
   │   • 移除死代码                       │
   │   • 移除无意义跳转                   │
   │   • 提取跳转目标                     │
   │   • 分析栈深度                       │
   │                                     │
   │   输出：清理后的指令列表              │
   └──────────────┬──────────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────────┐
   │   InstructionTranslator（动态模拟）  │
   │   symbolic_convert.py               │
   │                                     │
   │   • 逐条执行清理后的指令             │
   │   • 模拟栈和局部变量                  │
   │   • 遇到 Tensor 操作 → 记录到 FX     │
   │   • 遇到非 Tensor 操作 → graph break │
   │                                     │
   │   输出：FX Graph + Guards            │
   └─────────────────────────────────────┘

InstructionTranslator 如何逐条模拟执行、如何通过 ``call_function`` 派发、控制流与 graph break 如何嵌入 ``step()`` 循环——这些是第 3.3 节的主题。这里只保留** 静态分析 → 动态模拟** 的分工边界。

从帧到 IT 的入口
===================

当一个 Python 帧被 Dynamo 拦截后， ``convert_frame.py`` 中的 ``convert_frame_assert`` 大致做：

.. code-block:: text

   convert_frame_assert(frame)
       │
       ├─ 1. 提取字节码 (frame.f_code.co_code)
       │
       ├─ 2. 字节码分析（本节）
       │      bytecode_analysis.py:
       │      • remove_dead_code
       │      • remove_pointless_jumps
       │
       ├─ 3. 创建 InstructionTranslator，传入清理后的指令
       │      （详见第 3.3 节）
       │
       └─ 4. translator.run() → OutputGraph + Guards

静态分析的质量直接影响 IT 的稳定性：死代码、非法跳转若不在此阶段剔除，符号执行阶段会以更难懂的 ``Unsupported`` 失败。下一节我们打开 InstructionTranslator，看符号执行引擎的内部设计与执行细节。
