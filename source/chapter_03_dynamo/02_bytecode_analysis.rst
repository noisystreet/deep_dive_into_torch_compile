.. _bytecode-analysis:

==================
字节码分析
==================

.. note::

   **CPython 字节码有 200+ 条指令，Dynamo 只需要处理其中约 30 条。**
   大多数指令只与 Python 运行时相关（如 ``STORE_GLOBAL``、``DELETE_ATTR``）而不涉及 Tensor 操作。Dynamo 的策略是**忽略无关指令，只处理涉及 Tensor 的调用**。这类似于"鹰眼"——不分析代码中的每一行，只关注与计算相关的部分。在实际的模型中，Dynamo 通常能成功捕获 95% 以上的操作，剩下的 5% 通过 graph break 优雅回退。

上一节我们了解了 CPython 字节码的基本结构——指令、栈、值传递。这一节我们来看 Dynamo 具体是怎么分析这些字节码，从中识别出 Tensor 操作并构建 FX Graph 的。

Dynamo 的字节码分析器位于 ``pytorch/torch/_dynamo/bytecode_analysis.py``，它与符号执行引擎 ``InstructionTranslator``（在 ``symbolic_convert.py`` 中）配合工作。字节码分析器负责静态分析（不执行代码），而 InstructionTranslator 负责动态模拟（"假装执行"的同时记录图）。

为什么要做字节码分析？
==========================

在 Dynamo 开始符号执行之前，它需要对函数帧的字节码做一次**预分析**。原因有二：

**检测控制流和 liveness**。字节码分析器会扫描所有指令，识别出哪些变量在哪些指令之后不再被使用（dead code），哪些跳转是循环的回边（backedge）。这些信息帮助 Dynamo 在图捕获期间做出更好的决策——比如遇到 backedge 时知道这是个循环，而不是简单的条件分支。

**优化转换策略**。有些字节码模式可以被简化或消除。例如，连续的 ``LOAD_FAST`` + ``STORE_FAST`` 这种"搬移"操作在很多场景下可以被优化掉。分析器输出的是经过简化后的指令序列，减少了后续符号执行需要处理的工作量。

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

在字节码层面，``x.cos()`` 对应的指令序列在 ``RETURN_VALUE`` 之后，永远不会被执行到。``remove_dead_code`` 会将这些指令从指令列表中移除。

不过在实际的 PyTorch 模型中，死代码很少见（编译器优化过的代码或者自动生成的代码中可能出现）。这个 Pass 更多的作用是**清理输入**，确保后续处理不会因为奇怪的字节码而出错。

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

指令调度的核心机制
========================

在实际的符号执行阶段，``InstructionTranslator`` 使用一个**分发表**（dispatch table）来映射字节码指令到处理方法。这个分发表通过 ``BytecodeDispatchTableMeta`` 元类自动构建。

.. code-block:: python
   :caption: pytorch/torch/_dynamo/symbolic_convert.py（简化示意）

   class BytecodeDispatchTableMeta(type):
       """自动为每条字节码指令注册处理方法"""

       def __new__(cls, name, bases, dct):
           dispatch_table = {}
           for key, value in dct.items():
               if hasattr(dis, key):
                   # 如果类方法名匹配字节码指令名，自动注册
                   dispatch_table[getattr(dis, key)] = value
           dct["dispatch_table"] = [dispatch_table]
           return super().__new__(cls, name, bases, dct)


   class InstructionTranslatorBase(metaclass=BytecodeDispatchTableMeta):
       def step(self):
           """执行一条指令"""
           inst = self.instructions[self.instruction_pointer]
           handler = self.dispatch_table[inst.opcode]
           handler(self, inst)
           self.instruction_pointer += 1

这意味着 ``InstructionTranslatorBase`` 的子类中，方法名和字节码指令名一致的会自动成为该指令的 handler。例如 ``CALL_FUNCTION`` 方法自动处理 ``dis.opmap["CALL_FUNCTION"]`` 指令。

.. code-block:: python

   # InstructionTranslatorBase 中的处理方法
   def CALL_FUNCTION(self, inst):
       args = self.popn(inst.argval)  # 从模拟栈弹出 N 个参数
       fn = self.pop()                # 弹出函数对象
       self.call_function(fn, args, {})  # 派发到 VariableTracker

   def LOAD_FAST(self, inst):
       name = inst.argval
       self.push(self.symbolic_locals[name])  # 从模拟局部变量压入栈

   def STORE_FAST(self, inst):
       name = inst.argval
       self.symbolic_locals[name] = self.pop()  # 从栈弹出存入局部变量

这种基于分发表的架构让 Dynamo 对新 Python 版本的适配变得相对容易。每个新的 Python 版本可能引入新的字节码指令（CPython 3.11 引入了 ``CALL_METHOD`` 等），只需在 ``InstructionTranslatorBase`` 中添加对应的方法即可。

实际执行流程
=================

当一个 Python 帧被 Dynamo 拦截后，从字节码分析到构建出 FX Graph 的整体流程如下：

.. code-block:: text

   convert_frame_assert(frame)
       │
       ├─ 1. 提取字节码 (frame.f_code.co_code)
       │
       ├─ 2. 字节码分析
       │      bytecode_analysis.py:
       │      • remove_dead_code
       │      • remove_pointless_jumps
       │
       ├─ 3. 创建 InstructionTranslator
       │      传入清理后的指令列表
       │
       ├─ 4. 符号执行循环
       │      while translator.instruction_pointer < len(instructions):
       │          translator.step()
       │          # step 内部：
       │          #   1. 读取当前指令
       │          #   2. 查分发表找到 handler
       │          #   3. 执行 handler（可能插入 FX node）
       │          #   4. instruction_pointer += 1
       │
       ├─ 5. 构建 OutputGraph
       │      从 translator.output 提取 FX Graph
       │
       └─ 6. 生成 Guards
              从 translator.output 提取 Guard 条件

这就是 Dynamo 图捕获的骨架。下一节我们会进入细节：符号执行引擎具体是如何将 ``torch.sin(x)`` 这样的调用变成 FX Graph 中的一个节点的。
