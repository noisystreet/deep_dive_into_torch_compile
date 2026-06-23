.. _bytecode-intro:

========================
CPython 字节码基础
========================

前面两章我们反复提到：Dynamo 在"字节码级别"捕获计算图，这是它区别于 TorchScript（AST 解析）和 FX Graph（符号执行）的核心创新。在进入字节码细节之前，先把 **TorchDynamo 这一层要解决什么问题、为什么选这条路线** 说清楚——否则后面 FakeTensor、Guard、Graph Break 看起来会像一堆互不相关的技巧，而不是同一套设计下的必然产物。

TorchDynamo 的设计目标
==========================

Dynamo 处在编译栈最前端，职责可以用一句话概括：**在不改变用户 Python 代码的前提下，尽可能多地捕获 Tensor 计算，并安全地复用编译结果**。

它要同时对抗三个矛盾：

.. list-table::
   :header-rows: 1

   * - 矛盾
     - TorchScript 的做法
     - Dynamo 的选择
   * - Python 太动态
     - 限制语法子集，用户改写法
     - 字节码级捕获 + graph break
   * - 输入总在变
     - 固定 shape 或手动标注
     - guard 验证 + 符号形状（第 3.4、3.7 节）
   * - 编译结果不能乱用
     - 编译失败即报错
     - guard 失败则重编译；实在不行 fallback eager

**曾考虑过的替代方案**。PyTorch 内部和社区尝试过多种前端：

- **TorchScript（AST）**：要求静态类型、限制 Python 特性；与「research 代码随便写」的文化冲突（第 1.1 节）。
- **``torch.fx.symbolic_trace``**：需要用户提供可 trace 的函数，遇到 ``if x.sum() > 0`` 等数据依赖控制流就失败。
- **PEP 523 帧拦截 + 字节码符号执行**：不碰源码，在 CPython 每次进入函数帧时介入；遇到 ``print``、自定义 C 扩展等无法处理的操作，**断图继续**，而不是整体失败。

Dynamo 选择了第三条路。关键 invariant 是：**graph break 不是 bug，是设计特性** （第 1.1 节、第 3.5 节）。这直接体现了第 2.1 节的「编译器适应 Python」原则。

**三大机制如何协作**。后续几节会分别展开，这里先给出设计层面的分工——它们是一条链，不是三个独立模块：

.. code-block:: text

   字节码符号执行（InstructionTranslator）
       │  问题：如何在不真算的情况下知道代码做了什么？
       ├─ FakeTensor：提供 shape/dtype，不分配显存（第 3.3 节）
       └─ Proxy / VariableTracker：在 FX Graph 里占位（第 3.3 节）
       │
       ▼
   Guard 树（CheckFunctionManager）
       │  问题：编译结果能否用于这次输入？
       └─ 编译期记录条件，运行期 O(1) 检查（第 3.4 节）
       │
       ▼
   缓存 + Graph Break
       │  问题：何时复用、何时重编译、何时放弃编译？
       ├─ code object 缓存链表（第 3.6 节）
       └─ 无法捕获 → 断图，子图分别编译（第 3.5 节）

**代价与边界**。这套设计不是免费的：

- graph break 过多 → kernel launch 碎片化，加速比下降。
- guard 过细 → 形状略变就重编译，编译时间压过收益（第 3.7 节讨论符号形状作为缓解）。
- fallback eager → 用户可能感觉「compile 没生效」，需要日志诊断（第 8 章）。

理解这些 trade-off 后，再读字节码指令和 ``InstructionTranslator`` 的实现，就不会迷失在 opcode 细节里。

这一节我们先补充 CPython 字节码的基础知识，为后续深入 Dynamo 的图捕获机制做准备。

什么是字节码
================

当你写了一段 Python 代码，CPython 并不会直接执行它。它会经过两个步骤：

.. code-block:: text

   Python 源码 → 编译 → 字节码 → 解释执行

                            │
                            ▼
                      ┌──────────────────┐
                      │  字节码指令序列   │
                      │  LOAD_FAST   x   │
                      │  LOAD_ATTR  sin  │
                      │  CALL_FUNCTION   │
                      │  ...            │
                      └──────────────────┘

**字节码（bytecode）** 是 Python 源码编译后的中间表示，类似汇编语言之于 C 语言。它由一条条指令组成，每条指令对应一个编号（opcode）和一个或零个参数（arg）。

看一个具体的例子：

.. code-block:: python

   def fn(x):
       return torch.sin(x)

我们可以用 ``dis`` 模块查看它的字节码：

.. code-block:: python

   import dis

   def fn(x):
       return torch.sin(x)

   dis.dis(fn)

输出：

.. code-block:: text

   0  RESUME                   0
   1  LOAD_GLOBAL              torch
   2  LOAD_ATTR                sin
   3  LOAD_FAST                x
   4  CALL_FUNCTION           1
   5  RETURN_VALUE

这就是 Dynamo 看到的东西。你看到的是 ``torch.sin(x)`` 这一行 Python 代码，Dynamo 看到的是 5 条字节码指令。

字节码指令的结构
====================

每条字节码指令包含两个字段：

.. list-table::
   :header-rows: 1

   * - 字段
     - 含义
     - 例子
   * - opcode（操作码）
     - 指令类型
     - ``LOAD_FAST``, ``CALL_FUNCTION``
   * - arg（参数）
     - 指令的操作对象
     - 变量索引或函数参数个数

CPython 3.13 有大约 200 条指令，但在 Dynamo 的视角中，我们只需要关心几类核心指令：

.. list-table::
   :header-rows: 1

   * - 指令类别
     - 作用
     - 影响范围
   * - ``LOAD_*``
     - 将值压入栈顶
     - 局部变量、全局变量、属性、常量
   * - ``STORE_*``
     - 将栈顶值存入变量
     - 局部变量、全局变量、属性
   * - ``CALL_*``
     - 调用函数
     - ``CALL_FUNCTION``, ``CALL_METHOD``
   * - ``BUILD_*``
     - 构建容器/元组/切片
     - ``BUILD_LIST``, ``BUILD_TUPLE``
   * - ``UNARY_*`` / ``BINARY_*``
     - 一元/二元运算符
     - ``UNARY_NEGATIVE``, ``BINARY_ADD``
   * - ``JUMP_*``
     - 控制流跳转
     - ``JUMP_IF_TRUE_OR_POP``, ``JUMP_ABSOLUTE``
   * - ``RETURN_VALUE``
     - 返回值
     - 函数返回

指令的返回值
----------------

每个 ``CALL_FUNCTION`` 指令执行后，返回值会被压入栈顶，供下一条指令使用。这形成了一个**隐式的数据流依赖**：指令 A 压栈的值，可能被指令 B 出栈消费。Dynamo 的字节码分析器（``bytecode_analysis.py``）利用这个属性来追踪 Tensor 在字节码间的流动——它通过模拟栈的变化，知道 ``torch.sin(x)`` 的结果是 Tensor，而这个 Tensor 又被传给了后续哪个操作。

虚拟机和栈
==============

CPython 解释器是一个**基于栈的虚拟机**。这意味着它不直接操作寄存器，而是通过一个值栈来传递数据。

.. code-block:: text

   # 执行前的栈        # LOAD_GLOBAL torch     # LOAD_ATTR sin
   │           │       │           │            │           │
   │           │  →    │   torch   │       →    │   torch   │
   │           │       │           │            │ torch.sin │
   └───────────┘       └───────────┘            └───────────┘

   # LOAD_FAST x        # CALL_FUNCTION 1       # RETURN_VALUE
   │   torch   │        │           │            │           │
   │ torch.sin │   →    │  sin(x)   │       →    │    (空)   │
   │     x     │        │           │            │           │
   └───────────┘        └───────────┘            └───────────┘

这个过程可以用一个 Python 类比来理解：

.. code-block:: python

   # Python 源码
   torch.sin(x)

   # 栈的行为模拟
   stack = []
   stack.append(torch)       # LOAD_GLOBAL torch
   stack.append(stack[-1].sin)  # LOAD_ATTR sin（弹出 torch，压入 torch.sin）
   stack.append(x)           # LOAD_FAST x
   result = stack[-2](stack[-1])  # CALL_FUNCTION 1（弹出 torch.sin 和 x，压入结果）
   stack.pop()               # RETURN_VALUE（弹出结果并返回）

Dynamo 的 ``InstructionTranslator`` 内部维护了一个 ``stack`` 列表和 ``locals`` 字典，模拟的就是这个过程。当它看到 ``LOAD_FAST x``，它会在 ``locals`` 中查找 ``x`` 对应的跟踪变量（tracked variable）；当它看到 ``CALL_FUNCTION``，它会从栈中弹出函数和参数，在 FX Graph 中插入一个 ``call_function`` 节点，然后将新节点的输出压回栈。

这就是"符号执行"的本质——**用模拟代替真实执行，同时记录操作过程**。

为什么字节码级别捕获更强大
================================

理解了字节码的机制，就能明白 Dynamo 为什么选择在字节码级别捕获：

**精度更高**。AST（TorchScript 的方案）只能看到 ``torch.sin(x)`` 是一个函数调用表达式，但看不到调用过程中属性的解析路径。字节码则将这个过程展开为 ``LOAD_GLOBAL torch → LOAD_ATTR sin → LOAD_FAST x → CALL_FUNCTION``，每一层都看得清清楚楚。Dynamo 可以精确判断 ``torch`` 是模块、``sin`` 是属性、``CALL_FUNCTION`` 的参数只有一个。

**不需要源码**。AST 必须在函数定义后才能捕获它的源码。但字节码在执行帧中随时可用——即使函数是通过 ``exec()`` 动态创建的，或者来自 C 扩展模块，Dynamo 也能捕获它的执行过程。

**天然区分 Tensor 和普通对象**。在执行帧中，Dynamo 可以通过观察指令的操作数类型来判断某个值是不是 Tensor。如果 ``x`` 是 Tensor，``torch.sin(x)`` 会被记录到图里；如果 ``x`` 是 int，则不会。这种区分在 AST 层面几乎不可能做到。

第 3 章的后续内容会在这些字节码基础知识之上，深入 Dynamo 的具体实现——包括字节码分析算法、图捕获过程、guard 机制和 graph break 的处理。
