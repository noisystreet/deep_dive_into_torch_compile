.. _instruction-translator:

======================================
InstructionTranslator：符号执行引擎
======================================

第 3.2 节的字节码分析器输出 **清理后的指令列表 ** ；本节进入 Dynamo 的心脏——**InstructionTranslator** （ ``symbolic_convert.py`` ）。它是一份 **CPython 解释器的符号化替身** ：维护模拟栈和局部变量，逐条「执行」字节码，但遇到 Tensor 运算时不真算，而是往 FX Graph 里插节点。

第 3.1 节的设计链（FakeTensor / Guard / Graph Break）最终都落在这个类的 ``run()`` / ``step()`` 循环里。读懂 InstructionTranslator，后面 guard、graph break、符号形状才不会像互不相关的插件。

源码中的类层次
====================

InstructionTranslator 不是孤立存在的，它与 ``convert_frame.py`` 、 ``variables/`` 、 ``output_graph.py`` 分工明确：

.. code-block:: text

   convert_frame.py
   ├── convert_frame_assert()      # 帧级入口：缓存、重试、调用 IT
   └── compile_inner()             # IT 完成后触发后端编译

   symbolic_convert.py
   ├── InstructionTranslatorBase   # 分发表、step()、graph_break()
   └── InstructionTranslator       # 具体帧的符号执行（继承 Base）

   output_graph.py
   └── OutputGraph                 # IT.output：累积 FX Graph + guards

   variables/
   └── *Variable                   # IT 栈上的「值语义」（见第 3.4 节）

**帧级 vs 指令级 ** ：

- ``convert_frame`` 回答：「这个 Python 帧要不要编译？缓存命中了吗？编译失败怎么办？」
- ``InstructionTranslator`` 回答：「这条字节码对 Tensor 计算意味着什么？要不要进图？」

为什么要单独做一个「解释器替身」
======================================

TorchScript 在 AST 层做静态分析； ``torch.fx.symbolic_trace`` 在 Python 调用层做追踪。Dynamo 选择**字节码层 ** ，是因为只有在这里才能**精确复现 CPython 的控制流和数据流 ** ，同时又不执行真实 Tensor 计算。

InstructionTranslator 的设计 invariant 可以概括为三条：

1.**栈与 locals 必须与 CPython 一致 **——否则 ``CALL_FUNCTION`` 弹栈顺序错，图就错。
2.** 所有运行时值都是 VariableTracker**——IT 不直接操作 ``torch.Tensor`` ，只操作包装后的符号值。
3.** 副作用分流**——能进图的进 ``OutputGraph`` ；不能进的 ``graph_break`` ，把控制权交还 eager（第 3.6 节）。

核心状态：模拟器的「内存」
==============================

创建 ``InstructionTranslator`` 时，会初始化一整套模拟 CPython 帧的状态。下表是读源码时最该先认识的字段：

.. list-table::
   :header-rows: 1

   * - 字段
     - 对应 CPython 概念
     - 作用
   * - ``instructions``
     - ``co_code`` 解析结果
     - 经 3.2 节分析器清理后的指令流
   * - ``instruction_pointer``
     - 程序计数器 PC
     - 当前执行到哪条字节码
   * - ``stack``
     - 值栈
     - 元素均为 ``VariableTracker`` 子类
   * - ``symbolic_locals``
     - ``fast locals``
     - 局部变量名 → VariableTracker
   * - ``symbolic_globals``
     - ``f_globals``
     - 全局变量查找
   * - ``output``
     - （无直接对应）
     - ``OutputGraph`` ：FX Graph + guards 的累积容器
   * - ``f_code`` / ``f_locals`` / ``f_globals``
     - 真实帧
     - 编译期读取常量、闭包、真实对象身份

可以把 InstructionTranslator 想象成 **带类型系统的 CPU 模拟器 ** ： ``stack`` 是寄存器栈， ``instruction_pointer`` 是 PC， ``VariableTracker`` 是寄存器里存放的「符号值」。

分发表：一条 opcode 一个 handler
=====================================

IT 不为 200+ 条 CPython 指令各写一套 ad-hoc 逻辑，而是用**分发表 ** 统一调度。 ``BytecodeDispatchTableMeta`` 在类定义时扫描方法名——若与 ``dis`` 模块中的 opcode 名一致，自动注册为 handler：

.. code-block:: python
   :caption: pytorch/torch/_dynamo/symbolic_convert.py（简化示意）

   class BytecodeDispatchTableMeta(type):
       def __new__(cls, name, bases, dct):
           dispatch_table = {}
           for key, value in dct.items():
               if hasattr(dis, key):
                   dispatch_table[getattr(dis, key)] = value
           dct["dispatch_table"] = [dispatch_table]
           return super().__new__(cls, name, bases, dct)


   class InstructionTranslatorBase(metaclass=BytecodeDispatchTableMeta):
       def step(self):
           inst = self.instructions[self.instruction_pointer]
           self.current_instruction = inst
           self.instruction_pointer += 1
           handler = self.dispatch_table[inst.opcode]
           handler(self, inst)

**设计取舍 ** ：CPython 每个版本都可能新增 opcode（如 3.11 的 ``CALL`` 系列）。分发表让适配工作变成**「为缺失的 opcode 补一个同名方法」 ** ，而不是改中央 switch。代价是 ``InstructionTranslatorBase`` 极其庞大——这是用**代码体积 ** 换**版本兼容性与局部可读性** 。

常见 handler 的行为模式：

.. list-table::
   :header-rows: 1

   * - Handler
     - 典型行为
   * - ``LOAD_FAST`` / ``LOAD_GLOBAL``
     - 从 locals/globals 取出 VariableTracker， ``push`` 到栈
   * - ``STORE_FAST``
     - ``pop`` 栈顶，写入 ``symbolic_locals``
   * - ``CALL_FUNCTION`` / ``CALL``
     - ``popn`` 参数和 callable，转 ``call_function``
   * - ``BINARY_*`` / ``UNARY_*``
     - 弹出操作数，派发到 ``BuiltinVariable`` 或 graph break
   * - ``POP_JUMP_IF_*`` / ``JUMP_*``
     - 修改 ``instruction_pointer`` ，模拟分支/循环
   * - ``RETURN_VALUE``
     - 结束当前子图，收集输出

执行主循环：run 与 step
============================

帧捕获的主路径在 ``convert_frame_assert`` 中创建 ``InstructionTranslator`` 并调用 ``run()`` ：

.. code-block:: text

   convert_frame_assert(frame, ...)
       │
       ├─ bytecode = clean_instructions(frame.f_code)   # 3.2 节
       │
       ├─ translator = InstructionTranslator(
       │       instructions=bytecode,
       │       f_code=frame.f_code,
       │       f_locals=...,
       │       ...
       │   )
       │
       └─ translator.run()
               │
               while instruction_pointer < len(instructions):
                   step()
                   # 可能在中途 graph_break() 提前退出当前子图

``step()`` 每次只处理 **一条 ** 字节码。这与 CPython 解释器的行为对齐——遇到 ``JUMP_BACKWARD`` 时 ``instruction_pointer`` 回退，IT 就在**同一张图上反复执行循环体 ** （或触发 unbacked 符号 / graph break，取决于循环形态）。

控制流：IT 不是「从上到下扫一遍」
======================================

许多读者第一次读 Dynamo 会误以为 IT 线性扫描字节码。实际上**跳转指令会改变 PC** 。考虑：

.. code-block:: python

   def fn(x):
       for i in range(3):
           x = x + 1
       return x

字节码里会出现 ``JUMP_BACKWARD`` 回到循环头。InstructionTranslator 的处理策略大致是：

- **固定次数、可分析的循环 ** ：有时展开或符号化 induction variable
- **无法证明的循环 / 数据依赖边界 ** ： ``graph_break`` 或 ``Unsupported``

这体现了第 2.1 节**编译器适应 Python** 原则：不拒绝带 ``for`` 的代码，但在无法静态理解时 **降级** 而非报错。

call_function：所有调用的总线
=================================

无论 opcode 是 ``CALL_FUNCTION`` 、 ``CALL`` 还是 ``INVOKE_FUNCTION`` ，最终都会汇聚到 ``call_function`` ：

.. code-block:: python
   :caption: InstructionTranslatorBase.call_function（简化示意）

   def call_function(self, fn, args, kwargs):
       # fn、args、kwargs 都是 VariableTracker
       return fn.call_function(self, args, kwargs)

**这是 IT 与 VariableTracker 的边界 ** ：

- IT 负责**字节码语义 ** （弹栈、压栈、改 PC）
- VariableTracker 负责**值语义 ** （这个调用能不能变成 FX 节点？）

``BuiltinVariable(torch.sin).call_function`` 会创建 Proxy； ``UserFunctionVariable`` 可能 inline 或 graph break； ``PrintVariable`` 之类则直接 break。第 3.4 节用 ``torch.sin(x)`` 具体展示 VariableTracker 一侧的逻辑；这里只需记住：**IT 从不直接操作 ATen，一切通过 ``call_function`` 派发 ** 。

与 OutputGraph、Guard 的交界
=================================

``self.output`` 是 ``OutputGraph`` 实例，贯穿整个 ``run()`` ：

.. code-block:: text

   InstructionTranslator.run()
       │
       ├─ handler 需要建图
       │      output.create_proxy(...) / tracer.create_node(...)
       │
       ├─ handler 读取 x.shape / x.dtype
       │      output.add_guard(...)          → 第 3.5 节
       │
       └─ handler 无法继续
              graph_break()                  → 第 3.6 节
              当前 OutputGraph 封口，生成 RESUME

Guard 不是在编译结束后批量生成的——**在符号执行过程中，每次访问可能变化的属性就追加 guard** 。 ``CheckFunctionManager`` （第 3.5 节）只是把 ``output.guards`` 编译成高效检查代码。

Graph Break 如何嵌入执行流
==============================

``graph_break()`` 定义在 ``InstructionTranslatorBase`` 上（第 3.6 节详述）。从 IT 视角看，一次 break 意味着：

.. code-block:: text

   正在执行的 InstructionTranslator
       │
       ├─ 当前 OutputGraph 视为完成（可能只有部分字节码）
       ├─ 生成恢复用的 continuation / RESUME 点
       └─ 父级 convert_frame 可能：
              • 编译当前子图
              • 创建新的 IT 实例继续剩余字节码
              • 或 fallback eager

因此 **一个 Python 函数可能对应多个 FX 子图 + 多段 eager** ，这是设计特性而非实现 bug。

convert_frame 与 InstructionTranslator 的协作
=================================================

把层次再拉高一层，完整协作如下：

.. code-block:: text

   set_eval_frame 回调（第 2.1 节）
       │
       ▼
   convert_frame.convert_frame(frame, ...)
       │
       ├─ 查 code object 缓存链表（第 3.7 节）
       ├─ guard 检查失败 → 重新进入下方流程
       │
       ├─ convert_frame_assert
       │      ├─ bytecode 预分析（3.2）
       │      ├─ InstructionTranslator.run()（本节）
       │      └─ 得到 OutputGraph + guards
       │
       └─ backend(gm, example_inputs)   # 通常 → AOTAutograd / Inductor

InstructionTranslator**只产出 FX Graph 和 guards** ；何时缓存、何时调用 Inductor，是 ``convert_frame`` 的职责。这种拆分同样体现 **阶段专精** （第 2.1 节）。

设计权衡小结
================

.. list-table::
   :header-rows: 1

   * - 选择
     - 好处
     - 代价
   * - 完整模拟 CPython 栈/ locals
     - 语义正确，兼容任意 Python 控制流
     - 代码量大，每个 opcode 要维护
   * - VariableTracker 双层结构
     - IT 与「能否进图」逻辑解耦
     - 调用链深，调试栈帧多
   * - 分发表 + 元类注册
     - 新版本 opcode 易扩展
     - ``symbolic_convert.py`` 单文件数千行
   * - graph break 内建于 IT
     - 永不因无法捕获而 crash
     - 子图碎片化，性能不稳定

小结
======

- **InstructionTranslator** 是 Dynamo 的符号化解释器，维护 ``stack`` / ``symbolic_locals`` / ``instruction_pointer``
- **分发表 ** 将 opcode 映射到同名 handler； ``step()`` 驱动主循环
- **``call_function``** 是所有调用的总线，实际建图逻辑在 VariableTracker（第 3.4 节）
- **``output``（OutputGraph） ** 在运行中累积 FX Graph 与 guards；无法继续时 ``graph_break``
- **``convert_frame``** 负责帧级编排；IT 只负责「这一帧的字节码意味着什么」

下一节我们从 **值语义** 侧深入：VariableTracker、Proxy 和 FakeTensor 如何把 ``torch.sin(x)`` 变成 FX 节点。
