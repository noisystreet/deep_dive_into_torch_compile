.. _bytecode-analysis:

==================
字节码分析
==================

上一节我们了解了 CPython 字节码的基本结构——指令、栈、值传递。这一节我们来看 Dynamo 具体是怎么分析这些字节码，从中识别出 Tensor 操作并构建 FX Graph 的。

Dynamo 的字节码分析器位于 ``torch/_dynamo/bytecode_analysis.py``，它与符号执行引擎 ``InstructionTranslator``（在 ``symbolic_convert.py`` 中）配合工作。字节码分析器负责静态分析（不执行代码），而 InstructionTranslator 负责动态模拟（"假装执行"的同时记录图）。
