.. _appendix-glossary:

==================
附录 C  术语表
==================

.. glossary::

   TorchDynamo
      PyTorch 2.x 引入的 JIT 编译器前端，通过 CPython 的 PEP 523 框架在 Python
      字节码级别捕获计算图。具有动态性、与现有代码兼容性好的特点。

   字节码（bytecode）
      Python 源码编译后的中间表示，由一条条指令组成，每条指令对应一个编号
      （opcode）和一个或零个参数（arg）。TorchDynamo 在字节码级别进行分析和
      图捕获，这是它与 TorchScript（AST 级别）的关键区别。

   PEP 523
      即 "Adding a frame evaluation API to CPython"，是 Python 3.6 引入的
      CPython 扩展框架。它允许外部代码注册自定义的帧求值函数，替代 CPython
      默认的字节码解释执行。TorchDynamo 正是通过 PEP 523 接口在帧级别拦截
      Python 执行并捕获计算图。

   AOTAutograd
      Ahead-of-Time 自动微分模块。它在前向传播的同时追踪反向传播的计算图，
      "提前" 生成前向和反向的联合计算图，然后进行图分区。

   Inductor
      PyTorch 2.x 的默认编译器后端。它将计算图降级为循环级 IR，然后为 GPU
      生成 Triton 代码，为 CPU 生成 C++/OpenMP 代码。

   FX Graph
      PyTorch 的计算图中间表示（Intermediate Representation），是一个
      Python 层面的可操作图结构，由 ``torch.fx`` 模块提供。

   Graph Break
      图断裂。当 ``torch.compile`` 遇到无法捕获的 Python 操作时，会在该
      处打断图的捕获，形成多个子图的边界。

   Guard
      TorchDynamo 用于验证缓存编译结果是否仍然有效的运行时检查机制。
      当 guard 检查失败时，会触发重新编译。

   FakeTensor
      PyTorch 中用于元数据传播（meta propagation）的虚拟张量类型。它不包含
      实际数据，只携带 shape、dtype、device 等信息，使编译器可以在不执行
      实际计算的情况下推导整个计算图的张量元数据。在 TorchDynamo 和
      AOTAutograd 的符号执行中起核心作用。

   VariableTracker
      TorchDynamo 中用于追踪 Python 变量的核心抽象。当 Dynamo 符号化执行
      字节码时，每个 Python 对象都被包装为一个 ``VariableTracker`` 子类实例，
      记录其类型、值和操作历史，从而在模拟执行中逐步构建出 FX Graph。

   IRNode
      Inductor 内部的循环级中间表示，位于计算图（FX Graph）和最终代码
      之间的抽象层。

   Scheduler
      Inductor 中的调度器，负责将 IRNode 分组、融合，并决定执行顺序。

   融合区域（Fusion Regions）
      Inductor Scheduler 在执行融合时划分的逻辑区域。Scheduler 分析 IRNode
      之间的数据依赖和计算模式，将兼容的节点合并为 ``FusedSchedulerNode``，
      形成更大的 kernel 以提升执行效率。

   CSE（公共子表达式消除）
      编译器优化技术。当多个 IRNode 使用相同的索引表达式或中间值时，编译器
      自动复用已有的计算结果，避免重复计算和内存访问。在 Inductor 的
      ``SIMDKernel`` 中通过 CSE 缓存实现。

   延迟编译（deferred compilation）
      Inductor 避免在首次编译时一次性完成所有代码生成的策略。通过将部分
      kernel 的编译推迟到首次执行时，减少启动时的编译延迟，提升用户体验。

   再编译（recompilation）
      当 guard 检查失败或配置发生变化时，Inductor 触发重新编译以生成适配
      新输入形状或新配置的 kernel。

   Triton
      一种面向 GPU 编程的语言和编译器，提供比 CUDA 更高的抽象层级，
      Inductor 的 GPU 后端默认使用 Triton 作为代码生成目标。

   TTIR（Triton Dialect）
      Triton 编译器中最上层的中间表示（Intermediate Representation），
      直接对应 Triton 语言中的操作（如 ``tl.load``、``tl.dot``）。TTIR
      与设备无关，不包含 GPU 线程映射或内存层级信息。

   TTGIR（TritonGPU Dialect）
      Triton 编译器中层中间表示，在 TTIR 的基础上添加了 GPU 相关语义，
      包括数据布局（数据如何分布在 warp 和线程上）、共享内存分配、线程
      映射策略等信息。

   块级编程（Block-level Programming）
      Triton 的编程范式。开发者以数据块（block）为单位而非单个线程编写
      程序，每个 Triton program 处理一个数据块，块内的操作由编译器自动
      并行化到 warp 和线程上。

   自动内存合并（Automatic Memory Coalescing）
      Triton 编译器的核心特性。编译器自动分析 block 内的数据访问模式，
      生成合并的（coalesced）全局内存访问指令，开发者无需手动保证
      ``threadIdx.x`` 与地址的对应关系。

   自动调度（Automatic Scheduling）
      Triton 编译器自动决定如何将 block 内的计算映射到 warp 上，管理
      寄存器分配和指令流水线。同一个 Triton kernel 在不同 GPU 架构上
      可自动获得不同的 warp 映射策略。

   谓词执行（Predicated Execution）
      GPU 上实现条件操作的底层机制。当 warp 内部分线程的 mask 为 false 时，
      所有线程仍发射指令，但 mask 控制加载结果是否被保留。

   Functionalization
      将 PyTorch 的原位（in-place）操作转换为纯函数式操作的过程，
      是 AOTAutograd 的重要步骤。

   Min-Cut Recomputation
      AOTAutograd 使用的内存优化策略，通过在反向传播中重新计算某些
      前向中间结果来减少内存占用。

   Joint Graph
      AOTAutograd 生成的前向和反向联合计算图，包含完整的求导信息。

   proxy tensor
      AOTAutograd 在构建 joint graph 时使用的代理张量类型。它包装真实
      张量的元数据，在符号化追踪过程中记录每个操作的输入输出关系，最终
      生成完整的计算图。

   Define-by-run IR
      PyTorch 编译器的一种设计理念：不是从源码或 AST 中解析出计算图，
      而是"让 IR 随着 Python 程序的执行自然生长出来"。TorchDynamo 通过
      字节码追踪隐式定义图结构，成为这种理念的代表实现。

   Symbolic Shapes
      符号形状机制。用 ``SymNode`` / ``SymInt`` 代替具体维度数值，使一份编译
      结果适配多种输入尺寸。由 ``ShapeEnv`` （ ``symbolic_shapes.py`` ）管理。

   ShapeEnv
      PyTorch 中管理符号形状的核心组件，负责创建符号变量、记录约束、
      生成 guard 表达式。位于 ``torch/fx/experimental/symbolic_shapes.py`` 。

   ExportedProgram
      ``torch.export`` 的输出，包含带形状约束的 FX Graph 及元数据，
      可作为 Inductor、TensorRT 等后端的输入。

   AOTInductor
      Inductor 的离线（Ahead-of-Time）变体，在部署前编译所有 kernel，
      输出 C++ 可加载的共享库（ ``.so`` ），消除运行时编译延迟。

   预编译（Pre-compilation）
      在模型部署前通过一次虚拟运行触发所有 kernel 的编译，将编译时间
      前置到部署准备阶段，避免运行时编译延迟。

   Lowering（降级）
      将高层中间表示（FX Graph）转换为低层中间表示（Inductor IRNode）的过程。
      在 Inductor 中，每个 ATen 算子都有对应的 lowering 函数，将 FX 节点
      映射为 Pointwise、Reduction、TemplateBuffer 等 IR 类型。

   Pointwise（逐元素操作）
      对张量中每个元素独立应用相同计算的操作类型（如 ``sin``、``add``、
      ``mul``），是 Inductor 中最常见、最易融合的 IR 类型。

   Reduction（归约操作）
      将张量沿某个维度聚合为更少元素的操作类型（如 ``sum``、``mean``、
      ``max``），在 Inductor 中以 ``Reduction`` IRNode 表示。

   Decomposition（算子分解）
      将高层算子（如 ``layer_norm``、``softmax``）展开为基本算子
      （``mean``、``rsqrt``、``mul`` 等）的过程，由 AOTAutograd 在
      joint graph 构建时执行，降低后端的 lowering 负担。

   Dynamic Shapes（动态形状）
      编译时形状未知、运行时可能变化的张量维度。Inductor 使用符号形状
      （Symbolic Shapes）机制和 ``sympy`` 表达式处理动态形状，生成
      通用的 Triton kernel 而非为每个形状编译专用版本。

   Fusion（融合）
      将多个连续的操作合并为一个 kernel 执行，减少 kernel launch 开销和
      中间结果的显存读写。Inductor 的 Scheduler 负责在 IRNode 层面
      执行融合决策。

   时间步嵌入（Time Embedding）
      扩散模型中将当前时间步 ``t`` 编码为向量的机制，注入到 UNet 各层中
      使模型感知去噪进度。torch.compile 对这种动态输入模式需要特殊处理。

   预填充（Prefill）
      LLM 推理的第一阶段，一次性处理整个输入提示（prompt），生成第一个
      token 及对应的 Key-Value 缓存。torch.compile 在此阶段利用批量计算
      优势实现高吞吐。

   解码（Decode）
      LLM 推理的第二阶段，逐个生成后续 token，每次迭代只处理一个 token。
      此阶段的瓶颈是内存带宽而非计算，torch.compile 通过 kernel 融合和
      缓存优化缓解这一瓶颈。
