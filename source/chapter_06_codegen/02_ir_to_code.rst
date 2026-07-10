.. _ir-to-code:

=========================
IR 到代码的变换机制
=========================

第 5 章介绍过，Scheduler 完成融合后将 ``FusedSchedulerNode`` 分发给对应后端的 ``codegen()`` 方法。这一节深入这个分发过程，看一个融合的 IRNode 组如何被翻译为 Triton 或 C++ 的源码字符串。

入口：从 Scheduler 到 Scheduling
=======================================

Scheduler 遍历所有节点，调用每个节点的 ``codegen()`` ：

.. code-block:: python
   :caption: pytorch/torch/_inductor/scheduler.py（简化示意）

   class Scheduler:
       def codegen(self):
           for node in self.nodes:   # 遍历 SchedulerNode / FusedSchedulerNode
               backend = get_scheduling_for_device(node.device.type)
               backend.codegen(node)

对于 GPU 节点， ``TritonScheduling.codegen()`` 被调用。 ``TritonScheduling`` （继承自 ``SIMDScheduling`` ）接收到节点后，执行以下步骤：

.. code-block:: text

   TritonScheduling.codegen(node)
       │
       ├─ 1. 确定 tiling 参数
       │      get_tiling_and_scores()
       │      基于 numel / reduction_numel 计算
       │      BLOCK_SIZE、num_warps 等
       │
       ├─ 2. 创建 kernel 实例
       │      kernel = TritonKernel(...)
       │
       ├─ 3. codegen_node_schedule_with_kernel()
       │      两遍扫描，生成循环体
       │
       ├─ 4. codegen_kernel()
       │      生成完整的 Triton 源码
       │      （包括 @triton.jit 装饰器、参数签名等）
       │
       └─ 5. call_kernel()
              注册 kernel launch 到 wrapper

两遍扫描（Two-Pass Codegen）
================================

``codegen_node_schedule_with_kernel`` （在 ``simd.py`` 中）是代码生成的核心——它使用 **两遍扫描** 策略处理融合后的节点列表：

.. code-block:: python
   :caption: pytorch/torch/_inductor/codegen/simd.py（简化示意）

   def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
       with kernel:
           # 第一遍：收集索引信息
           for node in node_schedule:
               if node is DisableReduction:
                   kernel.disable_reduction()
               elif node is EnableReduction:
                   kernel.enable_reduction()
               else:
                   node.decide_inplace_update()
                   index_vars = kernel.split_and_set_ranges(node.get_ranges())
                   # 记录所有索引表达式
                   ...

           # 第二遍：生成代码
           for node in node_schedule:
               if node is DisableReduction:
                   kernel.disable_reduction()
               elif node is EnableReduction:
                   kernel.enable_reduction()
               else:
                   index_vars = kernel.split_and_set_ranges(node.get_ranges())
                   node.codegen(index_vars)   # ← 核心调用

两遍扫描的必要性：第一遍收集索引信息是为了确定 loop 的边界和 tiling 参数，第二遍才实际生成加载/计算/存储代码。其中 ``split_and_set_ranges`` 将 IRNode 的语义范围（如 ``[M, N]`` ）映射为 Triton 的并行索引（如 ``pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`` ）。

split_and_set_ranges：循环切分
==========================================

``split_and_set_ranges`` 方法（在 ``simd.py`` 的 ``SIMDKernel`` 中）是索引变换的核心。它接收 IRNode 的 ``ranges`` （如 ``[1024, 768]`` ），根据 tiling 策略将其切分为多个级别的循环：

.. code-block:: text

   IRNode.ranges = [1024, 768]

   tiling 策略:
       x_block = 1024  → TRITON_BLOCK_SIZE = 1024
       y_block = 768   → TRITON_BLOCK_SIZE = 768

   split_and_set_ranges 输出:
       xpid = tl.program_id(0)    ← 跨 program 并行
       ypid = tl.program_id(1)
       x = xpid * 1024 + tl.arange(0, 1024)
       y = ypid * 768  + tl.arange(0, 768)

这些索引变量在后续的 ``node.codegen(index_vars)`` 中被传递给 IRNode 的 ``inner_fn`` ，从而实现从语义索引到硬件索引的映射。

node.codegen：触发 inner_fn
=========================================

每个 ``SchedulerNode`` 的 ``codegen(index_vars)`` 方法调用其包含的 IRNode 的 codegen。对于 ``Pointwise`` 节点，核心逻辑在 ``Loops.codegen`` 中：

.. code-block:: text

   node.codegen(index_vars)
       │
       ├─ 1. kernel.loads.write()
       │      调用 ops.load 加载输入
       │      生成: x = tl.load(x_ptr + offsets, mask=mask)
       │
       ├─ 2. kernel.compute.write()
       │      调用 IRNode.inner_fn(index_vars)
       │      执行: ops.sin(ops.load(...))
       │      生成: sin_x = tl.sin(x)
       │
       └─ 3. kernel.stores.write()
               调用 ops.store 存储结果
               生成: tl.store(output_ptr + offsets, result, mask=mask)

这个过程中的关键机制是 **CSE（公共子表达式消除）** 。 ``SIMDKernel`` 维护了一个 CSE 缓存，当多个 IRNode 使用相同的索引表达式或中间值时，自动复用已有的计算结果：

.. code-block:: text

   两个 Pointwise 节点融合后的 CSE 效果:
       # 节点 A: output = sin(x) + cos(x)
       # 节点 B: output2 = sin(x) * cos(x)

       生成的代码:
       x_val = tl.load(x_ptr + offsets)   ← 一次加载
       sin_x = tl.sin(x_val)              ← 一次 sin
       cos_x = tl.cos(x_val)              ← 一次 cos
       tl.store(out1_ptr, sin_x + cos_x)  ← 复用 sin_x, cos_x
       tl.store(out2_ptr, sin_x * cos_x)  ← 复用 sin_x, cos_x

codegen_kernel：组装完整源码
===========================================

当循环体生成完毕后， ``codegen_kernel()`` 方法（在 ``triton.py`` 的 ``TritonKernel`` 中）将之前分散生成的代码片段组装为完整的 Triton kernel：

.. code-block:: text

   TritonKernel.codegen_kernel()
       │
       ├─ 组装 imports
       │      import triton
       │      import triton.language as tl
       │
       ├─ 组装 kernel 签名
       │      @triton.jit
       │      def kernel_name(
       │          x_ptr, y_ptr, output_ptr,
       │          n_elements,
       │          BLOCK_SIZE: tl.constexpr,
       │      ):
       │
       ├─ 组装循环体
       │      pid = tl.program_id(axis=0)
       │      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       │      mask = offsets < n_elements
       │      x = tl.load(x_ptr + offsets, mask=mask)
       │      ...
       │      tl.store(output_ptr + offsets, result, mask=mask)
       │
       └─ 组装 benchmark 代码
              if config.benchmark_kernel:
                  插入基准测试代码

生成的完整源码作为字符串返回，随后被 ``TritonCodeCache`` 或 ``AsyncCompile`` 提交给 Triton 编译器编译。

表达式生成：ops 原语的渲染
========================================

代码生成中最精细的工作是 **将 inner_fn 中的 ``ops.*`` 调用渲染为具体的代码字符串 **。这个过程由 ``OpsHandler`` 实现——每个后端实现自己的 OpsHandler，在 codegen 阶段替换全局的 ``V.ops`` 。

以 ``inner_fn = lambda idx: ops.sin(ops.load(x, idx))`` 为例，当 ``node.codegen(index_vars)`` 被调用时， ``inner_fn(index_vars)`` 的执行轨迹如下：

.. code-block:: text

   inner_fn(index_vars) 开始执行
       │
       ├─ 1. ops.load(x, index_vars) 被调用
       │      ↓
       │      TritonOverrides.load(self, name, index)
       │      → 检查 CSE 缓存：同样的 (name, index) 是否已加载？
       │      → 是：返回缓存的 CSEVariable
       │      → 否：生成 tl.load 代码，创建 CSEVariable，缓存
       │      → 返回: CSEVariable(name="x_val")
       │      →  body 中追加:
       │         x_val = tl.load(x_ptr + offsets, mask=mask)
       │
       ├─ 2. ops.sin(x_val) 被调用
       │      ↓
       │      TritonOverrides.sin(self, x: CSEVariable)
       │      → 检查 CSE 缓存：同样的 sin(x_val) 是否已计算？
       │      → 否：生成 tl.sin 代码
       │      → 返回: CSEVariable(name="sin_x")
       │      →  body 中追加:
       │         sin_x = tl.sin(x_val)
       │
       └─ 3. 返回 sin_x (CSEVariable)
              → 这个 CSEVariable 会被用于后续的 ops.store

这里的核心机制是**CSE 缓存 + 字符串追加 ** ：

.. code-block:: python
   :caption: pytorch/torch/_inductor/codegen/triton.py（简化）

   class TritonOverrides:
       def __init__(self):
           self.cse = CSE()
           self.body = IndentedBuffer()  # 代码片段的增量缓冲区

       def load(self, name: str, index: CSEVariable) -> CSEVariable:
           # 1. 检查 CSE 缓存
           key = (name, index)
           if key in self.cse.cache:
               return self.cse.cache[key]

           # 2. 生成代码字符串
           line = f"tl.load({name}_ptr + ({index}))"
           
           # 3. 创建 CSEVariable 并追加到 body
           var = self.cse.generate(self.body, line)
           return var

   class CSE:
       """公共子表达式消除：避免生成重复的代码行"""
       def __init__(self):
           self.cache: dict[Any, CSEVariable] = {}
           self.variable_counter = 0

       def generate(self, body: IndentedBuffer, line: str) -> CSEVariable:
           self.variable_counter += 1
           var = CSEVariable(f"tmp{self.variable_counter}")
           body.writeline(f"{var} = {line}")
           return var

``IndentedBuffer`` 是一个简单的字符串缓冲区——每次 ``body.writeline()`` 追加一行代码。codegen 完成后，整个 ``body`` 的内容就是 kernel 的循环体源码。

TritonKernel 的代码是**逐行追加** 到 ``body`` 中的。所有 ``ops.*`` 方法都在 body 中追加代码行，并返回一个 ``CSEVariable`` 作为中间值。最终生成的代码块如下：

.. code-block:: python

   # body 的内容（简化）
   tmp0 = tl.load(in_ptr0 + (x0))            # ops.load → CSEVariable tmp0
   tmp1 = tl.sin(tmp0)                        # ops.sin → CSEVariable tmp1
   tl.store(out_ptr0 + (x0), tmp1, mask=mask) # ops.store → 直接追加

``TritonOverrides`` 完整的 ops 映射如下（以 load/store/sin/add 为例）：

.. code-block:: python
   :caption: pytorch/torch/_inductor/codegen/triton.py（TritonOverrides 简化）

   class TritonOverrides:
       def load(self, name: str, index: CSEVariable) -> CSEVariable:
           return self._load_or_store(name, index, is_store=False)

       def store(self, name: str, index: CSEVariable, value: CSEVariable, mode=None):
           if mode == "atomic_add":
               line = f"tl.atomic_add({name}_ptr + ({index}), {value}, mask=mask)"
           else:
               line = f"tl.store({name}_ptr + ({index}), {value}, mask=mask)"
           self.body.writeline(line)

       def sin(self, x: CSEVariable) -> CSEVariable:
           return self.cse.generate(self.body, f"tl.sin({x})")

       def add(self, a: CSEVariable, b: CSEVariable) -> CSEVariable:
           return self.cse.generate(self.body, f"{a} + {b}")

       def reduction(self, dtype, reduction_type, value: CSEVariable) -> CSEVariable:
           fn = {"sum": "tl.sum", "max": "tl.max", "min": "tl.min"}[reduction_type]
           return self.cse.generate(self.body, f"{fn}({value}, 1)[:, None]")

每一个 ``ops.*`` 调用都通过 ``self.cse.generate()`` 或 ``self.body.writeline()`` 向 ``body`` 中追加一行或多行代码。不同后端的 OpsHandler 实现本质上就是 **将 IR 语义翻译为不同语言的三地址代码 ** 。

Kernel 参数：从 IRNode 到指针运算
==========================================

代码生成不仅要生成循环体，还需要确定 kernel 函数的参数。 ``TritonKernel`` 在构造时会遍历所有 IRNode，收集需要的参数信息：

.. code-block:: text

   TritonKernel.__init__()
       │
       ├─ 1. 遍历所有 IRNode，为每个 buffer 分配参数名
       │      InputBuffer(x) → "in_ptr0"
       │      ComputedBuffer(buf0) → "buf0_ptr"
       │      ...
       │
       ├─ 2. 计算每个 buffer 的指针偏移
       │      如果 buffer 有 layout（如 AsStridedLayout），
       │      则 stride 信息也会被编码为参数
       │
       ├─ 3. 确定 constexpr 参数
       │      BLOCK_SIZE、num_warps、num_stages 等 tiling 参数
       │
       └─ 4. 初始化 body（IndentedBuffer）
              所有后续代码追加到这个 body 中

具体而言，对于每个 buffer， ``TritonKernel`` 会计算其**指针表达式** ：

.. code-block:: text

   IRNode: InputBuffer(x), layout=FixedLayout(shape=[1024], stride=[1])
       → in_ptr0 = x.data_ptr()                     # 基础指针
       → in_ptr0 + (index)                           # 加上偏移量

   IRNode: InputBuffer(y), layout=AsStridedLayout(shape=[3,4], stride=[4,1])
       → in_ptr1 = y.data_ptr()
       → in_ptr1 + (index[0] * 4 + index[1])         # 考虑 stride

布局信息（shape、stride）决定了指针表达式的形式。对于视图操作（transpose、slice 等）， ``AsStridedLayout`` 的 stride 被编码为额外的 kernel 参数：

.. code-block:: text

   # transpose 视图的 kernel 参数会包含 stride 信息
   def kernel(in_ptr0, in_ptr1, out_ptr0,
              ks0, ks1,               # kernel 的 stride 参数（在 Triton 中为 tl.constexpr）
              BLOCK_SIZE: tl.constexpr):
       ...
       offset = x0 * ks0 + x1 * ks1   # 使用 stride 参数计算偏移

这种设计使得 Inductor 无需为每个不同 stride 的组合重新编译 kernel——stride 通过参数传入，kernel 在运行时自适应。

Wrapper 层：从 kernel 源码到可运行函数
==================================================

Scheduling 层生成 kernel 源码后，Wrapper 层负责 **组装多个 kernel 的 launch 代码 ** 并**编译为可执行函数 ** 。

整个过程分为三步：

.. code-block:: text

   Scheduler.codegen()
       │
       ├─ Step 1: TritonScheduling.codegen(node)
       │    生成 Triton kernel 源码字符串
       │    注册到 PythonWrapperCodegen
       │
       ├─ Step 2: PythonWrapperCodegen.finalize()
       │    将所有注册的 kernel launch 组装为
       │    一个完整的 Python 函数
       │
       └─ Step 3: codecache.py
              将生成的 Python 代码编译为 .so
              加载为可调用对象

**Step 1** 中， ``TritonScheduling.call_kernel()`` 向 Wrapper 注册一条 launch：

.. code-block:: python
   :caption: pytorch/torch/_inductor/codegen/triton.py（简化）

   class TritonScheduling:
       def call_kernel(self, kernel: TritonKernel):
           # 向 wrapper 注册 kernel launch
           self.wrapper.generate_kernel_call(
               kernel_name=kernel.kernel_name,
               kernel=kernel,
               grid=self.grid_fn(kernel),    # grid 计算函数
           )

**Step 2** 中， ``PythonWrapperCodegen.finalize()`` 将所有注册的 kernel launch 组装为完整的 Python 函数：

.. code-block:: text

   PythonWrapperCodegen.finalize()
       │
       ├─ 1. 生成函数签名
       │      def call(args_0, args_1, ...):
       │
       ├─ 2. 生成 buffer 分配代码
       │      buf0 = torch.empty([1024, 768], device='cuda')
       │      ...
       │
       ├─ 3. 按拓扑顺序插入 kernel launch
       │      # kernel 1
       │      triton_kernel_1[(1,)](
       │          in_ptr0=args_0,
       │          out_ptr0=buf0,
       │          BLOCK_SIZE=1024,
       │      )
       │      # kernel 2
       │      triton_kernel_2[(1,)](
       │          in_ptr0=buf0,
       │          out_ptr0=args_1,
       │          BLOCK_SIZE=512,
       │      )
       │
       └─ 4. 生成返回值
               return (args_1,)

``PythonWrapperCodegen`` 同时维护了 kernel 之间的 **依赖关系**——如果一个 kernel 的输出是另一个 kernel 的输入，Wrapper 会确保它们按顺序执行，并在适当的位置插入同步操作。

**Step 3** 中， ``codecache.py`` 将生成的 Python 代码提交给 Triton 编译器：

.. code-block:: text

   codecache.py
       │
       ├─ PyCodeCache → 编译 Python wrapper 代码
       │      exec() 或 compile() 后返回可调用对象
       │
       ├─ TritonCodeCache → 编译单个 Triton kernel
       │      triton.compile(kernel_src)
       │      返回 triton.Function
       │
       └─ AsyncCompile → 异步编译队列
              在后台线程中执行编译
              主线程继续处理下一个 kernel
              通过 Future 获取编译结果

AsyncCompile 利用了 Triton kernel 之间的编译独立性——既然多个 kernel 之间没有依赖关系（依赖已由 Scheduler 在 Fusion 阶段解决），就可以在后台线程中并行编译。对于 LLM 推理场景，这可以将编译时间减少 30-50%。

从 IRNode 到 Triton：完整追踪
====================================

以下用一个简单模型 ``sin(x) + cos(x)`` 追踪从 IRNode 到最终 Triton 代码的完整变换过程：

.. code-block:: python

   # 原始函数
   def fn(x):
       return torch.sin(x) + torch.cos(x)

   # Step 1: 经 Dynamo + AOTAutograd 后的 FX Graph
   # %sin = call_function[target=aten.sin](args = (%x,))
   # %cos = call_function[target=aten.cos](args = (%x,))
   # %add = call_function[target=aten.add](args = (%sin, %cos))

   # Step 2: Lowering 后的 IRNode（GraphLowering 的输出）
   # operations = [
   #   Pointwise(  # sin
   #     ranges=[N],
   #     inner_fn=lambda idx: ops.sin(ops.load("x", idx)),
   #   ),
   #   Pointwise(  # cos
   #     ranges=[N],
   #     inner_fn=lambda idx: ops.cos(ops.load("x", idx)),
   #   ),
   #   Pointwise(  # add
   #     ranges=[N],
   #     inner_fn=lambda idx: ops.load("sin", idx) + ops.load("cos", idx),
   #   ),
   # ]

   # Step 3: Scheduler 融合（三个 Pointwise 可融合为同一个 kernel）
   # FusedSchedulerNode([
   #   Pointwise(sin), Pointwise(cos), Pointwise(add)
   # ])

   # Step 4: codegen 执行 inner_fn
   # 两遍扫描后，第二遍执行时：
   #   inner_fn(index_vars) 在 TritonOverrides 上下文中执行
   #   每次 ops.* 调用向 body 追加一行代码

   # Step 5: 生成的 Triton 代码
   # @triton.jit
   # def triton_kernel(in_ptr0, out_ptr0, out_ptr1, out_ptr2,
   #                   xnumel, XBLOCK: tl.constexpr):
   #     xoffset = tl.program_id(0) * XBLOCK
   #     xindex = xoffset + tl.arange(0, XBLOCK)[:]
   #     xmask = xindex < xnumel
   #     x0 = xindex
   #     # Pointwise(sin) 生成的代码
   #     tmp0 = tl.load(in_ptr0 + (x0), xmask)
   #     tmp1 = tl.sin(tmp0)
   #     # Pointwise(cos) 生成的代码
   #     tmp2 = tl.cos(tmp0)              # CSE 复用 tmp0
   #     # Pointwise(add) 生成的代码
   #     tmp3 = tmp1 + tmp2
   #     tl.store(out_ptr0 + (x0), tmp1, xmask)   # sin 的输出
   #     tl.store(out_ptr1 + (x0), tmp2, xmask)   # cos 的输出
   #     tl.store(out_ptr2 + (x0), tmp3, xmask)   # add 的输出

   # Step 6: Wrapper 生成的 Python 代码
   # def compiled_call(x):
   #     buf0 = torch.empty([N], device='cuda')
   #     buf1 = torch.empty([N], device='cuda')
   #     buf2 = torch.empty([N], device='cuda')
   #     triton_kernel[(grid,)](x, buf0, buf1, buf2, N, XBLOCK=1024)
   #     return buf0, buf1, buf2

这个例子中可以看到：

- **CSE 复用 ** ： ``tmp0``\ （x 的加载）在 sin 和 cos 之间被自动复用——只需要加载一次
- **每个物理 kernel 对应一个 Triton 函数 ** ：三个 Pointwise 被融合为单个 ``triton_kernel``
- **中间 buffer 由 Wrapper 管理 ** ：sin/cos/add 各自的输出 ``buf0/buf1/buf2`` 在 Wrapper 中分配和传递
- **编译推迟到 codecache** ：生成的源代码以字符串形式传递给 Triton 编译器

CPU 后端：从 IRNode 到 C++/OpenMP
========================================

CPU 后端（ ``CPPScheduling`` + ``CPPKernel`` ）的流程与 GPU 后端镜像对称，但最终输出是 C++/OpenMP 代码：

.. code-block:: text

   CPPScheduling.codegen(node)
       │
       ├─ 1. 确定 tiling（基于缓存行）
       │      Tiling 以 64 bytes 为单位对齐
       │      确保每次加载填满缓存行
       │
       ├─ 2. 创建 kernel 实例
       │      kernel = CPPKernel(...)
       │
       ├─ 3. 两遍扫描（同 Triton 后端）
       │      第一遍收集索引
       │      第二遍生成代码
       │
       ├─ 4. 生成 OpenMP 循环
       │      #pragma omp parallel for
       │      for (long i0 = 0; i0 < 1024; i0 += 16) {
       │
       ├─ 5. 生成循环体（CPPOverrides）
       │      float tmp0 = in_ptr0[i0];
       │      float tmp1 = std::sin(tmp0);
       │      float tmp2 = std::cos(tmp0);
       │      out_ptr0[i0] = tmp1 + tmp2;
       │
       └─ 6. 注册到 wrapper
              CppWrapperCpu 接收 C++ 代码字符串

CPU 端的 OpsHandler 实现（ ``CPPOverrides`` ）与 ``TritonOverrides`` 结构相同，只是将 ``tl.sin`` 替换为 ``std::sin`` ：

.. code-block:: python
   :caption: pytorch/torch/_inductor/codegen/cpp.py（简化）

   class CPPOverrides:
       def load(self, name: str, index: CSEVariable) -> CSEVariable:
           return self.cse.generate(self.body, f"{name}[{index}]")

       def store(self, name: str, index: CSEVariable, value: CSEVariable):
           self.body.writeline(f"{name}[{index}] = {value};")

       def sin(self, x: CSEVariable) -> CSEVariable:
           return self.cse.generate(self.body, f"std::sin({x})")

       def add(self, a: CSEVariable, b: CSEVariable) -> CSEVariable:
           return self.cse.generate(self.body, f"{a} + {b}")

与 Triton 后端的区别在于：

- **循环结构 ** ：CPU 后端使用 ``#pragma omp parallel for`` 的多线程循环，不依赖 GPU 的 program_id 机制
- **tiling 粒度 ** ：CPU 以缓存行（64 bytes）为步长，GPU 以 warp 大小（32 threads）为步长
- **向量化** ：CPU 后端会尝试自动生成向量化代码（如 ``__m256`` 类型的 SIMD 指令），而 GPU 后端则依赖 Triton 编译器自动向量化

向量化是 CPU 代码生成中最重要的优化。CPPKernel 会分析 IRNode 的 ``ranges`` ，如果连续维度的大小大于 4（针对 AVX2 的 256-bit 寄存器），将循环步长扩展为向量宽度：

.. code-block:: text

   标量循环：
       for (int i = 0; i < N; i++) {
           out[i] = std::sin(in[i]);
       }

   向量化后（CPPKernel 自动生成）：
       for (int i = 0; i < N; i += 8) {
           __m256 vec = _mm256_loadu_ps(&in[i]);
           __m256 result = _mm256_sin_ps(vec);  # 使用 Intel SVML 库
           _mm256_storeu_ps(&out[i], result);
       }

CPU 后端的 Wrapper 也不同于 PythonWrapperCodegen——``CppWrapperCpu`` 会生成可直接被 Python 调用的 C 扩展代码，通过 ``pybind11`` 导出为 Python 模块。

GPU/CPU 代码生成的对比
================================

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - 维度
     - GPU（Triton）
     - CPU（C++/OpenMP）
   * - 循环模型
     - 基于 program_id 的并行
     - 基于 OpenMP 的多线程循环
   * - 索引生成
     - ``tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)``
     - ``for (int i0 = 0; i0 < N; i0 += step)``
   * - mask 处理
     - 显式 mask (``tl.where(xmask, ...)``)
     - 边界检查隐式处理
   * - 向量化
     - Triton 编译器自动处理
     - Inductor 手动生成 SIMD 指令
   * - 共享内存
     - ``tl.atomic_add`` 、 ``tl.atomic_max``
     - 不支持（无 CUDA shared memory）
   * - 编译流程
     - Triton JIT 编译 → PTX → SASS
     - g++/clang JIT 编译 → .so
   * - Wrapper 类型
     - ``PythonWrapperCodegen``
     - ``CppWrapperCpu``

小结
======

这一节介绍了 IRNode 到代码的变换机制：

- **入口 ** ：Scheduler 分发节点给 ``TritonScheduling.codegen()`` 或 ``CPPScheduling.codegen()``
- **两遍扫描 ** ：第一遍收集索引信息，第二遍触发 ``inner_fn`` 执行
- **表达式生成** ： ``inner_fn`` 中的 ``ops.*`` 调用被 ``OpsHandler`` 渲染为代码字符串，通过 CSE 缓存避免冗余
- **Kernel 参数 ** ：从 IRNode 的 layout 信息计算指针和 stride 参数
- **Wrapper 层 ** ：PythonWrapperCodegen 组装多个 kernel 的 launch 代码，codecache.py 编译为可执行模块
- **CPU 后端** ：结构对称，但输出为 C++/OpenMP 代码，支持 SIMD 向量化
