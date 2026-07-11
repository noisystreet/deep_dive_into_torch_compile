.. _preface:

==========
前   言
==========

.. 本篇内容

    - 为什么要写这本书
    - 目标读者
    - 预备知识
    - 全书结构
    - 如何阅读
    - 环境说明

欢迎来到 **浅入深出 torch.compile** ！本书旨在帮助读者系统性地理解 PyTorch 2.x 引入的即时编译（JIT）框架 ``torch.compile`` 的内部设计与实现。

为什么要写这本书？
========================

PyTorch 2.x 的 ``torch.compile`` 是 PyTorch 生态中里程碑式的创新。它将 Python 级别的模型代码通过 **TorchDynamo** 捕获为计算图，经 **AOTAutograd** 进行自动微分处理，最终由 **Inductor** 后端生成高效的 GPU/CPU 代码。这套编译栈让用户无需修改模型代码即可获得显著的性能提升。

然而， ``torch.compile`` 的架构复杂、组件众多，官方文档偏向 API 使用层面，对内部实现机制的深入讲解相对分散。本书的写作目标正是填补这一空白：

- **梳理架构全貌** ：将三大组件（Dynamo、AOTAutograd、Inductor）串联成一条完整的故事线
- **深入关键实现** ：通过源码分析，揭示图捕获、图分区、代码生成等核心环节的实现细节
- **配套可运行示例** ：每个关键概念都配有最小化可运行示例，让理论落地

目标读者
============

本教程面向：

- 有一定 PyTorch 使用经验，想了解 ``torch.compile`` 工作原理的开发者
- 对深度学习编译器感兴趣的读者
- 希望为 PyTorch 编译器贡献代码或自定义后端的开发者

预备知识
============

阅读本书前，建议读者具备：

1.**Python 基础** ：熟悉 Python 语言，了解装饰器、上下文管理器等特性
2.**PyTorch 基础** ：了解 Tensor 操作、自动求导（ ``autograd`` ）、基本模型训练流程
3.**编译原理基础** （可选）：了解 AST、IR、JIT 等基本概念会有所帮助，但不是必须的

本书结构
============

.. list-table::
   :header-rows: 1

   * - 章节
     - 内容
   * - 第 1 章
     - torch.compile 简介：Hello World、基本用法、性能初探
   * - 第 2 章
     - 整体架构：编译流水线、数据流、FX Graph 基础、编译缓存
   * - 第 3 章
     - TorchDynamo：字节码基础、字节码分析、图捕获、guard 机制、符号形状
   * - 第 4 章
     - AOTAutograd：联合求导、图分区、min-cut 重计算、functionalization、算子分解
   * - 第 5 章
     - Inductor 后端：FX Passes、IRNode、Scheduler、融合与布局、Pattern Matcher
   * - 第 6 章
     - 代码生成：CPU/GPU 平台代码生成与 kernel launch
   * - 第 7 章
     - Triton 编程：Triton 语言基础与自定义 kernel
   * - 第 8 章
     - 调试与分析：日志、minimizer、性能分析、Dynamic Shapes 调试
   * - 第 9 章
     - 进阶优化：自定义后端、编译配置调优、Export 与 AOTInductor 离线部署
   * - 第 10 章
     - 实战案例：模型优化、训练全流程、Dynamic Shapes、多 GPU

如何阅读
============

- **按顺序阅读** ：本书章节按知识递进关系组织，建议从第 1 章开始顺序阅读
- **动手实践** ：每章包含代码示例，建议在本地环境中实际运行
- **深入源码** ：文中会引用 PyTorch 源码的特定位置，建议结合源码对照阅读
- **使用附录** ：附录中包含关键代码阅读指南和术语表，阅读中遇到不熟悉的术语可随时查阅

环境说明
============

本书示例基于以下环境：

- **Python**3.13
- **PyTorch**v2.12.1
- **CUDA**12.x（GPU 相关章节）
- **Linux x86_64**

.. note::

   本书示例在 Linux 平台上开发和测试。macOS 和 Windows 用户可通过 WSL2 或 Docker 获得相同的实验环境。
