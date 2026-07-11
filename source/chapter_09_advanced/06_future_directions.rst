.. _future-directions:

================
未来方向
================

torch.compile 是一个快速发展的项目。这一节梳理当前已知的局限性和社区正在推进的改进方向。

更快的编译时间
==================

编译时间是 torch.compile 当前最主要的痛点。优化的方向包括：

**缓存层次增强** 。当前缓存主要基于源码哈希。未来计划引入基于图结构的缓存，允许形状不同但结构相同的图共享编译结果。

**增量编译** 。如果模型只有部分结构变化（如添加一个新的 layer），只有变化的子图需要重新编译，其余部分复用已有缓存。

**磁盘缓存共享** 。在大规模分布式训练中，让所有 GPU 进程共享同一个磁盘缓存，避免 N 个进程独立编译 N 次。这在第 5.10 节已有初步实现，但还有优化空间。

更广泛的算子覆盖
===================

虽然 Inductor 已经覆盖了 ATen 的大部分算子，但仍有少量算子没有 lowering 或 decomposition：

- 某些高阶操作（ ``while_loop`` 、 ``cond`` 、 ``map`` ）
- 特定的量化算子
- 自定义的 C++ 扩展算子

社区正在通过以下途径逐步扩大覆盖范围：

- 更完整的 ``core_aten_decompositions``
- 自动 fallback 机制（ ``implicit_fallbacks`` ）
- 算子贡献指南

更好的动态形状支持
=====================

动态形状（Dynamic Shapes）是 torch.compile 面临的核心挑战之一。

**符号形状优化** 。当前动态形状的 kernel 不如静态形状版本高效，因为编译器无法利用具体数值信息做优化（如循环展开、常量折叠）。正在研究的方法包括：

- 对符号形状做值域分析（range analysis），在值域内做优化
- 运行时 profiling，收集形状分布后生成特化 kernel

**延迟编译** 。对于形状变化的情况，可以先使用通用 kernel 运行，同时在后端编译特化版本，后续调用自动切换到特化版本。

多设备支持
==============

**AMD ROCm 支持** 。Triton 对 AMD GPU 的支持（通过 ROCm）正在积极开发中。当前 Inductor 在 AMD GPU 上可以运行，但部分 Triton kernel 的 autotune 功能尚未完全就绪。

**Apple Metal 支持** 。 ``codegen/mps.py`` 是 Inductor 的 Apple Silicon 后端，但当前成熟度远低于 GPU 和 CPU 后端。社区正在推动 MPS 后端的完善。

**更多硬件后端** 。通过自定义后端机制（第 9.1 节），社区已经为 Habana Gaudi、Graphcore IPU 等硬件开发了后端适配。

编译器基础设施改进
=====================

**Inductor IR 的进一步发展** 。当前的 IRNode 设计（Pointwise、Reduction 等）对于许多操作是足够的，但对于稀疏操作、动态控制流、复数类型等场景还需要扩展。

**融合区域（Fusion Regions）** 。第 5.7 节提到的 融合区域（Fusion Regions） 还在持续演进中，未来可能在 FX Graph 级别做更精细的融合规划。更多细节见 ``fx_passes/fusion_regions.py`` 。

**更智能的布局优化** 。当前的 ``FlexibleLayout`` 允许 codegen 选择输出布局，但选择的策略是启发式规则。未来可能引入基于 cost model 的布局选择。

与推理优化框架的整合
========================

**TensorRT 集成** 。社区正在探索将 Inductor 生成的 Triton kernel 导入 TensorRT 进行进一步的图优化和融合。

**vLLM 兼容性** 。vLLM 等推理框架使用自定义的 Triton kernel（如 PagedAttention），需要与 Inductor 生成 kernel 的缓存和调度机制兼容。

**AOT Inductor** 。AOT（Ahead-of-Time）Inductor 允许在部署前编译所有 kernel，消除运行时编译。完整流程与 API 见第 9.5 节。

可调试性和可观测性
=====================

编译流水线的可视化、性能分析工具正在不断完善：

- **Perfetto 集成** ：将编译事件导出到 Perfetto trace 格式，与 Chrome Trace 兼容但提供更丰富的信息
- **编译时间报告** ：在每个编译步骤结束后输出耗时统计，帮助定位编译瓶颈
- **Graph 可视化** ：在 ``TORCH_COMPILE_DEBUG`` 报告中加入更详细的 Graph 可视化

降低使用门槛
=================

- **更友好的错误信息** ：将 ``MissingOperatorWithoutDecomp`` 等错误翻译为更易于理解的提示，并给出修复建议
- **自动诊断工具** ：一键诊断命令，分析现有模型的编译状况
- **最佳实践模板** ：针对不同模型类型（CNN、Transformer、LLM）的推荐配置集合

总结
======

torch.compile 的未来发展聚焦于三个方向：

1.**更快** ：减少编译时间，提升缓存效率
2.**更广** ：支持更多硬件、更多算子、更多场景
3.**更易用** ：降低调试和配置的门槛，让开发者不需要深入理解编译器内部就可以使用
