# agents.md — 浅入深出 torch.compile 项目

## 项目概述

本项目编写一本 **浅入深出 torch.compile** 技术文档，使用 reStructuredText（`.rst`）格式，基于 Sphinx 构建。

- 文档源目录：`source/`
- 构建输出：`./_build/html/`（`make html` 后生成）
- 目标读者：有 PyTorch 使用经验、希望了解 torch.compile 内部机制的开发者
- 参考实现：**PyTorch 2.x**，主线为 PyTorch 官方实现
- 平台：Linux x86_64，CUDA 12.x（GPU 相关章节）

## 项目文件说明

| 文件 | 说明 |
|------|------|
| `source/preface/index.rst` | 前言：编写动机、目标读者、预备知识、全书结构 |
| `source/index.rst` | Sphinx 根文档（toctree 入口） |
| `source/chapter_01_intro/` | torch.compile 简介（Hello World、基本用法、性能初探） |
| `source/chapter_02_overview/` | 整体架构（编译流水线、数据流、FX Graph 基础、编译缓存） |
| `source/chapter_03_dynamo/` | TorchDynamo（字节码基础、字节码分析、图捕获、guard 机制、符号形状） |
| `source/chapter_04_aotautograd/` | AOTAutograd（联合求导、图分区、算子分解） |
| `source/chapter_05_inductor/` | Inductor 后端（FX Passes、Lowering 流程、IRNode、Scheduler、Pattern Matcher、延迟编译） |
| `source/chapter_06_codegen/` | 代码生成（IR 到代码变换、CPU/GPU 代码生成、kernel launch） |
| `source/chapter_07_triton/` | Triton 编程（语言基础、自定义 kernel） |
| `source/chapter_08_debug/` | 调试与分析（日志、minimizer、profiling、Dynamic Shapes 调试） |
| `source/chapter_09_advanced/` | 进阶优化（自定义后端、编译策略、Export 与 AOTInductor 离线部署） |
| `source/chapter_10_cases/` | 实战案例（模型优化、训练全流程、Dynamic Shapes、多 GPU） |
| `source/appendix/` | 附录（参考资源、代码阅读指南、术语表） |
| `source/examples/` | 可运行的 Python 示例代码 |
| `source/conf.py` | Sphinx 构建配置 |
| `Makefile` | 构建入口（`make html` / `make clean`） |
| `scripts/precommit-check.sh` | 预提交检查脚本（验证 RST 文档语法） |
| `requirements.txt` | 构建依赖（sphinx, sphinx-rtd-theme, sphinxcontrib-mermaid） |
| `.readthedocs.yaml` | Read the Docs 构建配置 |
| `LICENSE` | CC BY-SA 4.0 许可证 |
| `.gitignore` | 版本控制忽略规则 |
| `agents.md` | **本文件**：AI 助手的工作上下文和约束 |

## 通用约束

1. **许可证**：本文档采用 CC BY-SA 4.0（Creative Commons Attribution-ShareAlike 4.0 International），详见 `LICENSE` 文件
2. **文档格式**：使用 reStructuredText（`.rst`）格式，中文写作
3. **git hooks**：clone 后首次提交前，运行以下命令启用 pre-commit 检查：

   ```bash
   git config --local core.hooksPath .githooks
   ```

   否则 pre-commit 检查不会自动生效。
4. **引用源码**：使用绝对路径的 `file:///` 链接引用源码文件，格式为 `` `链接文本 <file:///绝对路径/文件>`__ ``
4. **避免冗余**：不创建不必要的文件，优先编辑已有文件
5. **权限**：不做 `git push --force`、`reset --hard` 等破坏性操作
6. **代码示例**：在文档中引用代码时，说明其所属文件和行号范围
7. **示例验证**：所有 `.py` 示例代码应保证可运行

## 文档写作规范

### 文档结构
- 每篇文档应有标题
- 按章节组织，章节层级不超过三级
- 内容末尾标注生成日期和项目名称

### 引用规范
- 引用源码文件使用绝对路径 markdown 链接
- 引用 API 或概念使用 `` ` `` 反引号标记
- 关键代码片段应提供文件定位

### 内容深度
- 概念讲解与代码示例相结合
- 复杂流程配合 Mermaid 图表说明
- 关键抽象用表格列出其核心字段与方法
- 避免大段堆叠代码，优先提炼核心模式

### 写作风格（核心：浅入深出，夹叙夹议）

**禁止罗列结论**。每一个知识点都必须有推导过程，遵循"是什么 → 为什么 → 怎么用 → 源码长什么样"的递进链条。

- **浅入深出**：从直观可运行的例子出发引入概念，读者能"看见"它在做什么，再逐步揭开底层实现。每一节都遵循：表象问题 → 直观解法 → 引出深层机制 → 源码印证。**不要一上来就甩概念定义或架构图。**
- **夹叙夹议**：叙述"代码做了什么"的同时，必须穿插"为什么这样设计"——性能考量、历史背景、与其他方案的权衡对比。代码是论据，不是结论。
- **避免知识点罗列**：每个新概念必须有上下文铺垫才引入。如果出现"XX有以下几个特点：1... 2... 3..."这种列表体，必须有前置案例让读者自然感受到这些特点的存在，而不是突兀地堆砌。
- **代码即证据**：每一个论断必须附代码或源码引用佐证。没有源码引用支撑的观点都是空谈。关键代码片段要标注来自 PyTorch 源码的具体文件和行号。
- **过渡自然**：段落之间、章节之间要有承上启下的过渡句。比如"上一节我们看到了 X 的行为，但它背后依赖 Y 机制，接下来我们深入 Y"。禁止生硬切换话题。

## 写作路线图

按以下顺序推进内容编写：

1. **第 1 章：torch.compile 简介** — torch.compile 概念、Hello World、基本用法、性能初探
2. **第 2 章：整体架构** — 编译流水线、数据流、FX Graph 基础、编译缓存
3. **第 3 章：TorchDynamo** — 字节码基础 → 字节码分析 → 图捕获 → guard → graph break → 缓存 → 符号形状
4. **第 4 章：AOTAutograd** — 联合求导 → 图分区 → min-cut 重计算 → 算子分解
5. **第 5 章：Inductor 后端** — FX Passes → Lowering 流程 → 虚拟化 → IRNode → Scheduler → Pattern Matcher → 延迟编译
6. **第 6 章：代码生成** — 代码生成概览 → IR 到代码变换 → CPU/GPU 代码生成 → Kernel Launch
7. **第 7 章：Triton 编程** — Triton 语言基础与自定义 kernel
8. **第 8 章：调试与分析** — 日志系统、minimizer、性能分析、Dynamic Shapes
9. **第 9 章：进阶优化** — 自定义后端、编译配置调优、Export 与 AOTInductor 离线部署
10. **第 10 章：实战案例** — 模型优化、训练全流程、Dynamic Shapes、多 GPU

## 构建方法

```bash
# 安装依赖
pip install -r requirements.txt

# 构建 HTML 文档
make html

# 构建产物位于 _build/html/
```

自动部署到 Read the Docs 后，文档会自动构建并托管。本地构建也可通过 `make html` 完成。

## Cursor Cloud specific instructions

本项目是一个 **Sphinx 文档站点**（中文 torch.compile 教程），"运行应用"即构建并预览 HTML 文档。依赖已由启动 update script (`pip install -r requirements.txt`) 安装好（`sphinx` / `sphinx-rtd-theme` / `sphinxcontrib-mermaid`）。常用命令见 `README.md` 与 `Makefile`，下面只记录非显而易见的注意事项：

- **构建**：`make html`（产物在 `_build/html/`）。本地开发用 `make html` 即可；不要加 `-W`。CI（`.github/workflows/ci.yml`）使用 `make html SPHINXOPTS="-W"` 把警告当错误，当前内容存在 1 个 Pygments 代码高亮警告（`chapter_03_dynamo/05_graph_break.rst`），所以 CI 的 Build 步骤会失败——这是既有内容问题，不是环境问题。
- **预览**：`make serve` 会先构建再用 `python3 -m http.server` 启动预览，默认端口 8000（`PORT` 变量未设时为 8000）。在 cloud VM 中已验证 `http://localhost:8000/` 可正常访问并渲染。也可直接 `cd _build/html && python3 -m http.server 8000`。
- **Lint / RST 检查**：`bash scripts/precommit-check.sh`（即 CI 的 "Check RST syntax" 步骤）。注意：脚本除 Sphinx 语法解析外，还会用 grep 检查 `**bold**` 后紧跟中文标点等内联标记风格；当前 master 内容触发了这些风格警告，脚本因 `set -e` 返回退出码 1，**这是 master 上既有的 CI 失败原因**（Sphinx 解析本身报告"无警告"）。环境本身正常。
- **git hooks**（仅在需要提交触发 pre-commit RST 检查时）：`git config --local core.hooksPath .githooks`。
- 修改 `.rst` 内容后无热重载，需重新 `make html` 才能在预览中看到更新。
