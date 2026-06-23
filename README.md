# 浅入深出 torch.compile

PyTorch 2.x `torch.compile` 编译器的深入分析教程，从入门到源码实现。

## 文档

在线文档（Read the Docs）：

> **https://torch-compile-deep-dive.readthedocs.io/zh-cn/latest/**

## 目录结构

```
source/
├── preface/                   # 前言
├── chapter_01_intro/          # torch.compile 简介
├── chapter_02_overview/       # 整体架构
├── chapter_03_dynamo/         # TorchDynamo
├── chapter_04_aotautograd/    # AOTAutograd
├── chapter_05_inductor/       # Inductor 后端
├── chapter_06_codegen/        # 代码生成
├── chapter_07_triton/         # Triton 编程
├── chapter_08_debug/          # 调试与分析
├── chapter_09_advanced/       # 进阶优化
├── chapter_10_cases/          # 实战案例
├── examples/                  # 可运行示例代码
└── appendix/                  # 附录
```

## 本地构建

```bash
pip install -r requirements.txt
make html       # 构建 HTML
make serve      # 构建并在 localhost:8000 启动预览
```

## 许可证

[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
