.. _appendix-environment:

==================
附录 D  环境搭建
==================

本书所有示例基于以下环境开发和测试：

- **Python** 3.13
- **PyTorch** v2.12.1
- **CUDA** 12.x（GPU 相关章节）
- **Linux x86_64**

安装 PyTorch v2.12.1
=========================

使用 pip 安装：

.. code-block:: bash

   pip install torch==2.12.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu124

使用 conda 安装：

.. code-block:: bash

   conda install pytorch==2.12.1 torchvision==0.22.1 pytorch-cuda=12.4 -c pytorch -c nvidia

验证安装
============

.. code-block:: python

   import torch
   print(torch.__version__)  # 应输出 2.12.1
   print(torch.cuda.is_available())  # GPU 可用时应为 True

.. note::

   macOS 和 Windows 用户可通过 WSL2 或 Docker 获得与本书一致的 Linux 实验环境。

示例代码
============

本书配套示例代码位于 ``source/examples/`` 目录：

.. code-block:: text

   examples/
   ├── hello_compile.py       # Hello World 示例
   ├── basic_fx_graph.py      # FX Graph 基础
   ├── dynamo_capture.py      # Dynamo 图捕获
   ├── aotautograd_demo.py    # AOTAutograd 演示
   ├── inductor_backend.py    # Inductor 后端
   └── triton_kernel.py       # Triton kernel

运行任一示例：

.. code-block:: bash

   python source/examples/hello_compile.py

工具推荐
============

调试与日志
----------------

- **TORCH_LOGS**：最常用的调试工具。通过环境变量控制日志输出：

  .. code-block:: bash

     TORCH_LOGS="+dynamo" python train.py          # 查看 Dynamo 捕获过程
     TORCH_LOGS="+guards" python train.py          # 查看 guard 生成
     TORCH_LOGS="+inductor" python train.py        # 查看 Inductor 编译
     TORCH_LOGS="+schedule" python train.py        # 查看 Scheduler 调度

  多个模块可以组合：

  .. code-block:: bash

     TORCH_LOGS="+dynamo,+guards,+inductor" python train.py

  关于日志系统的完整说明，见第 8 章。

- **torch.compiler.debug**：生成 HTML 格式的编译调试报告：

  .. code-block:: python

     with torch.compiler.debug():
         compiled_fn(x, y)

  报告包含捕获到的 FX Graph、graph break 位置、生成的代码等。

性能分析
--------------

- **torch.cuda.Event**：手动基准测试（第 1.6 节使用）：

  .. code-block:: python

     start = torch.cuda.Event(enable_timing=True)
     end = torch.cuda.Event(enable_timing=True)
     start.record()
     result = compiled_fn(x, y)
     end.record()
     torch.cuda.synchronize()
     print(start.elapsed_time(end))

- **PyTorch Profiler**：详细的 kernel 级别性能分析：

  .. code-block:: python

     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
         compiled_fn(x, y)
     print(prof.key_averages().table(sort_by="cuda_time_total"))

- **torchinp**：社区工具，用于分析 torch.compile 的编译时间和内存开销。

缓存管理
--------------

- **torch.compiler.reset()**：清空内存中的编译缓存
- **TORCHINDUCTOR_CACHE_DIR**：持久化编译缓存到磁盘

  .. code-block:: bash

     TORCHINDUCTOR_CACHE_DIR=/tmp/compile_cache python train.py
     # 第二次运行复用缓存，跳过编译

- **TORCHINDUCTOR_FORCE_DISABLE_CACHES=1**：禁用缓存（开发调试时有用）
