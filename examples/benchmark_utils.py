"""
benchmark_utils.py — 基准测试工具函数

为 torch.compile 提供标准化的性能测量工具，包括：
- 预热 (warmup)
- CUDA 同步 (synchronize)
- 中位数计时 (median timing)
- 多种编译模式对比
- 自动报告生成

使用方法:
    from benchmark_utils import BenchmarkRunner, timing_median

    runner = BenchmarkRunner(model, x, n_warmup=10, n_iter=50)
    result = runner.measure()
    print(result)
"""

import torch
import time
import functools
from typing import Callable, Optional, Union, Any


def timing_median(
    fn: Callable,
    *args,
    n_warmup: int = 10,
    n_iter: int = 50,
    sync: bool = True,
    **kwargs,
) -> float:
    """测量函数的中位数执行时间（毫秒）。

    参数:
        fn: 要测量的函数
        n_warmup: 预热次数 (GPU 驱动预热、cuDNN 自动调优)
        n_iter: 测量次数
        sync: 是否在每次迭代后执行 CUDA synchronize
        *args, **kwargs: 传递给 fn 的参数

    返回:
        中位数执行时间 (毫秒)
    """
    # 预热阶段
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    if sync:
        torch.cuda.synchronize()

    # 测量阶段
    timings = []
    for _ in range(n_iter):
        start = time.perf_counter()
        fn(*args, **kwargs)
        if sync:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append(elapsed_ms)

    # 排序后取中位数，排除前端异常值
    timings.sort()
    median_ms = timings[len(timings) // 2]
    return median_ms


def timing_mean(
    fn: Callable,
    *args,
    n_warmup: int = 10,
    n_iter: int = 50,
    sync: bool = True,
    **kwargs,
) -> float:
    """测量函数的平均执行时间（毫秒）。

    与 timing_median 的接口一致，但返回算术平均。
    适用于方差较小的场景。
    """
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    if sync:
        torch.cuda.synchronize()

    total = 0.0
    for _ in range(n_iter):
        start = time.perf_counter()
        fn(*args, **kwargs)
        if sync:
            torch.cuda.synchronize()
        total += (time.perf_counter() - start) * 1000

    return total / n_iter


class BenchmarkResult:
    """单个 benchmark 的结果。"""

    def __init__(self, label: str, time_ms: float):
        self.label = label
        self.time_ms = time_ms

    @property
    def speedup(self) -> Optional[float]:
        """相对于基线的加速比，仅在存在基线时可用。"""
        return (
            getattr(self, "_baseline_ms", None) / self.time_ms
            if hasattr(self, "_baseline_ms")
            else None
        )

    def __repr__(self) -> str:
        base = f"{self.label}: {self.time_ms:.2f} ms"
        if self.speedup:
            base += f" (加速比: {self.speedup:.2f}x)"
        return base


class BenchmarkRunner:
    """基准测试运行器，支持多种编译模式和自动报告。

    用法:
        model = resnet50().cuda()
        x = torch.randn(32, 3, 224, 224, device='cuda')

        runner = BenchmarkRunner(model, x, n_warmup=10, n_iter=50)
        results = runner.compare_modes(
            modes=["eager", "default", "reduce-overhead"]
        )
        runner.print_report(results)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *example_args,
        n_warmup: int = 10,
        n_iter: int = 50,
        sync: bool = True,
        **example_kwargs,
    ):
        self.original_model = model
        self.example_args = example_args
        self.example_kwargs = example_kwargs
        self.n_warmup = n_warmup
        self.n_iter = n_iter
        self.sync = sync

    def measure(self, fn: Optional[Callable] = None) -> float:
        """测量函数的执行时间。

        如果未提供 fn，使用原始模型的 forward。
        """
        if fn is None:
            fn = self.original_model
        return timing_median(
            fn,
            *self.example_args,
            n_warmup=self.n_warmup,
            n_iter=self.n_iter,
            sync=self.sync,
            **self.example_kwargs,
        )

    def measure_mode(self, mode: str) -> BenchmarkResult:
        """测量指定编译模式的执行时间。

        mode 可以是 "eager"、"default"、"reduce-overhead"、"max-autotune"。
        """
        if mode == "eager":
            fn = self.original_model
        else:
            fn = torch.compile(self.original_model, mode=mode)
        time_ms = self.measure(fn)
        return BenchmarkResult(mode, time_ms)

    def compare_modes(self, modes: list) -> list:
        """对比多种编译模式，返回按时间排序的结果列表。"""
        results = []
        for mode in modes:
            result = self.measure_mode(mode)
            results.append(result)

        # 以 eager 为基线计算加速比
        eager_result = next((r for r in results if r.label == "eager"), None)
        if eager_result:
            baseline_ms = eager_result.time_ms
            for r in results:
                r._baseline_ms = baseline_ms

        results.sort(key=lambda r: r.time_ms)
        return results

    @staticmethod
    def print_report(results: list) -> None:
        """打印格式化的 benchmark 报告。"""
        print("=" * 60)
        print("Benchmark 报告")
        print("=" * 60)
        for r in results:
            speedup_str = f" ({r.speedup:.2f}x)" if r.speedup else ""
            print(f"  {r.label:<20s} {r.time_ms:>8.2f} ms{speedup_str}")
        print("=" * 60)

        if len(results) >= 2:
            best = results[0]
            worst = results[-1]
            print(f"  最佳: {best.label} ({best.time_ms:.2f} ms)")
            print(f"  最慢: {worst.label} ({worst.time_ms:.2f} ms)")
            if best.speedup:
                print(f"  最大加速比: {best.speedup:.2f}x")
        print("=" * 60)


@torch.no_grad()
def warmup_cuda() -> None:
    """预热 CUDA 驱动和 cuDNN。

    在正式 benchmark 前调用一次，确保 GPU 处于稳定状态。
    """
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    for _ in range(10):
        z = torch.mm(x, y)
        z = torch.sigmoid(z)
    torch.cuda.synchronize()


def cuda_memory_snapshot(device: int = 0) -> dict:
    """获取当前 CUDA 显存快照。"""
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }


def format_speedup_table(
    labels: list, eager_times_ms: list, compiled_times_ms: list
) -> str:
    """生成加速比表格的文本表示。"""
    header = f"{'场景':<25s} {'Eager (ms)':<15s} {'Compiled (ms)':<15s} {'加速比':<10s}"
    sep = "-" * len(header)
    lines = [header, sep]
    for label, eager_t, comp_t in zip(labels, eager_times_ms, compiled_times_ms):
        speedup = eager_t / comp_t
        lines.append(f"{label:<25s} {eager_t:<15.2f} {comp_t:<15.2f} {speedup:<10.2f}x")
    return "\n".join(lines)


def torch_compile_info() -> str:
    """获取 torch.compile 的当前配置信息。"""
    import torch._inductor.config as inductor_config
    import torch._dynamo.config as dynamo_config

    lines = ["torch.compile 配置:", "-" * 40]
    lines.append(f"  inductor.compile_threads: {inductor_config.compile_threads}")
    lines.append(f"  inductor.max_fusion_size: {inductor_config.max_fusion_size}")
    lines.append(f"  inductor.triton.cudagraphs: {inductor_config.triton.cudagraphs}")
    lines.append(f"  dynamo.cache_size_limit: {dynamo_config.cache_size_limit}")
    lines.append(
        f"  dynamo.assume_static_by_default: {dynamo_config.assume_static_by_default}"
    )

    if hasattr(inductor_config, "autotune_in_subproc"):
        lines.append(
            f"  inductor.autotune_in_subproc: {inductor_config.autotune_in_subproc}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # 简单的自测
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1024, 1024)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel().cuda()
    x = torch.randn(64, 1024, device="cuda")

    print(torch_compile_info())
    print()

    runner = BenchmarkRunner(model, x, n_warmup=5, n_iter=20)
    results = runner.compare_modes(["eager", "default", "reduce-overhead"])
    runner.print_report(results)
