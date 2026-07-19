"""量化与 torch.compile — 示例

展示动态量化 (Dynamic Quantization) 与 torch.compile 的配合，
以及模型大小和推理性能的对比。

注意：PyTorch 2.12+ 中 torch.ao.quantization 已弃用，
PT2E 量化已迁移到独立包 torchao (https://github.com/pytorch/ao)。
"""

import torch
import torch.nn as nn
import time
import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="torch.ao.quantization"
)

# ============================================================
# 定义一个模型
# ============================================================


class MLP(nn.Module):
    """多层感知机，线性层占主导，适合演示量化效果"""

    def __init__(self, in_dim=512, hidden_dim=1024, out_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# 动态量化
# ============================================================


# --- docs: dynamic_quant ---

import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic


class MLP(nn.Module):
    """多层感知机，线性层占主导，适合演示量化"""

    def __init__(self, in_dim=512, hidden_dim=1024, out_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def create_quantized_model():
    """构建 FP32 模型并执行动态量化。

    动态量化将 Linear 层的权重离线量化为 int8，
    激活值在推理时动态量化（保持 fp32）。
    这是对 Linear 密集型模型最简单的加速手段。
    """
    model = MLP()
    model.eval()

    quantized = quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


quantized_model = create_quantized_model()

# 验证输出
x = torch.randn(1, 512)
with torch.no_grad():
    out = quantized_model(x)
    print(f"量化模型输出 shape: {out.shape}, dtype: {out.dtype}")

# 量化模型可直接传给 torch.compile
compiled_quantized = torch.compile(quantized_model)
with torch.no_grad():
    out2 = compiled_quantized(x)
    print(f"量化 + 编译输出 shape: {out2.shape}")

# --- docs: end ---


# ============================================================
# 性能与模型大小对比
# ============================================================


# --- docs: perf_compare ---


def benchmark(model, example_inputs, n_warmup=20, n_iter=500, desc=""):
    with torch.no_grad():
        for _ in range(n_warmup):
            model(*example_inputs)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(*example_inputs)
        elapsed = time.perf_counter() - t0
    avg_ms = elapsed / n_iter * 1000
    print(f"  {desc:35s} {avg_ms:8.2f} ms/iter")
    return avg_ms


# 重建 FP32 模型做对比
fp32_model = MLP()
fp32_model.eval()
compiled_fp32 = torch.compile(fp32_model)

print()
print("=" * 55)
print("  性能对比 (CPU)")
print("=" * 55)

x = torch.randn(1, 512)

benchmark(fp32_model, (x,), desc="FP32 Eager")
benchmark(compiled_fp32, (x,), desc="FP32 + compile")
benchmark(quantized_model, (x,), desc="INT8 (Dynamic) Eager")
benchmark(compiled_quantized, (x,), desc="INT8 (Dynamic) + compile")

print()
# 模型大小
fp32_size = sum(p.numel() for p in fp32_model.parameters()) * 4 / 1024
print(f"FP32 权重大小: {fp32_size:.0f} KB")
print(f"INT8 权重大小（理论）: {fp32_size / 4:.0f} KB")
print(f"预期压缩比: 4x")
print()

# --- docs: end ---


# ============================================================
# PT2E 量化说明（概念代码，非可运行示例）
# ============================================================
# PT2E (PyTorch 2 Export) 量化是推荐的静态量化路径，但需要
# 安装 torchao 包（pip install torchao）。
#
# 完整流程：
#
#   import torchao
#   from torchao.quantization.pt2e import prepare_pt2e, convert_pt2e
#
#   exported = torch.export.export(model, example_inputs)
#   prepared = prepare_pt2e(exported, quantizer)
#   # calibrate...
#   converted = convert_pt2e(prepared)
#   compiled = torch.compile(converted)
