"""Export 与 AOTInductor 示例

展示 torch.export 和 torch._export.aot_compile 的基本用法。
"""

# ============================================================
# 最小 export 示例
# ============================================================


# --- docs: basic_export ---

import torch


class M(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x @ self.weight.T)


model = M()
model.weight = torch.nn.Parameter(torch.randn(10, 20))

example_inputs = (torch.randn(4, 20),)

# 导出为 ExportedProgram
exported = torch.export.export(model, example_inputs)
print(exported.graph_module)

# --- docs: end ---

# ============================================================
# 动态形状 export
# ============================================================


# --- docs: dynamic_shapes ---

import torch

batch = torch.export.Dim("batch", min=1, max=1024)
dynamic_shapes = {"x": {0: batch}}

exported = torch.export.export(
    model,
    example_inputs,
    dynamic_shapes=dynamic_shapes,
)

# --- docs: end ---

# ============================================================
# AOTInductor 编译导出
# ============================================================


# --- docs: aot_compile ---

import torch


class M2(torch.nn.Module):
    def forward(self, x):
        return x.sin() + x.cos()


model2 = M2()
example_inputs2 = (torch.randn(8, 16),)

# 指定输出目录，Inductor 在此生成 .so 等产物
so_path = torch._export.aot_compile(
    model2,
    example_inputs2,
    options={
        "aot_inductor.output_path": "/tmp/aot_model",
    },
)
print(so_path)

# --- docs: end ---
