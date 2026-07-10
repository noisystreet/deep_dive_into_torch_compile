"""使用 torch.fx 获取计算图"""
# --- docs: start ---

import torch


class MyModel(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x) + torch.cos(x)


model = MyModel()
fx_model = torch.fx.symbolic_trace(model)
print(fx_model.graph)
fx_model.graph.print_tabular()
# --- docs: end ---
