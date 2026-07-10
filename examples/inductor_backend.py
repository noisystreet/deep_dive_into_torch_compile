"""使用 Inductor 后端并查看生成的 Triton 代码"""
# --- docs: start ---

import torch


def fn(x, y):
    return torch.sin(x) + torch.cos(y)


compiled_fn = torch.compile(fn, backend="inductor")
x = torch.randn(4).cuda()
y = torch.randn(4).cuda()
result = compiled_fn(x, y)
print(result)
# --- docs: end ---
