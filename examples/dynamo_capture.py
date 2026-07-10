"""演示 TorchDynamo 图捕获与 graph break"""
# --- docs: start ---

import torch


def complex_function(x):
    x = torch.sin(x)
    if x.sum() > 0:
        x = torch.cos(x)
    else:
        x = torch.tanh(x)
    return x


compiled_fn = torch.compile(complex_function, backend="eager", fullgraph=False)
x = torch.randn(4)
print(compiled_fn(x))
# --- docs: end ---
