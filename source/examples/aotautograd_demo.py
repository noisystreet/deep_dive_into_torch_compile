"""演示 AOTAutograd 的前向/反向联合追踪"""
import torch
from torch._functorch.aot_autograd import aot_function


def my_grad_fn(flat_args, flat_grads):
    return flat_grads


def fn(x, y):
    return (x * y).sum()


aot_fn = aot_function(fn, fw_compiler=my_grad_fn, bw_compiler=my_grad_fn)
x = torch.randn(3, requires_grad=True)
y = torch.randn(3)
result = aot_fn(x, y)
result.backward()
print(x.grad)
