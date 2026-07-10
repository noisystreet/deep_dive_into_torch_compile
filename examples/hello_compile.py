"""Hello torch.compile - 最小示例"""

import torch


def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


compiled_foo = torch.compile(foo, backend="eager", fullgraph=True)

x = torch.randn(3, 3)
y = torch.randn(3, 3)
result = compiled_foo(x, y)
print(result)
