"""演示 AOTAutograd 的前向/反向联合追踪"""

import torch
from torch._functorch.aot_autograd import aot_function
from functorch.compile import make_boxed_func


# 编译器：接收 fx.GraphModule，返回可调用函数（包装为 boxed 格式）
def simple_compiler(gm, example_inputs):
    print(f"[AOT] 正在编译图，包含 {len(gm.graph.nodes)} 个节点")
    return make_boxed_func(gm.forward)


def fn(x, y):
    return (x * y).sum()


aot_fn = aot_function(fn, fw_compiler=simple_compiler, bw_compiler=simple_compiler)
x = torch.randn(3, requires_grad=True)
y = torch.randn(3)
result = aot_fn(x, y)
result.backward()
print(f"输入: x={x.data}, y={y.data}")
print(f"输出: {result}")
print(f"x.grad: {x.grad}")
