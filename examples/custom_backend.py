"""自定义后端示例

展示如何使用 torch.compiler.register_backend 注册自定义后端，
以及如何与 AOTAutograd 交互。
"""

import torch
import torch.fx as fx

# ============================================================
# 基础自定义后端
# ============================================================


# --- docs: basic_backend ---

import torch
from torch import fx


@torch.compiler.register_backend
def my_backend(gm: fx.GraphModule, example_inputs):
    """自定义后端：打印图结构后用 eager 执行"""
    print("FX Graph 节点数:", len(gm.graph.nodes))
    for node in gm.graph.nodes:
        print(f"  {node.op}: {node.target}")
    return gm.forward  # 使用 eager 执行


@torch.compile(backend="my_backend")
def fn(x):
    return torch.sin(x) + torch.cos(x)


if __name__ == "__main__":
    result = fn(torch.randn(3))
    print("结果:", result)

# --- docs: end ---

# ============================================================
# 外部编译器后端
# ============================================================


# --- docs: external_compiler ---

import json
import torch
import torch.fx as fx


class ExternalCompiler:
    """模拟外部编译器"""

    def compile(self, graph_json):
        print(f"接收到图: {len(graph_json['nodes'])} 个节点")
        # 这里连接到真实的外部编译器
        return lambda *args: None


compiler = ExternalCompiler()


@torch.compiler.register_backend
def external_backend(gm: fx.GraphModule, example_inputs):
    # 将 FX Graph 序列化为可 JSON 序列化的格式
    graph_json = {"nodes": []}
    for node in gm.graph.nodes:
        graph_json["nodes"].append(
            {
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "args": [str(a) for a in node.args],
            }
        )

    # 发送到外部编译器
    compiled_fn = compiler.compile(graph_json)
    return compiled_fn


@torch.compile(backend="external_backend")
def fn_ext(x):
    return torch.sin(x) + torch.cos(x)


if __name__ == "__main__":
    result = fn_ext(torch.randn(3))
    print("结果:", result)

# --- docs: end ---

# ============================================================
# 与 AOTAutograd 交互
# ============================================================


# --- docs: aot_backend ---

from functorch.compile import min_cut_rematerialization_partition


def my_compiler(gm, example_inputs):
    """处理 AOTAutograd 分区后的子图"""
    print(f"编译子图: {len(gm.graph.nodes)} 个节点")
    return gm.forward


@torch.compile(backend=my_compiler)
def fn_aot(x):
    return torch.sin(x) + torch.cos(x)


if __name__ == "__main__":
    result = fn_aot(torch.randn(3))
    print("结果:", result)

# --- docs: end ---
