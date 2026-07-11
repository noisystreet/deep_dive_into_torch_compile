"""
Scheduler 融合流程图 — Inductor 的 Scheduler 流水线
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from style import setup_figure, save_or_show, box, arrow, label, diamond, COLORS


def draw():
    fig, ax = setup_figure(width=10, height=5)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    # 标题
    label(
        ax,
        5.5,
        5.7,
        "Scheduler 融合与调度流程",
        fontsize=14,
        bold=True,
        color=COLORS["aot"],
    )

    # === 步骤框 ===
    steps = [
        (
            1.0,
            4.5,
            3.0,
            1.2,
            "Lowering 输出\n[IRNode1, IRNode2, ...]",
            COLORS["inductor"],
        ),
        (4.5, 4.5, 3.0, 1.2, "SchedulerNode\n包装每个 IRNode", COLORS["inductor"]),
        (8.0, 4.5, 3.0, 1.2, "依赖分析\n构建依赖边", COLORS["aot"]),
        (
            3.5,
            2.0,
            4.5,
            1.2,
            "融合循环\n基于启发式算法\n→ FusedSchedulerNode",
            COLORS["dynamo"],
        ),
        (8.5, 2.0, 3.0, 1.2, "codegen() 调用\n分发到后端", COLORS["aot"]),
    ]
    for x, y, w, h, text, color in steps:
        box(ax, x, y, w, h, text, color=color)

    # 箭头
    for x1, y1, x2, y2 in [
        (4.0, 5.1, 4.5, 5.1),
        (7.5, 5.1, 8.0, 5.1),
        (9.5, 4.5, 9.5, 2.6),
        (8.0, 2.6, 8.5, 2.6),
    ]:
        arrow(ax, x1, y1, x2, y2)

    # 步骤 5 → 判定
    arrow(ax, 10.0, 2.0, 10.0, 1.4)

    # 迭代融合回环
    arrow(
        ax,
        4.0,
        3.5,
        3.5,
        3.2,
        color=COLORS["dynamo"],
        lw=1.5,
        connectionstyle="arc3,rad=0.3",
    )
    label(
        ax,
        2.5,
        3.5,
        "迭代融合\n直至无可融合节点",
        fontsize=8,
        color=COLORS["dynamo"],
        italic=True,
    )

    # GPU/CPU 判定
    diamond(ax, 10.5, 0.8, 0.8, "GPU\nor\nCPU?")
    arrow(ax, 11.3, 0.8, 11.8, 0.8)
    arrow(ax, 9.7, 0.8, 9.2, 0.8)
    label(
        ax,
        12.2,
        0.8,
        "TritonScheduling\n(codegen/triton.py)",
        fontsize=8,
        ha="left",
        color=COLORS["triton"],
        bold=True,
    )
    label(
        ax,
        8.8,
        0.8,
        "CPPScheduling\n(codegen/cpp.py)",
        fontsize=8,
        ha="right",
        color=COLORS["dynamo"],
        bold=True,
    )

    return fig


if __name__ == "__main__":
    save_or_show(draw(), "scheduler_fusion")
