"""
Scheduler 融合流程图
替代 mermaid flowchart，展示 Inductor Scheduler 的融合流水线
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from style import setup_figure, COLORS
import matplotlib.patches as mpatches


def draw():
    fig, ax = setup_figure(width=10, height=5)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # === 步骤框（沿波形路径排列）===
    steps = [
        (
            1.0,
            4.5,
            3.0,
            1.2,
            "Lowering 输出\n[IRNode1, IRNode2, ...]",
            COLORS["inductor"],
            False,
        ),
        (
            4.5,
            4.5,
            3.0,
            1.2,
            "SchedulerNode\n包装每个 IRNode",
            COLORS["inductor"],
            False,
        ),
        (8.0, 4.5, 3.0, 1.2, "依赖分析\n构建依赖边", COLORS["aot"], False),
        (
            3.5,
            2.0,
            4.5,
            1.2,
            "融合循环\n基于启发式算法\n→ FusedSchedulerNode",
            COLORS["dynamo"],
            False,
        ),
        (8.5, 2.0, 3.0, 1.2, "codegen() 调用\n分发到后端", COLORS["aot"], False),
    ]

    for x, y, w, h, label, color, _ in steps:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08",
            facecolor=color,
            alpha=0.1,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    # 箭头（步骤间）
    arrows = [
        (4.0, 5.1, 4.5, 5.1),  # 1→2
        (7.5, 5.1, 8.0, 5.1),  # 2→3
        (9.5, 4.5, 9.5, 2.6),  # 3→4（向下）
        (8.0, 2.6, 8.5, 2.6),  # 4→5
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=2),
        )

    # 从步骤 4 回到步骤 2 的循环箭头（迭代融合）
    ax.annotate(
        "",
        xy=(4.0, 3.5),
        xytext=(3.5, 3.2),
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["dynamo"],
            lw=1.5,
            connectionstyle="arc3,rad=0.3",
        ),
    )
    ax.text(
        2.5,
        3.5,
        "迭代融合\n直至无可融合节点",
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["dynamo"],
        style="italic",
    )

    # === GPU/CPU 分支 ===
    # 判定菱形
    diamond = mpatches.Polygon(
        [[10.5, 1.4], [11.5, 0.8], [10.5, 0.2], [9.5, 0.8]],
        facecolor=COLORS["gray"],
        alpha=0.1,
        edgecolor=COLORS["gray"],
        linewidth=1.5,
    )
    ax.add_patch(diamond)
    ax.text(
        10.5,
        0.8,
        "GPU\nor\nCPU?",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=COLORS["gray"],
    )

    # GPU 分支
    ax.annotate(
        "",
        xy=(10.5, 0.2),
        xytext=(11.8, 0.2),
        arrowprops=dict(arrowstyle="->", color=COLORS["triton"], lw=1.5),
    )
    ax.text(
        12.2,
        0.2,
        "TritonScheduling\n(codegen/triton.py)",
        ha="left",
        va="center",
        fontsize=8,
        color=COLORS["triton"],
        fontweight="bold",
    )

    # CPU 分支
    ax.annotate(
        "",
        xy=(10.5, 0.2),
        xytext=(9.2, 0.2),
        arrowprops=dict(arrowstyle="->", color=COLORS["dynamo"], lw=1.5),
    )
    ax.text(
        8.8,
        0.2,
        "CPPScheduling\n(codegen/cpp.py)",
        ha="right",
        va="center",
        fontsize=8,
        color=COLORS["dynamo"],
        fontweight="bold",
    )

    # 从步骤 5 到判定的箭头
    ax.annotate(
        "",
        xy=(10.0, 2.0),
        xytext=(10.0, 1.4),
        arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5),
    )

    # 标题
    ax.text(
        5.5,
        5.7,
        "Scheduler 融合与调度流程",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=COLORS["aot"],
    )

    return fig


if __name__ == "__main__":
    fig = draw()
    out = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "../source/_static/figures/scheduler_fusion.svg"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"Saved: {out}")
