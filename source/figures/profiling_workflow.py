"""
Profiling 工作流与瓶颈诊断图
"""

from style import setup_figure, save_or_show, box, arrow, label, diamond, COLORS


def draw():
    fig, ax = setup_figure(width=10, height=6.5)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    label(ax, 6, 7.7, "Profiling 工作流", fontsize=14, bold=True, color=COLORS["aot"])

    # 前三步
    steps = [
        (1.0, 6.2, 3.0, 0.9, "编译函数\n@torch.compile", COLORS["dynamo"]),
        (4.5, 6.2, 3.0, 0.9, "使用 torch.profiler\n捕获", COLORS["inductor"]),
        (8.0, 6.2, 3.0, 0.9, "导出 Chrome Trace", COLORS["aot"]),
    ]
    for x, y, w, h, text, c in steps:
        box(ax, x, y, w, h, text, color=c)

    arrow(ax, 4.0, 6.65, 4.5, 6.65)
    arrow(ax, 7.5, 6.65, 8.0, 6.65)

    box(ax, 3.5, 4.5, 5.0, 0.9, "在 chrome://tracing 中加载", color=COLORS["aot"])
    arrow(ax, 6.0, 6.2, 6.0, 5.4)

    diamond(ax, 6.0, 3.5, 0.8, "分析 GPU\n时间线")
    arrow(ax, 6.0, 4.5, 6.0, 4.3)

    label(
        ax, 3.0, 3.5, "识别 kernel 间隙", fontsize=9, bold=True, color=COLORS["dynamo"]
    )
    label(ax, 9.0, 3.5, "识别性能瓶颈", fontsize=9, bold=True, color=COLORS["inductor"])
    arrow(ax, 5.2, 3.5, 4.0, 3.5)
    arrow(ax, 6.8, 3.5, 8.0, 3.5)

    box(
        ax,
        0.3,
        2.2,
        4.0,
        0.8,
        "Kernel launch 开销过高",
        color=COLORS["dynamo"],
        fontsize=8,
    )
    box(
        ax,
        7.7,
        2.2,
        4.0,
        0.8,
        "计算瓶颈 / 内存瓶颈",
        color=COLORS["inductor"],
        fontsize=8,
    )

    for i, txt in enumerate(["reduce-overhead 模式", "CUDA Graph"]):
        box(
            ax,
            0.3 + i * 2.0,
            0.8,
            2.0,
            0.6,
            txt,
            color=COLORS["dynamo"],
            fontsize=7,
            bold=False,
        )
        ax.plot(
            [1.3 + i * 2.0, 1.3 + i * 2.0],
            [1.7, 1.4],
            color=COLORS["arrow"],
            linewidth=1.5,
            linestyle="--",
        )

    ax.plot(
        [2.3, 2.3, 1.3],
        [2.2, 1.8, 1.4],
        color=COLORS["arrow"],
        linewidth=1.5,
        linestyle="--",
    )
    ax.plot(
        [2.3, 2.3, 3.3],
        [2.2, 1.8, 1.4],
        color=COLORS["arrow"],
        linewidth=1.5,
        linestyle="--",
    )

    for i, txt in enumerate(["max-autotune", "更低精度"]):
        box(
            ax,
            8.7 + i * 1.8,
            0.8,
            1.8,
            0.6,
            txt,
            color=COLORS["inductor"],
            fontsize=7,
            bold=False,
        )
        ax.plot(
            [9.6 + i * 1.8, 9.6 + i * 1.8],
            [1.7, 1.4],
            color=COLORS["arrow"],
            linewidth=1.5,
            linestyle="--",
        )

    ax.plot(
        [9.7, 9.7, 9.6],
        [2.2, 1.8, 1.4],
        color=COLORS["arrow"],
        linewidth=1.5,
        linestyle="--",
    )
    ax.plot(
        [9.7, 9.7, 11.4],
        [2.2, 1.8, 1.4],
        color=COLORS["arrow"],
        linewidth=1.5,
        linestyle="--",
    )

    box(ax, 4.5, 0.1, 3.0, 0.6, "验证性能提升", color=COLORS["triton"], fontsize=8)
    for x in [1.3, 3.3, 9.6, 11.4]:
        ax.plot(
            [x, x, 6.0],
            [0.8, 0.5, 0.7],
            color=COLORS["arrow"],
            linewidth=1,
            linestyle=":",
        )

    return fig


if __name__ == "__main__":
    save_or_show(draw(), "profiling_workflow")
