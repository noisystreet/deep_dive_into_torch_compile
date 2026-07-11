"""
异步 Autotune 过程：主进程 + AutoTuneProcess
"""

import matplotlib.patches as mpatches
from style import setup_figure, save_or_show, box, arrow, label, COLORS


def draw():
    fig, ax = setup_figure(width=10, height=6)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)

    label(
        ax, 6, 7.2, "异步 Autotune 过程", fontsize=14, bold=True, color=COLORS["triton"]
    )

    # Left column: Main Process
    box(ax, 0.5, 6.0, 4.0, 0.7, "主进程", color=COLORS["inductor"], fontsize=10)

    main_steps = [
        (
            0.5,
            5.0,
            4.0,
            0.7,
            "提交 autotune 请求\n(kernel 模板 + 输入信息)",
            COLORS["inductor"],
        ),
        (
            0.5,
            1.2,
            4.0,
            2.5,
            "继续执行其他 kernel\n的代码生成和编译",
            COLORS["inductor"],
        ),
    ]
    for x, y, w, h, text, c in main_steps:
        box(ax, x, y, w, h, text, color=c, fontsize=7)
    arrow(ax, 2.5, 6.0, 2.5, 5.7)

    # Right column: AutoTuneProcess
    box(ax, 7.5, 6.0, 4.0, 0.7, "AutoTuneProcess", color=COLORS["triton"], fontsize=10)
    box(
        ax,
        7.5,
        4.3,
        4.0,
        0.7,
        "生成 config 组合列表",
        color=COLORS["triton"],
        fontsize=7,
    )
    arrow(ax, 9.5, 6.0, 9.5, 5.0)

    # Loop box
    loop_rect = mpatches.FancyBboxPatch(
        (7.2, 1.5),
        4.6,
        2.3,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["aot"],
        alpha=0.06,
        edgecolor=COLORS["aot"],
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(loop_rect)
    label(ax, 9.5, 3.6, "每个 config", fontsize=8, bold=True, color=COLORS["aot"])

    loop_steps = [
        (7.5, 2.8, 4.0, 0.5, "编译 kernel（特定 config）", COLORS["triton"]),
        (7.5, 2.0, 4.0, 0.5, "执行基准测试 → 延迟数据", COLORS["triton"]),
    ]
    for x, y, w, h, text, c in loop_steps:
        box(ax, x, y, w, h, text, color=c, fontsize=6)
    arrow(ax, 9.5, 2.8, 9.5, 2.5)

    # Return arrow
    arrow(ax, 9.5, 1.5, 2.5, 1.2, color=COLORS["arrow"], lw=1.5)
    label(
        ax, 6.0, 1.2, "返回最优 config", fontsize=7, color=COLORS["arrow"], ha="center"
    )

    # Legend
    box(
        ax,
        4.5,
        4.7,
        3.0,
        0.5,
        "Triton Compiler",
        color=COLORS["triton"],
        fontsize=6,
        bold=False,
        alpha=0.06,
    )

    return fig


if __name__ == "__main__":
    save_or_show(draw(), "autotune_process")
