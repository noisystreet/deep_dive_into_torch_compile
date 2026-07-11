"""
Eager Autograd vs AOTAutograd 对比图
"""

from style import setup_figure, save_or_show, box, arrow, label, COLORS


def draw():
    fig, ax = setup_figure(width=9, height=5)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    # 左侧：Eager Background
    bg_eager = ax.axhspan(0, 5, 0, 0.48, facecolor=COLORS["dynamo"], alpha=0.04)
    box(ax, 0.8, 5.2, 4.0, 0.6, "Eager Autograd", color=COLORS["dynamo"], fontsize=11)
    label(
        ax, 2.8, 5.5, "Eager Autograd", fontsize=11, bold=True, color=COLORS["dynamo"]
    )

    steps_eager = [
        (0.5, 3.8, 4.6, 0.8, "前向执行\n实时写 tape", COLORS["dynamo"]),
        (0.5, 2.5, 4.6, 0.8, "backward\n逐段解释执行", COLORS["dynamo"]),
        (0.5, 0.8, 4.6, 1.2, "编译器看不到完整\n前向+反向", COLORS["gray"]),
    ]
    for x, y, w, h, text, c in steps_eager:
        box(ax, x, y, w, h, text, color=c, fontsize=8)
    arrow(ax, 2.8, 3.8, 2.8, 3.3)

    # 右侧：AOTAutograd Background
    bg_aot = ax.axhspan(0, 5, 0.52, 1, facecolor=COLORS["inductor"], alpha=0.04)
    box(ax, 6.6, 5.2, 4.8, 0.6, "AOTAutograd", color=COLORS["inductor"], fontsize=11)
    label(ax, 9.0, 5.5, "AOTAutograd", fontsize=11, bold=True, color=COLORS["inductor"])

    steps_aot = [
        (6.6, 3.8, 4.8, 0.8, "编译期 trace 联合图", COLORS["inductor"]),
        (6.6, 2.5, 4.8, 0.8, "图分区", COLORS["inductor"]),
        (6.6, 1.2, 4.8, 0.8, "分别交给 Inductor", COLORS["inductor"]),
    ]
    for x, y, w, h, text, c in steps_aot:
        box(ax, x, y, w, h, text, color=c, fontsize=8)
    arrow(ax, 9.0, 3.8, 9.0, 3.3)
    arrow(ax, 9.0, 2.5, 9.0, 2.0)

    box(
        ax,
        6.6,
        0.3,
        4.8,
        0.7,
        "编译器看到全局\n可做跨前向/反向优化",
        color=COLORS["triton"],
        fontsize=8,
    )
    arrow(ax, 9.0, 1.2, 9.0, 1.0)

    return fig


if __name__ == "__main__":
    save_or_show(draw(), "eager_vs_aotautograd")
