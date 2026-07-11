"""
朴素分区 vs Min-Cut 分区对比图
"""

from style import setup_figure, save_or_show, box, arrow, label, COLORS


def draw():
    fig, ax = setup_figure(width=9, height=5)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)

    label(
        ax,
        6,
        6.2,
        "朴素分区 vs Min-Cut 分区",
        fontsize=14,
        bold=True,
        color=COLORS["aot"],
    )

    # === 左侧：朴素分区 ===
    ax.axhspan(0, 5.5, 0, 0.47, facecolor=COLORS["dynamo"], alpha=0.04)
    label(ax, 2.8, 5.3, "朴素分区", fontsize=11, bold=True, color=COLORS["dynamo"])

    naive_steps = [
        (0.3, 4.3, 5.0, 0.6, "前向保存所有中间结果", COLORS["gray"]),
        (0.6, 3.2, 2.0, 0.6, "exp(x): 保存", COLORS["dynamo"]),
        (1.6, 2.1, 2.4, 0.6, "cos(exp(x)): 保存", COLORS["dynamo"]),
        (2.6, 1.0, 2.4, 0.6, "sin(cos(exp(x))): 保存", COLORS["dynamo"]),
    ]
    for x, y, w, h, text, c in naive_steps:
        box(ax, x, y, w, h, text, color=c, fontsize=7)
    box(
        ax,
        0.6,
        0.1,
        4.4,
        0.6,
        "反向直接使用所有保存值",
        color=COLORS["gray"],
        fontsize=7,
        bold=False,
    )
    arrow(ax, 2.8, 4.3, 2.8, 3.8)
    arrow(ax, 1.6, 3.2, 1.6, 2.7)
    arrow(ax, 2.8, 2.1, 2.8, 1.6)
    arrow(ax, 3.8, 1.0, 3.8, 0.7)
    ax.plot(
        [1.6, 1.6, 3.8],
        [0.6, 0.4, 0.4],
        color=COLORS["arrow"],
        linewidth=1,
        linestyle=":",
    )
    ax.plot(
        [2.8, 2.8, 3.8],
        [0.6, 0.4, 0.4],
        color=COLORS["arrow"],
        linewidth=1,
        linestyle=":",
    )

    # === 右侧：Min-Cut 分区 ===
    ax.axhspan(0, 5.5, 0.53, 1, facecolor=COLORS["inductor"], alpha=0.04)
    label(
        ax, 9.2, 5.3, "Min-Cut 分区", fontsize=11, bold=True, color=COLORS["inductor"]
    )

    mincut_steps = [
        (6.3, 4.3, 5.4, 0.6, "前向选择性保存", COLORS["gray"]),
        (6.6, 3.2, 2.4, 0.6, "exp(x): 保存\n（计算昂贵）", COLORS["inductor"]),
        (7.0, 2.1, 2.8, 0.6, "cos(exp(x)): 丢弃\n（反向重计算）", COLORS["dynamo"]),
        (8.4, 1.0, 2.8, 0.6, "sin(cos(exp(x))): 保存", COLORS["inductor"]),
    ]
    for x, y, w, h, text, c in mincut_steps:
        box(ax, x, y, w, h, text, color=c, fontsize=7)
    box(
        ax,
        6.3,
        0.1,
        5.2,
        0.6,
        "反向: exp→cos→ 梯度链",
        color=COLORS["gray"],
        fontsize=7,
        bold=False,
    )
    box(
        ax,
        8.4,
        -0.35,
        2.8,
        0.3,
        "sin→ 直接使用",
        color=COLORS["gray"],
        fontsize=6,
        bold=False,
        alpha=0.06,
    )

    arrow(ax, 9.0, 4.3, 9.0, 3.8)
    arrow(ax, 7.8, 3.2, 7.8, 2.7)
    arrow(ax, 9.8, 1.0, 9.8, 0.7)

    ax.plot(
        [7.8, 7.8, 9.8],
        [0.6, 0.4, 0.4],
        color=COLORS["arrow"],
        linewidth=1,
        linestyle=":",
    )
    ax.plot(
        [9.8, 9.8, 9.8],
        [0.4, -0.2, -0.35],
        color=COLORS["arrow"],
        linewidth=1,
        linestyle=":",
    )
    # 重计算标注
    arrow(ax, 8.4, 2.1, 8.4, 1.7, color=COLORS["dynamo"], lw=1.5, style="->")
    label(ax, 4.6, 1.2, "重计算", fontsize=7, color=COLORS["dynamo"], italic=True)

    return fig


if __name__ == "__main__":
    save_or_show(draw(), "min_cut_partition")
