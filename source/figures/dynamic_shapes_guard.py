"""
静态形状 vs 动态形状的 Guard 行为对比
"""

from style import setup_figure, save_or_show, box, arrow, label, diamond, COLORS


def draw():
    fig, ax = setup_figure(width=9, height=5.5)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)

    label(
        ax,
        6,
        6.2,
        "静态形状 vs 动态形状 Guard 行为",
        fontsize=13,
        bold=True,
        color=COLORS["aot"],
    )

    # === 左侧：静态形状 ===
    ax.axhspan(0, 5.5, 0, 0.47, facecolor=COLORS["dynamo"], alpha=0.04)
    label(
        ax, 2.8, 5.5, "静态形状 (默认)", fontsize=10, bold=True, color=COLORS["dynamo"]
    )

    box(
        ax,
        0.5,
        4.3,
        4.6,
        0.6,
        "输入 shape=(32, 784)",
        color=COLORS["dynamo"],
        fontsize=7,
    )
    arrow(ax, 2.8, 4.3, 2.8, 3.7)

    box(
        ax,
        0.5,
        3.0,
        4.6,
        0.6,
        "Guard: x.shape[0] == 32\n       x.shape[1] == 784",
        color=COLORS["dynamo"],
        fontsize=6,
    )
    arrow(ax, 2.8, 3.0, 2.8, 2.3)

    diamond(ax, 2.8, 1.8, 0.7, "下次\n输入\nshape=\n(64,784)?")
    arrow(ax, 2.1, 1.8, 1.0, 1.8)
    arrow(ax, 3.5, 1.8, 4.6, 1.8)

    box(
        ax,
        0.3,
        0.8,
        2.0,
        0.6,
        "Guard 失败\n→ 重新编译",
        color=COLORS["gray"],
        fontsize=6,
    )
    box(
        ax,
        3.7,
        0.8,
        2.0,
        0.6,
        "shape=(32, 784)\n→ 命中缓存",
        color=COLORS["gray"],
        fontsize=6,
    )

    # === 右侧：动态形状 ===
    ax.axhspan(0, 5.5, 0.53, 1, facecolor=COLORS["inductor"], alpha=0.04)
    label(
        ax,
        9.2,
        5.5,
        "动态形状 (dynamic=True)",
        fontsize=10,
        bold=True,
        color=COLORS["inductor"],
    )

    box(
        ax,
        6.9,
        4.3,
        4.6,
        0.6,
        "输入 shape=(32, 784)",
        color=COLORS["inductor"],
        fontsize=7,
    )
    arrow(ax, 9.2, 4.3, 9.2, 3.7)

    box(
        ax,
        6.9,
        3.0,
        4.6,
        0.6,
        "Guard: x.shape[0] >= 1\n       x.shape[1] == 784",
        color=COLORS["inductor"],
        fontsize=6,
    )
    arrow(ax, 9.2, 3.0, 9.2, 2.3)

    diamond(ax, 9.2, 1.8, 0.7, "下次\n输入\nshape=\n(64,784)?")
    arrow(ax, 8.5, 1.8, 7.4, 1.8)
    arrow(ax, 9.9, 1.8, 11.0, 1.8)

    box(
        ax,
        6.7,
        0.8,
        2.0,
        0.6,
        "Guard 通过\n→ 命中缓存",
        color=COLORS["gray"],
        fontsize=6,
    )
    box(
        ax,
        10.0,
        0.8,
        2.0,
        0.6,
        "shape[0] ≥ 1\n→ 无需重编译",
        color=COLORS["gray"],
        fontsize=6,
    )

    # Labels
    label(ax, 1.0, 1.8, "否", fontsize=7, color=COLORS["gray"])
    label(ax, 4.6, 1.8, "是", fontsize=7, color=COLORS["gray"])
    label(ax, 7.4, 1.8, "否", fontsize=7, color=COLORS["gray"])
    label(ax, 11.0, 1.8, "是", fontsize=7, color=COLORS["gray"])

    return fig


if __name__ == "__main__":
    save_or_show(draw(), "dynamic_shapes_guard")
