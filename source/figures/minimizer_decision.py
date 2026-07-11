"""
Minimizer 模式选择与二分搜索算法
"""

from style import setup_figure, save_or_show, box, arrow, label, diamond, COLORS


def draw():
    fig, ax = setup_figure(width=8, height=5.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)

    # === 上：决策树 ===
    label(ax, 5, 6.2, "Minimizer 模式选择", fontsize=13, bold=True, color=COLORS["aot"])

    diamond(ax, 5.0, 5.3, 0.7, "编译失败?")
    arrow(ax, 5.0, 4.9, 5.0, 4.4)

    diamond(ax, 5.0, 4.0, 0.7, "选择\nMinimizer 模式")
    arrow(ax, 3.6, 4.0, 2.2, 4.0)
    arrow(ax, 6.4, 4.0, 7.8, 4.0)

    box(ax, 0.3, 3.3, 3.0, 0.5, "dynamo 模式", color=COLORS["dynamo"], fontsize=7)
    box(ax, 6.7, 3.3, 3.0, 0.5, "aot 模式", color=COLORS["inductor"], fontsize=7)

    arrow(ax, 1.8, 3.3, 1.8, 2.6)
    arrow(ax, 8.2, 3.3, 8.2, 2.6)

    diamond(ax, 1.8, 2.1, 0.6, "Dynamo\n报错?")
    diamond(ax, 8.2, 2.1, 0.6, "Inductor\n报错?")

    arrow(ax, 1.2, 2.1, 0.3, 2.1)
    arrow(ax, 2.4, 2.1, 3.5, 2.1)
    arrow(ax, 7.6, 2.1, 6.5, 2.1)
    arrow(ax, 8.8, 2.1, 9.7, 2.1)

    label(ax, 0.5, 2.1, "是→定位\nPython 代码", fontsize=6, color=COLORS["dynamo"])
    label(ax, 3.5, 2.1, "否→尝试 aot", fontsize=6, color=COLORS["gray"], italic=True)
    label(ax, 6.5, 2.1, "是→定位\nFX 子图", fontsize=6, color=COLORS["inductor"])
    label(ax, 9.7, 2.1, "否→检查\nCUDA 错误", fontsize=6, color=COLORS["gray"])

    # === 下：二分搜索 ===
    ax.axhspan(0, 1.5, 0, 1, facecolor=COLORS["aot"], alpha=0.04)
    label(
        ax,
        5,
        1.4,
        "二分搜索算法：在最少的步骤中定位有问题的节点",
        fontsize=8,
        color=COLORS["aot"],
    )

    # Bisect steps
    bisect_steps = [
        (0.5, 0.6, 2.0, 0.5, "完整 FX Graph\nN 个节点", COLORS["gray"]),
        (3.0, 0.6, 2.0, 0.5, "前半 eager\n后半 compiled", COLORS["gray"]),
        (5.5, 0.6, 2.0, 0.5, "测试是否\n复现错误", COLORS["aot"]),
        (8.0, 0.6, 1.8, 0.5, "缩小范围\n二分迭代", COLORS["dynamo"]),
    ]
    for x, y, w, h, text, c in bisect_steps:
        box(ax, x, y, w, h, text, color=c, fontsize=6, bold=False)
    arrow(ax, 2.5, 0.85, 3.0, 0.85)
    arrow(ax, 5.0, 0.85, 5.5, 0.85)
    arrow(ax, 7.5, 0.85, 8.0, 0.85)
    # 回环箭头
    arrow(ax, 8.9, 0.85, 9.2, 0.85, style="->")
    label(ax, 9.7, 0.85, "输出\n最小节点", fontsize=6, color=COLORS["triton"])

    return fig


if __name__ == "__main__":
    save_or_show(draw(), "minimizer_decision")
