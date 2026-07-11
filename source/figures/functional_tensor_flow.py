"""
FunctionalTensor 拦截 in-place 操作流程图
"""

from style import setup_figure, save_or_show, box, arrow, label, COLORS


def draw():
    fig, ax = setup_figure(width=7, height=6)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 7.5)

    label(
        ax,
        4.5,
        7.2,
        "FunctionalTensor 拦截 in-place 操作",
        fontsize=13,
        bold=True,
        color=COLORS["aot"],
    )

    steps = [
        (2.5, 5.8, 4.0, 0.8, "x.add_(1)\n← 被 FunctionalTensor 拦截", COLORS["dynamo"]),
        (2.5, 4.6, 4.0, 0.7, "FunctionalTensor 内部处理", COLORS["aot"]),
        (0.5, 3.2, 3.5, 0.7, "1. 读取 x 的当前值", COLORS["inductor"]),
        (
            5.0,
            3.2,
            3.5,
            0.7,
            "2. 执行 out-of-place\ntorch.add(x, 1)",
            COLORS["inductor"],
        ),
        (0.5, 1.8, 3.5, 0.7, "3. 将 x 的内部存储\n替换为计算结果", COLORS["inductor"]),
        (5.0, 1.8, 3.5, 0.7, "4. 递增版本号", COLORS["inductor"]),
    ]
    for x, y, w, h, text, color in steps:
        box(ax, x, y, w, h, text, color=color, fontsize=8 if "\n" in text else 9)

    # 箭头
    for x1, y1, x2, y2 in [
        (4.5, 5.8, 4.5, 5.3),
        (2.25, 4.6, 2.25, 3.9),
        (6.75, 4.6, 6.75, 3.9),
    ]:
        arrow(ax, x1, y1, x2, y2)

    # 并行分支箭头 (1←2, 3←4)
    label(ax, 3.8, 3.55, "并行", fontsize=7, color=COLORS["gray"])
    arrow(ax, 4.0, 3.55, 5.0, 3.55, color=COLORS["gray"], lw=1)

    arrow(ax, 2.25, 3.2, 2.25, 2.5)
    arrow(ax, 6.75, 3.2, 6.75, 2.5)
    arrow(ax, 2.25, 1.8, 4.0, 1.3)
    arrow(ax, 6.75, 1.8, 5.0, 1.3)

    # 结果
    box(
        ax,
        2.0,
        0.2,
        5.0,
        0.8,
        "从外部看: x 的值被更新了\nFX Graph 记录: torch.add(x, 1)（out-of-place）",
        color=COLORS["triton"],
        fontsize=7,
    )

    arrow(ax, 4.5, 1.1, 4.5, 1.0)

    return fig


if __name__ == "__main__":
    save_or_show(draw(), "functional_tensor_flow")
