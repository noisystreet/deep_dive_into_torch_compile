"""
Triton 编译器流水线：Python AST → SASS
"""

from style import setup_figure, save_or_show, box, arrow, label, COLORS

COLORS.update(
    {
        "blue": "#4a9eff",
        "green": "#6abf69",
        "orange": "#ffa94d",
        "pink": "#d94f8a",
        "purple": "#b07cd8",
    }
)


def draw():
    fig, ax = setup_figure(width=11, height=4.5)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)

    stages = [
        (0.3, 2.5, 1.8, 1.4, "Python AST\n@triton.jit 函数", COLORS["blue"]),
        (2.4, 2.5, 1.8, 1.4, "类型推断\nType Inference", COLORS["green"]),
        (4.5, 2.5, 1.8, 1.4, "TTIR 生成\nTriton IR", COLORS["orange"]),
        (
            6.6,
            2.5,
            1.8,
            1.4,
            "TTIR 优化\n循环展开/常量折叠\n内存合并分析",
            COLORS["orange"],
        ),
        (8.7, 2.5, 1.8, 1.4, "PTX 生成\nTensorCore 指令", COLORS["pink"]),
    ]
    # Second row
    stages2 = [
        (1.35, 0.5, 2.0, 1.2, "ptxas\nNVIDIA 汇编器", COLORS["pink"]),
        (3.8, 0.5, 2.0, 1.2, "SASS\nGPU 机器码", COLORS["purple"]),
        (6.2, 0.5, 2.0, 1.2, "cubin\n可执行 kernel", COLORS["purple"]),
    ]

    for x, y, w, h, text, c in stages:
        box(ax, x, y, w, h, text, color=c, fontsize=7)
    for x, y, w, h, text, c in stages2:
        box(ax, x, y, w, h, text, color=c, fontsize=7)

    # Main flow arrows (top row)
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + stages[i][2]
        arrow(ax, x1, stages[i][1] + 0.7, x1 + 0.3, stages[i][1] + 0.7)

    # Down arrows to second row
    arrow(ax, 2.35, 2.5, 2.35, 1.7)
    arrow(ax, 4.8, 2.5, 4.8, 1.7)

    # Second row arrows
    arrow(ax, 3.35, 1.1, 3.8, 1.1)
    arrow(ax, 5.8, 1.1, 6.2, 1.1)

    # Phase labels
    label(
        ax, 0.7, 4.3, "Triton 编译器前端", fontsize=8, bold=True, color=COLORS["blue"]
    )
    label(ax, 5.8, 4.3, "设备相关优化", fontsize=8, bold=True, color=COLORS["orange"])
    label(ax, 9.9, 4.3, "NVIDIA 后端", fontsize=8, bold=True, color=COLORS["pink"])

    # Phase dividers
    for x in [4.0, 8.0]:
        ax.plot(
            [x, x], [0.2, 4.8], color=COLORS["light_gray"], linewidth=1.5, linestyle=":"
        )

    return fig


if __name__ == "__main__":
    save_or_show(draw(), "triton_compiler_pipeline")
