"""
调试报告生成流程：TORCH_COMPILE_DEBUG 追踪图
"""

from style import setup_figure, save_or_show, box, arrow, label, COLORS


def draw():
    fig, ax = setup_figure(width=10, height=6.5)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    label(ax, 6, 7.7, "调试报告生成流程", fontsize=14, bold=True, color=COLORS["aot"])

    # Participants (columns)
    participants = [
        (1.0, "用户代码", COLORS["dynamo"]),
        (3.5, "TORCH_COMPILE_DEBUG", COLORS["gray"]),
        (5.5, "TorchDynamo", COLORS["dynamo"]),
        (7.5, "TorchInductor", COLORS["inductor"]),
        (9.5, "报告生成器", COLORS["aot"]),
    ]
    for x, text, color in participants:
        box(ax, x - 0.8, 6.8, 1.6, 0.7, text, color=color, fontsize=7)

    # Flow arrows (y, x1, x2, text, color)
    flows = [
        (6.3, 1.0, 3.5, "设置环境变量", COLORS["dynamo"]),
        (5.7, 3.5, 5.5, "详细记录模式", COLORS["gray"]),
        (5.2, 3.5, 7.5, "详细记录模式", COLORS["gray"]),
        (4.7, 5.5, 9.5, "Graph Break 位置", COLORS["dynamo"]),
        (4.2, 5.5, 9.5, "Guard 表达式", COLORS["dynamo"]),
        (3.7, 5.5, 9.5, "子图 FX Graph", COLORS["dynamo"]),
        (3.2, 7.5, 9.5, "Lowering 过程", COLORS["inductor"]),
        (2.7, 7.5, 9.5, "融合决策", COLORS["inductor"]),
        (2.2, 7.5, 9.5, "生成的 Kernel 代码", COLORS["inductor"]),
        (1.5, 9.5, 1.0, "输出文件（HTML/TXT/PY）", COLORS["aot"]),
    ]
    for y, x1, x2, text, color in flows:
        arrow(ax, x1, y, x2, y, color=color, lw=1.5, style="->")
        label(ax, (x1 + x2) / 2, y + 0.15, text, fontsize=6, ha="center")

    # Vertical swimlane lines
    for x, _, _ in participants:
        ax.plot(
            [x, x], [0.3, 6.5], color=COLORS["light_gray"], linewidth=1, linestyle=":"
        )

    # Output files at bottom
    files = [
        "torchdynamo_debug.html",
        "inductor.html",
        "fx_graph_readable.txt",
        "fx_graph_runnable.py",
        "replay.py",
    ]
    for i, f in enumerate(files):
        box(
            ax,
            0.5 + i * 2.2,
            0.2,
            2.0,
            0.4,
            f,
            color=COLORS["triton"],
            fontsize=6,
            bold=False,
        )

    return fig


if __name__ == "__main__":
    save_or_show(draw(), "debug_report_flow")
