"""
AOTAutograd 完整处理流程：功能化 → 联合图 → 分区 → 编译 → 运行时包装
"""

from style import setup_figure, save_or_show, box, arrow, label, COLORS


def draw():
    fig, ax = setup_figure(width=12, height=7)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)

    label(
        ax,
        7,
        8.5,
        "AOTAutograd 训练函数处理流程",
        fontsize=14,
        bold=True,
        color=COLORS["aot"],
    )

    # Top: stages (8 participants)
    stages = [
        (0.3, 7.0, "用户\n训练函数", COLORS["dynamo"]),
        (2.0, 7.0, "功能化", COLORS["aot"]),
        (3.7, 7.0, "准备\nautograd", COLORS["aot"]),
        (5.4, 7.0, "创建\n联合图", COLORS["aot"]),
        (7.1, 7.0, "make_fx\n追踪", COLORS["aot"]),
        (8.8, 7.0, "图分区", COLORS["aot"]),
        (10.5, 7.0, "Inductor\n编译", COLORS["inductor"]),
        (12.2, 7.0, "运行时\n包装", COLORS["triton"]),
    ]
    for x, y, text, c in stages:
        box(ax, x - 0.5, y, 1.6, 0.8, text, color=c, fontsize=6)

    # Arrows between stages (row 1)
    arrow(ax, 1.9, 7.4, 2.0, 7.4)
    arrow(ax, 3.6, 7.4, 3.7, 7.4)
    arrow(ax, 5.3, 7.4, 5.4, 7.4)
    arrow(ax, 7.0, 7.4, 7.1, 7.4)
    arrow(ax, 8.7, 7.4, 8.8, 7.4)
    arrow(ax, 10.4, 7.4, 10.5, 7.4)
    arrow(ax, 12.1, 7.4, 12.2, 7.4)

    # Step annotations below each stage (y=5.8)
    annotations = [
        (1.1, 5.8, "原始函数 fn\nfn", 0),
        (2.8, 5.8, "fn_func\n功能化后的函数", 0),
        (4.5, 5.8, "fn_prepped", 0),
        (6.2, 5.8, "joint_fn", 0),
        (7.9, 5.8, "fx_g\njoint graph", 0),
        (9.7, 5.8, "fwd_module\nbwd_module", 0),
        (11.3, 5.8, "compiled_fwd\ncompiled_bwd", 0),
        (13.0, 5.8, "AOTDispatch\nAutograd", 0),
    ]
    for x, y, text, _ in annotations:
        box(
            ax,
            x - 0.6,
            y - 0.05,
            1.5,
            0.5,
            text,
            color=COLORS["gray"],
            fontsize=6,
            bold=False,
            alpha=0.08,
        )

    # Notes below each stage (y=4.2)
    notes = [
        (1.1, 4.0, "将 in-place 操作\n转换为纯函数式"),
        (2.8, 4.0, "准备前向输出的\ntangent_mask"),
        (4.5, 4.0, "用 autograd.grad\n追踪前向和反向"),
        (6.2, 4.0, "用 proxy tensor\n生成 joint FX Graph"),
        (7.9, 4.0, "min-cut 分区\n前向/反向子图"),
        (9.7, 4.0, "分别编译\n前向和反向"),
        (11.3, 4.0, "管理执行流和\nsaved tensors"),
    ]
    for x, y, text in notes:
        if text:
            box(
                ax,
                x - 0.7,
                y,
                1.8,
                0.7,
                text,
                color=COLORS["aot"],
                fontsize=6,
                bold=False,
                alpha=0.06,
            )

    # Down arrows from stage → annotation
    for s, a in zip(stages, annotations):
        ax.plot(
            [s[0], s[0]],
            [s[1], a[1] + 0.45],
            color=COLORS["arrow"],
            linewidth=1,
            linestyle=":",
        )

    # Down arrows from annotation → note
    for a, n in zip(annotations, notes):
        if n:
            ax.plot(
                [a[0], a[0]],
                [a[1] - 0.1, n[1] + 0.35],
                color=COLORS["arrow"],
                linewidth=1,
                linestyle=":",
            )

    return fig


if __name__ == "__main__":
    save_or_show(draw(), "aot_full_flow")
