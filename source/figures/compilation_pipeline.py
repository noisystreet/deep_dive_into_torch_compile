"""
torch.compile 编译流水线架构图
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from style import setup_figure, save_or_show, box, arrow, label, legend, COLORS


def draw():
    fig, ax = setup_figure(width=9, height=6)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)

    # === 四列标题 ===
    cols = [
        (0.5, "用户代码", COLORS["gray"]),
        (3.0, "Dynamo", COLORS["dynamo"]),
        (5.5, "AOTAutograd", COLORS["aot"]),
        (8.0, "Inductor", COLORS["inductor"]),
    ]
    for x, text, color in cols:
        label(
            ax,
            x,
            6.2,
            text,
            fontsize=12,
            bold=True,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=color,
                alpha=0.12,
                edgecolor=color,
                linewidth=1.5,
            ),
        )
        ax.plot(
            [x, x],
            [0.3, 5.8],
            color=COLORS["light_gray"],
            linewidth=1.5,
            linestyle="--",
        )

    # === 箭头流 ===
    flows = [
        (5.4, 0.5, 3.0, "torch.compile(fn)", COLORS["arrow"], False),
        (5.0, 3.0, 0.5, "注册回调", COLORS["dynamo"], True),
        (4.5, 0.5, 3.0, "compiled_fn()", COLORS["arrow"], False),
        (4.3, 3.0, 3.0, "convert_frame (FX Graph)", COLORS["dynamo"], False),
        (3.3, 3.0, 5.5, "lookup_backend", COLORS["dynamo"], False),
        (2.7, 5.5, 5.5, "aot_autograd (Joint Graph)", COLORS["aot"], False),
        (2.1, 5.5, 8.0, "compile_fx", COLORS["aot"], False),
        (1.4, 8.0, 8.0, "lowering → scheduler → codegen", COLORS["inductor"], False),
        (0.9, 8.0, 5.5, "cached_fn", COLORS["inductor"], True),
        (0.6, 5.5, 3.0, "cached_fn", COLORS["aot"], True),
        (0.3, 3.0, 0.5, "cached_fn", COLORS["dynamo"], True),
    ]
    for y, x1, x2, text, color, dashed in flows:
        if x1 == x2:
            arrow(
                ax,
                x1,
                y - 0.05,
                x2 + 0.8,
                y,
                color=color,
                lw=1.5,
                style="->" if not dashed else "->",
            )
            label(ax, x1 + 1.2, y + 0.1, text, fontsize=8, ha="left")
        else:
            arrow(ax, x1, y, x2, y, color=color, lw=1.5, dashed=dashed)
            label(ax, (x1 + x2) / 2, y + 0.15, text, fontsize=8)

    # 后续调用阶段
    ax.axhspan(0, 0.2, alpha=0.08, color=COLORS["gray"])
    label(
        ax,
        4.5,
        0.1,
        "后续调用: guard check → hit cache → 直接返回",
        fontsize=9,
        color=COLORS["gray"],
        italic=True,
    )

    legend(
        ax,
        [
            (COLORS["dynamo"], "Dynamo"),
            (COLORS["aot"], "AOTAutograd"),
            (COLORS["inductor"], "Inductor"),
        ],
    )
    return fig


if __name__ == "__main__":
    save_or_show(draw(), "compilation_pipeline")
