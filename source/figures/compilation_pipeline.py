"""
torch.compile 编译流水线架构图
替代 mermaid sequenceDiagram，用 matplotlib 绘制更专业的序列图
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from style import setup_figure, COLORS
import matplotlib.patches as mpatches


def draw():
    fig, ax = setup_figure(width=9, height=6)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # === 四列标题 ===
    cols = [
        (0.5, "用户代码", COLORS["gray"]),
        (3.0, "Dynamo", COLORS["dynamo"]),
        (5.5, "AOTAutograd", COLORS["aot"]),
        (8.0, "Inductor", COLORS["inductor"]),
    ]
    for x, label, color in cols:
        ax.text(
            x,
            6.2,
            label,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=color,
                alpha=0.12,
                edgecolor=color,
                linewidth=1.5,
            ),
        )

    # 竖线（生命线）
    for x, _, _ in cols:
        ax.plot(
            [x, x],
            [0.3, 5.8],
            color=COLORS["light_gray"],
            linewidth=1.5,
            linestyle="--",
        )

    # === 箭头和标注（按时间从上到下）===
    arrows = [
        # (y, x1, x2, label, color, style)
        (5.4, 0.5, 3.0, "torch.compile(fn)", COLORS["arrow"], "->"),
        (5.0, 3.0, 0.5, "注册回调", COLORS["dynamo"], "-->"),
        (4.5, 0.5, 3.0, "compiled_fn()", COLORS["arrow"], "->"),
        (3.9, 3.0, 3.0, "convert_frame\n(FX Graph)", COLORS["dynamo"], "-"),
        (3.3, 3.0, 5.5, "lookup_backend", COLORS["dynamo"], "->"),
        (2.7, 5.5, 5.5, "aot_autograd\n(Joint Graph)", COLORS["aot"], "-"),
        (2.1, 5.5, 8.0, "compile_fx", COLORS["aot"], "->"),
        (1.5, 8.0, 8.0, "lowering → scheduler\n→ codegen", COLORS["inductor"], "-"),
        (0.9, 8.0, 5.5, "cached_fn", COLORS["inductor"], "-->"),
        (0.6, 5.5, 3.0, "cached_fn", COLORS["aot"], "-->"),
        (0.3, 3.0, 0.5, "cached_fn", COLORS["dynamo"], "-->"),
    ]

    for y, x1, x2, label, color, style in arrows:
        # 箭头
        dx = x2 - x1
        if style == "->":
            ax.annotate(
                "",
                xy=(x2, y),
                xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            )
        elif style == "-->":
            ax.annotate(
                "",
                xy=(x2, y),
                xytext=(x1, y),
                arrowprops=dict(
                    arrowstyle="->", color=color, lw=1.5, linestyle="dashed"
                ),
            )
        elif style == "-":
            pass  # 自我循环

        # 标注
        mid_x = (x1 + x2) / 2
        # 自我循环标注在右侧
        if x1 == x2:
            ax.text(
                x1 + 0.6,
                y + 0.15,
                label,
                ha="left",
                va="center",
                fontsize=8,
                color=color,
            )
        else:
            ax.text(
                mid_x,
                y + 0.15,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )

    # 后续调用阶段（灰色区域）
    ax.axhspan(0, 0.2, alpha=0.08, color=COLORS["gray"])
    ax.text(
        4.5,
        0.1,
        "后续调用: guard check → hit cache → 直接返回",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["gray"],
        style="italic",
    )

    # === 图例 ===
    legend_elements = [
        mpatches.Patch(
            facecolor=COLORS["dynamo"],
            alpha=0.12,
            edgecolor=COLORS["dynamo"],
            label="Dynamo",
        ),
        mpatches.Patch(
            facecolor=COLORS["aot"],
            alpha=0.12,
            edgecolor=COLORS["aot"],
            label="AOTAutograd",
        ),
        mpatches.Patch(
            facecolor=COLORS["inductor"],
            alpha=0.12,
            edgecolor=COLORS["inductor"],
            label="Inductor",
        ),
    ]
    ax.legend(
        handles=legend_elements, loc="lower center", ncol=3, fontsize=9, frameon=False
    )

    return fig


if __name__ == "__main__":
    fig = draw()
    out = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "../source/_static/figures/compilation_pipeline.svg"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"Saved: {out}")
