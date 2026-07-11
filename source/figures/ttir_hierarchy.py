"""
TTIR 分层架构图
"""

import matplotlib.patches as mpatches
from style import setup_figure, save_or_show, arrow, label, COLORS


def draw():
    fig, ax = setup_figure(width=8, height=5.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)

    label(ax, 5, 6.7, "TTIR 分层架构", fontsize=14, bold=True, color=COLORS["aot"])

    layers = [
        (
            2.5,
            5.0,
            5.0,
            1.4,
            "TTIR (Triton Dialect)\nttir.add · ttir.load · ttir.dot\nttir.store · ttir.reduce",
            "#4a9eff",
        ),
        (
            2.5,
            3.2,
            5.0,
            1.4,
            "TTGIR (TritonGPU Dialect)\n添加 GPU 特定信息\nwarp 映射 · shared memory · 数据布局",
            "#ffa94d",
        ),
        (
            2.5,
            1.6,
            5.0,
            1.2,
            "LLVM Dialect\nllvm.add · llvm.load · llvm.store",
            "#6abf69",
        ),
        (2.5, 0.3, 5.0, 1.0, "PTX\nNVIDIA 指令集", "#d94f8a"),
    ]

    for x, y, w, h, text, c in layers:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=c,
            alpha=0.12,
            edgecolor=c,
            linewidth=2.5,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=8,
            color=c,
            fontweight="bold",
        )

    arrow(ax, 5.0, 5.0, 5.0, 4.6)
    arrow(ax, 5.0, 3.2, 5.0, 2.8)
    arrow(ax, 5.0, 1.6, 5.0, 1.3)

    label(ax, 8.5, 5.7, "设备无关", fontsize=9, bold=True, color="#4a9eff")
    label(ax, 8.5, 3.9, "设备相关", fontsize=9, bold=True, color="#ffa94d")
    label(ax, 8.5, 2.2, "后端无关", fontsize=9, bold=True, color="#6abf69")
    label(ax, 8.5, 0.8, "后端特定", fontsize=9, bold=True, color="#d94f8a")

    return fig


if __name__ == "__main__":
    save_or_show(draw(), "ttir_hierarchy")
