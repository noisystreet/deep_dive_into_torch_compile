"""
统一 Matplotlib 风格与绘图原语：浅入深出 torch.compile
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys

matplotlib.rcParams.update(
    {
        "font.family": ["Noto Sans CJK JP", "DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "svg.fonttype": "path",
    }
)

COLORS = {
    "dynamo": "#E2882C",
    "aot": "#2C5F8A",
    "inductor": "#4CAF50",
    "triton": "#9C27B0",
    "gray": "#757575",
    "light_gray": "#E0E0E0",
    "light_bg": "#F5F5F5",
    "white": "#FFFFFF",
    "arrow": "#616161",
}


# ─── 通用绘图原语 ─────────────────────────────────


def setup_figure(width=10, height=5):
    """创建统一风格的 Figure / Axes。"""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_facecolor(COLORS["white"])
    fig.patch.set_facecolor(COLORS["white"])
    ax.axis("off")
    return fig, ax


def save_or_show(fig, name=None):
    """保存 SVG（直接运行脚本时）或返回 figure（被 import 时）。"""
    if len(sys.argv) > 1 and sys.argv[1].endswith(".svg"):
        out = sys.argv[1]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out)
        print(f"Saved: {out}")
    elif name:
        out = f"../source/_static/figures/{name}.svg"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out)
        print(f"Saved: {out}")
    else:
        plt.show()


def box(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    color=COLORS["aot"],
    alpha=0.1,
    fontsize=9,
    bold=True,
    halign="center",
    valign="center",
    linewidth=2,
):
    """绘制圆角矩形框并添加居中文字。"""
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08",
        facecolor=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=linewidth,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha=halign,
        va=valign,
        fontsize=fontsize,
        color=color,
        fontweight="bold" if bold else "normal",
    )


def arrow(
    ax, x1, y1, x2, y2, *, color=COLORS["arrow"], lw=1.5, style="->", dashed=False, **kw
):
    """绘制箭头。"""
    arrowprops = dict(arrowstyle=style, color=color, lw=lw)
    if dashed:
        arrowprops["linestyle"] = "dashed"
    arrowprops.update(kw)
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrowprops)


def label(
    ax,
    x,
    y,
    text,
    *,
    color=COLORS["gray"],
    fontsize=9,
    ha="center",
    va="center",
    bold=False,
    italic=False,
    **kw,
):
    """在指定位置添加文字。"""
    style = "italic" if italic else "normal"
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        fontweight="bold" if bold else "normal",
        fontstyle=style,
        **kw,
    )


def diamond(ax, cx, cy, size, text, *, color=COLORS["gray"], fontsize=8):
    """绘制菱形判断框。"""
    d = size / 2
    poly = mpatches.Polygon(
        [[cx, cy + d], [cx + d, cy], [cx, cy - d], [cx - d, cy]],
        facecolor=color,
        alpha=0.1,
        edgecolor=color,
        linewidth=1.5,
    )
    ax.add_patch(poly)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
    )


def legend(ax, items, loc="lower center", ncol=4):
    """创建图例。items = [(color, label), ...]"""
    handles = [
        mpatches.Patch(facecolor=c, alpha=0.12, edgecolor=c, label=l) for c, l in items
    ]
    ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=9, frameon=False)
