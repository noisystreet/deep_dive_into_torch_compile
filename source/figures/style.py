"""
统一 Matplotlib 风格：浅入深出 torch.compile
"""

import matplotlib
import matplotlib.pyplot as plt

# 使用 Noto Sans CJK 支持中文
matplotlib.rcParams.update(
    {
        "font.family": ["Noto Sans CJK JP", "DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "svg.fonttype": "path",  # 文字转为路径，无需依赖中文字体
    }
)

# 配色方案（与 PyTorch 橙 / Inductor 蓝 保持一致）
COLORS = {
    "dynamo": "#E2882C",  # 橙 — Dynamo
    "aot": "#2C5F8A",  # 蓝 — AOTAutograd
    "inductor": "#4CAF50",  # 绿 — Inductor
    "triton": "#9C27B0",  # 紫 — Triton
    "gray": "#757575",
    "light_gray": "#E0E0E0",
    "light_bg": "#F5F5F5",
    "white": "#FFFFFF",
    "arrow": "#616161",
}


def setup_figure(width=10, height=5):
    """创建统一风格的 Figure。"""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_facecolor(COLORS["white"])
    fig.patch.set_facecolor(COLORS["white"])
    return fig, ax
