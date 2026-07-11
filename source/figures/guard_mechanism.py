"""
Dynamo Guard 层次结构图
"""

from style import setup_figure, save_or_show, box, arrow, label, COLORS


def draw():
    fig, ax = setup_figure(width=8, height=5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Root
    box(ax, 3.8, 4.8, 2.4, 0.7, "RootGuardManager", color=COLORS["dynamo"], fontsize=10)

    # TensorGuardManager x
    box(
        ax, 0.5, 2.8, 3.0, 0.6, "TensorGuardManager(x)", color=COLORS["aot"], fontsize=9
    )
    labels_x = [
        "ShapeGuard(shape=(32, 784))",
        "DTypeGuard(dtype=float32)",
        "DeviceGuard(device=cuda:0)",
    ]
    for i, txt in enumerate(labels_x):
        box(
            ax,
            0.7,
            2.0 - i * 0.55,
            2.6,
            0.45,
            txt,
            color=COLORS["aot"],
            fontsize=7,
            bold=False,
            alpha=0.06,
        )

    # TensorGuardManager y
    box(
        ax, 6.5, 2.8, 3.0, 0.6, "TensorGuardManager(y)", color=COLORS["aot"], fontsize=9
    )
    labels_y = [
        "ShapeGuard(shape=(32, 784))",
        "DTypeGuard(dtype=float32)",
        "DeviceGuard(device=cuda:0)",
    ]
    for i, txt in enumerate(labels_y):
        box(
            ax,
            6.7,
            2.0 - i * 0.55,
            2.6,
            0.45,
            txt,
            color=COLORS["aot"],
            fontsize=7,
            bold=False,
            alpha=0.06,
        )

    # ID_MATCH Guard
    box(
        ax,
        2.0,
        0.3,
        6.0,
        0.6,
        "ID_MATCH_Guard(model=<MyModel at 0x1234>)",
        color=COLORS["gray"],
        fontsize=8,
        bold=False,
    )

    # 箭头
    arrow(ax, 5.0, 4.8, 2.0, 3.4, color=COLORS["arrow"])
    arrow(ax, 5.0, 4.8, 8.0, 3.4, color=COLORS["arrow"])
    arrow(ax, 5.0, 1.5, 5.0, 0.9, color=COLORS["arrow"])

    label(
        ax,
        5.0,
        5.6,
        "Dynamo Guard 层次结构",
        fontsize=13,
        bold=True,
        color=COLORS["dynamo"],
    )

    return fig


if __name__ == "__main__":
    save_or_show(draw(), "guard_mechanism")
