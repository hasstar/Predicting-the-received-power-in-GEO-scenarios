import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
# ==========================
# Load Excel
# ==========================
df = pd.read_excel("allResultDetailed.xlsx")  # ← change to your path if needed

x = df.iloc[:, 0]
x_name = df.columns[0]

y_df = df.iloc[:, 1:]
y_names = list(y_df.columns)

# ==========================
# Use English only → no Chinese font needed
# ==========================

# ==========================
# Colors (distinct & paper-friendly)
# ==========================
colors = [
    "#4C72B0", "#DD8452", "#55A868",
    "#C44E52", "#8172B2", "#937860"
]

# All curves use the SAME marker and SAME linestyle
marker_style = 'o'
line_style = '-'

plt.figure(figsize=(8, 5), dpi=300)

# ==========================
# Plot curves
# ==========================
for i, col in enumerate(y_names):
    y = y_df[col]

    plt.plot(
        x, y,
        marker=marker_style,
        linestyle=line_style,
        linewidth=1.8,
        markersize=3,
        markeredgewidth=0.8,
        color=colors[i % len(colors)],
        label=col
    )

    # Annotate ONLY the last point (skip the first point)
    # plt.annotate(
    #     f"{y.iloc[-1]:.3f}",
    #     xy=(x.iloc[-1], y.iloc[-1]),
    #     xytext=(6, 0),
    #     textcoords="offset points",
    #     fontsize=7,
    #     va="center"
    # )

# ==========================
# Styling
# ==========================
plt.xlabel(x_name, fontsize=12)
plt.ylabel("R²", fontsize=12)

plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# plt.legend(
#     title="Models",
#     fontsize=9,
#     title_fontsize=10,
#     bbox_to_anchor=(1.02, 1),
#     loc="down right",
#     frameon=False
# )
plt.legend(
    # title="Models",
    fontsize=10,
    # title_fontsize=10,
    loc="lower right",  # 自动选择最佳位置
    frameon=True
)
plt.tight_layout()


# 一行代码添加透明层
ax = plt.gca()
y_min, y_max = ax.get_ylim()

# 背景矩形
ax.add_patch(patches.Rectangle((0, y_min), 6, y_max-y_min, alpha=0.6,
                              facecolor='lightblue', linewidth=0, zorder=0))
# 虚线网格
[plt.axvline(x, color='red', linestyle='--', alpha=0.2) for x in range(5, 5)]
[plt.axhline(y, color='green', linestyle=':', alpha=0.2) for y in np.linspace(y_min, y_max, 8)]

plt.draw()


plt.savefig("plot_r2.png", dpi=300, bbox_inches="tight")
plt.savefig("plot_r2.pdf", dpi=300, bbox_inches="tight")
plt.show()
