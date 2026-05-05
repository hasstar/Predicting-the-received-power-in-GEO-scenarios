import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# ==========================
# 1. Load Data
# ==========================
file_path = "allResultDetailed.xlsx"  # 请确保文件名正确
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    # 生成假数据用于演示 (如果你没有文件，这段代码也能跑)
    x = np.arange(1, 61)
    df = pd.DataFrame({'Features': x})
    df['CatBoost'] = 0.85 - 0.5 * np.exp(-0.5 * x) + np.random.normal(0, 0.005, 60)
    df['XGBoost'] = 0.78 - 0.4 * np.exp(-0.3 * x) + np.random.normal(0, 0.005, 60)
    df['MLP'] = 0.75 - 0.3 * np.exp(-0.2 * x) + np.random.normal(0, 0.008, 60)
    df['SVM'] = 0.55 - 0.2 * np.exp(-0.1 * x) + np.random.normal(0, 0.01, 60)

# 第一列是X轴（特征数量）
x = df.iloc[:, 0]
# 其余列是Y轴（各模型数据）
y_df = df.iloc[:, 1:]
models = y_df.columns

# ==========================
# 2. Plot Setup
# ==========================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 定义核心模型名称 (自动匹配包含此字符串的列)
target_model_keyword = 'CatBoost'

# -------------------------------------------------------
# 核心绘图逻辑：做减法，只突出重点
# -------------------------------------------------------
# 定义对比模型的样式配置
baseline_styles = [
    {'color': 'grey', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.7, 'marker': 'o', 'zorder': 2},
    {'color': 'grey', 'linestyle': '-.', 'linewidth': 1.5, 'alpha': 0.7, 'marker': 's', 'zorder': 3},
    {'color': 'grey', 'linestyle': ':', 'linewidth': 1.5, 'alpha': 0.7, 'marker': '^', 'zorder': 4},
    {'color': 'grey', 'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.6, 'marker': 'v', 'zorder': 5}
]

baseline_idx = 0
for col in models:
    y = y_df[col]

    # === A. 核心模型 (Proposed) ===
    if target_model_keyword.lower() in col.lower():
        # 1. 画线：红色，实线，稍粗
        ax.plot(x, y,
                color='#D62728',  # 鲜艳的红色
                linestyle='-',
                linewidth=2.5,
                label=f"{col} ",
                zorder=10)  # 保证在最上层

        # 2. 画点：不要每个点都画！
        # 策略：只在稀疏区域(前10个点)都画，后面每隔5个点画一个，避免密集恐惧症
        # 创建一个掩码
        mask = np.zeros_like(x, dtype=bool)
        mask[:10] = True  # 前10个点全画
        mask[10::5] = True  # 后面每隔5个画一个

        ax.scatter(x[mask], y[mask],
                   color='#D62728',
                   edgecolor='white',  # 加个白边，更精致
                   marker='s',  # 方块
                   s=40,  # 大小
                   linewidth=1.0,
                   zorder=11)

        # 3. 特别标注“拐点” (Elbow Point, 假设是第5个特征)
        # 找到 x=5 对应的 y 值 (如果数据里有5的话)
        try:
            elbow_idx = list(x).index(6)  # 找到特征数为5的索引
            elbow_x = x[elbow_idx]
            elbow_y = y[elbow_idx]

            # 画一个特殊的圈圈强调这里
            ax.plot(elbow_x, elbow_y, 'o', ms=15, mfc='none', mec='blue', mew=2, zorder=12)

            # # 加箭头注释
            # ax.annotate('Elbow Point\n(5 Features)',
            #             xy=(elbow_x, elbow_y),
            #             xytext=(elbow_x + 5, elbow_y - 0.1),
            #             arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=.2"),
            #             fontsize=11, fontweight='bold', color='#333333')
        except ValueError:
            pass  # 如果X轴里没有5这个数，就跳过

    # === B. 对比模型 (Baselines) ===
    else:
        # 为每个对比模型分配不同的样式
        style = baseline_styles[baseline_idx % len(baseline_styles)]
        baseline_idx += 1

        # 使用不同的颜色、线型和标记
        ax.plot(x, y,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                alpha=style['alpha'],
                marker=style['marker'],
                markersize=4,
                markeredgewidth=0.5,
                markeredgecolor='white',
                markevery=max(1, len(x) // 20),  # 每隔20个点画一个标记
                label=col,
                zorder=style['zorder'])

# ==========================
# 3. Styling & Decoration
# ==========================

# 坐标轴标签
ax.set_xlabel('Number of Features', fontsize=18, fontweight='bold')
ax.set_ylabel(r'$R^2$ ', fontsize=18, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# 美化坐标轴刻度
ax.tick_params(axis='both', which='major', labelsize=11)

# 网格线 (淡一点)
ax.grid(True, linestyle=':', alpha=0.6)

# 高亮“稀疏特征区” (前10个特征)
ax.axvspan(0, 10, color='#e6f3ff', alpha=0.5, zorder=0)
# ax.text(5, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 'Optimal Feature Number',ha='center', fontsize=10, color='#1f77b4', fontweight='bold')

# 图例优化
# 去掉边框，放在合适的位置
legend = ax.legend(fontsize=16, loc='lower right', frameon=True, fancybox=True, framealpha=0.9)
# 把图例标题加粗
# legend.set_title("Models", prop={'size':11, 'weight':'bold'})

# 去掉上方和右方的边框线 (Spines)，让图看起来更现代
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ==========================
# 4. Save & Show
# ==========================
plt.tight_layout()
timestamp = int(time.time())
plt.savefig(f'R2_Clean_{timestamp}.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'R2_Clean_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()