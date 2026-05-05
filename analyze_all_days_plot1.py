import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ==========================================
# 1. 审美配置 (Aesthetics)
# ==========================================
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2

colors = {
    'Weibull': '#C0392B',  # 深绯红
    'Gaussian': '#2980B9',  # 宝石蓝
    'Rice': '#27AE60',  # 森林绿
    'Rayleigh': '#7F8C8D'  # 钢灰
}

linestyles = {
    'Weibull': '-',
    'Gaussian': '--',
    'Rice': '-.',
    'Rayleigh': ':'
}

# 加载数据
try:
    df_summary = pd.read_csv('granularity_analysis_summary.csv')
except FileNotFoundError:
    # 模拟数据
    x_dummy = np.arange(1, 121)
    df_summary = pd.DataFrame({
        'Granularity_Min': x_dummy,
        'Pct_Gaussian': 45 * np.exp(-x_dummy / 15) + 5,
        'Pct_Rice': 35 * np.exp(-x_dummy / 25) + 5,
        'Pct_Rayleigh': 10 + 2 * np.sin(x_dummy / 10),
        'Pct_Weibull': 10 + 85 * (1 - np.exp(-x_dummy / 35))
    })

# ==========================================
# 2. 绘图 (Plotting)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))

# --- A. 先画基准线 (Reference Line) ---
# zorder=0 确保它在所有曲线的后面
# color='#555555' 深灰色，既清晰又不抢眼
ax.axhline(y=90, color='#555555', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)

# 在线旁边添加文字标注
# transform=ax.get_yaxis_transform() 确保文字紧贴 Y 轴数值，或者直接指定 x 坐标
# 这里我们放在图的左侧内部，稍微上方一点
ax.text(1, 91.5, '90% Benchmark',
        color='#555555',
        fontsize=12,
        fontweight='bold',
        style='italic',  # 斜体增加标注感
        ha='left', va='bottom')

# --- B. 画数据曲线 (Data Curves) ---
x = df_summary['Granularity_Min']
labels_order = ['Rayleigh', 'Rice', 'Gaussian', 'Weibull']

for label in labels_order:
    y_raw = df_summary[f'Pct_{label}']

    # B-Spline 平滑
    x_smooth = np.linspace(x.min(), x.max(), 500)
    spl = make_interp_spline(x, y_raw, k=3)
    y_smooth = spl(x_smooth)
    y_smooth = np.clip(y_smooth, 0, 100)

    # Halo Effect (光晕描边)
    # 这步很关键：白色的描边会盖住后面的灰色基准线，
    # 制造出"曲线穿过并遮挡基准线"的物理前后关系
    ax.plot(x_smooth, y_smooth, color='white', linewidth=5, alpha=1.0, zorder=2)

    # 彩色主线
    is_main = (label == 'Weibull')
    ax.plot(x_smooth, y_smooth,
            label=label,
            color=colors[label],
            linestyle=linestyles[label],
            linewidth=3.5 if is_main else 2.5,
            alpha=0.95 if is_main else 0.85,
            zorder=3)

    # Sparse Markers
    # idx_markers = np.linspace(0, len(x_smooth) - 1, 8).astype(int)
    # ax.plot(x_smooth[idx_markers], y_smooth[idx_markers],
    #         marker='o', linestyle='None',
    #         color=colors[label], markerfacecolor='white',
    #         markeredgewidth=2, markersize=8 if is_main else 6,
    #         zorder=4)

# ==========================================
# 3. 修饰 (Refinement)
# ==========================================
# ax.spines['left'].set_position(('outward', 10))
# ax.spines['bottom'].set_position(('outward', 10))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.3)
ax.xaxis.grid(False)

ax.set_xlim(0, 120)
ax.set_ylim(-2, 105)

ax.set_xlabel('Time Granularity (min)', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('Best Fit Proportion (%)', fontsize=16, fontweight='bold', labelpad=10)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# 图例优化
legend = ax.legend(loc='lower right', fontsize=14, frameon=False,
                   bbox_to_anchor=(1.0, 0.15))
                   #title='Distribution Model', title_fontsize=12)
legend.get_title().set_fontweight('bold')

plt.tight_layout()
plt.savefig('Plot1_With_90Benchmark.pdf', dpi=300, bbox_inches='tight')
print("已生成带基准线的图表: Plot1_With_90Benchmark.pdf")
plt.show()