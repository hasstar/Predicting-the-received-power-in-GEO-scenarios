import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置与读取
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

print("Reading data...")
try:
    df = pd.read_csv('数据/snapshot_SinglePoint_results_2s.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.dropna(subset=['peak_power'])
except FileNotFoundError:
    print("Error: File not found.")
    exit()

# ==========================================
# 2. 更稳健的筛选逻辑 (Robust Selection)
# ==========================================
granularity_min = 15
groups = df.groupby(pd.Grouper(key='datetime', freq=f'{granularity_min}min'))

segment_stats = []
print(f"Scanning {granularity_min}-min segments...")

for time_key, group in groups:
    data_segment = group['peak_power'].values
    if len(data_segment) < 100: continue  # 数据太少跳过

    # 【核心修改】：直接基于原始数据计算 1% 分位点
    # 不依赖拟合，这样就不会被错误的拟合参数误导
    empirical_p1 = np.percentile(data_segment, 1)

    segment_stats.append({
        'time': time_key,
        'data': data_segment,
        'empirical_p1': empirical_p1,
        'mean': np.mean(data_segment)
    })

if not segment_stats:
    print("No valid segments found.")
    exit()

df_stats = pd.DataFrame(segment_stats)

# 1. 找到“真实”最差的情况 (Empirical Worst)
# 即：原始数据中，最低的那 1% 真的掉得最深的那一段
worst_row = df_stats.loc[df_stats['empirical_p1'].idxmin()]

# 2. 找到典型情况 (Median)
median_idx = df_stats['empirical_p1'].sort_values().index[len(df_stats) // 2]
avg_row = df_stats.loc[median_idx]

print("\n" + "=" * 60)
print("Screening Result (Based on Raw Data):")
print(f"Worst Time:   {worst_row['time']} (Real P1: {worst_row['empirical_p1']:.2f} dBm)")
print(f"Typical Time: {avg_row['time']} (Real P1: {avg_row['empirical_p1']:.2f} dBm)")
print("=" * 60)

# ==========================================
# 3. 拟合与绘图 (带有保护机制)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 7))


# 辅助函数：安全的 Weibull 拟合
def safe_weibull_fit(data, label_prefix, color_hist, color_line):
    # 画直方图
    ax.hist(data, bins=50, density=True, color=color_hist, alpha=0.3, label=f'{label_prefix} Data')

    # 拟合
    # floc=None 让它自动寻找，但如果数据离散，有时需要限制
    shape, loc, scale = stats.weibull_min.fit(data)

    # 【保护机制】：如果拟合出 k < 1 (形状不对)，尝试强制修正 loc
    # Weibull 对 loc 非常敏感。如果 loc 太远，k 就会变小。
    # 强制 loc 略小于数据最小值
    if shape < 1.0:
        print(f"Warning: {label_prefix} initial fit bad (k={shape:.2f}). Retrying with constrained loc.")
        # 强制 loc 固定在数据最小值左侧一点点
        fixed_loc = data.min() - 0.1
        shape, loc, scale = stats.weibull_min.fit(data, floc=fixed_loc)

    # 生成曲线点
    x_plot = np.linspace(data.min() - 5, data.max() + 5, 1000)
    y_plot = stats.weibull_min.pdf(x_plot, shape, loc=loc, scale=scale)

    # 画线
    ax.plot(x_plot, y_plot, color=color_line, linewidth=3,
            label=f'{label_prefix} Fit (k={shape:.2f})')

    # 计算拟合后的 P1 (用于展示)
    fit_p1 = stats.weibull_min.ppf(0.01, shape, loc=loc, scale=scale)

    return shape, loc, scale, fit_p1


# 绘制典型情况
k_avg, loc_avg, lam_avg, p1_avg = safe_weibull_fit(
    avg_row['data'], "Typical", '#3498db', '#2980B9'
)

# 绘制最差情况
k_worst, loc_worst, lam_worst, p1_worst = safe_weibull_fit(
    worst_row['data'], "Worst", '#e74c3c', '#c0392b'
)

# ==========================================
# 4. 修饰
# ==========================================
ax.set_title(f'Reliability Analysis: Worst vs. Typical ({granularity_min}-min Segments)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Received Peak Power (dBm)', fontsize=14)
ax.set_ylabel('Probability Density', fontsize=14)

# 参数框 (更新为最差情况的参数)
text_str = (f"$\\bf{{Worst\\ Case\\ Params}}$\n"
            f"Shape ($k$): {k_worst:.3f}\n"
            f"Scale ($\\lambda$): {lam_worst:.3f}\n"
            f"Loc: {loc_worst:.3f}\n"
            f"Risk ($P_{{1\%}}$): {p1_worst:.2f} dBm")

ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fce4ec', edgecolor='#c0392b', alpha=0.9))

ax.legend(loc='upper right', frameon=True, fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('Worst_Case_Analysis_Fixed.png', dpi=300)
print("Done. Saved as Worst_Case_Analysis_Fixed.png")
plt.show()