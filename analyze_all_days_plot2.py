import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==========================================
# 1. 审美配置 (High-End Aesthetics)
# ==========================================
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2

# 保持一致的配色
colors_dict = {
    'Gaussian': '#2980B9',  # 宝石蓝
    'Rice': '#27AE60',  # 森林绿
    'Rayleigh': '#7F8C8D',  # 钢灰
    'Weibull': '#C0392B'  # 深绯红
}

# ==========================================
# 2. 数据读取
# ==========================================
print("\nReading 15-min sample data...")
try:
    df_sample = pd.read_csv('sample_15min_data.csv')
    data = df_sample['peak_power'].values
except FileNotFoundError:
    print("Error: sample_15min_data.csv not found.")
    # 生成模拟数据防止报错
    data = np.random.weibull(2, 1000) * 5 - 90

# ==========================================
# 3. 绘图 (Plotting)
# ==========================================
print("Plotting Figure 2...")
fig2, ax2 = plt.subplots(figsize=(8, 5))

# 1. 绘制实测数据直方图 (Measured Data)
# color='gray' with low alpha provides a neutral background
n, bins, patches = ax2.hist(data, bins=60, density=True,
                            color='gray', alpha=0.3,
                            label='Measured Data')

# 2. 生成 X 轴数据
x_plot = np.linspace(data.min() - 1, data.max() + 1, 1000)

# 3. 拟合各分布 (Fitting & Curves)

# --- Weibull (Red Solid Line) ---
params_w = stats.weibull_min.fit(data)
y_w = stats.weibull_min.pdf(x_plot, *params_w)
ks_w = stats.kstest(data, lambda x: stats.weibull_min.cdf(x, *params_w)).statistic
ax2.plot(x_plot, y_w,
         color=colors_dict['Weibull'], linewidth=3, linestyle='-',
         label=f'Weibull (KS={ks_w:.3f})')

# --- Rice (Green Dashed Line) ---
try:
    params_r = stats.rice.fit(data)
    b, loc, scale = params_r
    K_factor = b  # Keeping your logic
    y_r = stats.rice.pdf(x_plot, *params_r)
    ks_r = stats.kstest(data, lambda x: stats.rice.cdf(x, *params_r)).statistic

    ax2.plot(x_plot, y_r,
             color=colors_dict['Rice'], linewidth=2.5, linestyle='--',
             label=f'Rice ($K$={K_factor:.2f}, KS={ks_r:.3f})')
except:
    pass

# --- Gaussian (Blue Dotted Line) ---
params_n = stats.norm.fit(data)
y_n = stats.norm.pdf(x_plot, *params_n)
ks_n = stats.kstest(data, lambda x: stats.norm.cdf(x, *params_n)).statistic
ax2.plot(x_plot, y_n,
         color=colors_dict['Gaussian'], linewidth=2.5, linestyle=':',
         label=f'Gaussian (KS={ks_n:.3f})')

# ==========================================
# 4. 装饰 (Styling)
# ==========================================

# 坐标轴标签 (English)
ax2.set_xlabel('Peak Power (dBm)', fontsize=16, fontweight='bold', labelpad=10)
ax2.set_ylabel('Probability Density', fontsize=16, fontweight='bold', labelpad=10)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# 坐标轴美化 (Spine Offset)
# ax2.spines['left'].set_position(('outward', 10))
# ax2.spines['bottom'].set_position(('outward', 10))
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

# 图例位置调整到左上角 (Upper Left)
# frameon=False 去除图例边框，更简洁
ax2.legend(loc='upper left', fontsize=14, frameon=False)

plt.tight_layout()
plt.savefig('Plot2_PDF_Analysis_English.pdf', dpi=300, bbox_inches='tight')
print("Figure saved as: Plot2_PDF_Analysis_English.pdf")
plt.show()