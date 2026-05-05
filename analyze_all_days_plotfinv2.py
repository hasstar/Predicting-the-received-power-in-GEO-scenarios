import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy import stats
import seaborn as sns

# ==========================================
# 0. 设置绘图环境
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 更新颜色字典，加入 Rayleigh
colors_dict = {
    'Gaussian': '#3498db',  # 蓝色 (Blue)
    'Rice': '#2ecc71',      # 绿色 (Green)
    'Rayleigh': '#8e44ad',  # 紫色 (Purple) - 新增，颜色鲜明以便区分
    'Weibull': '#e74c3c'    # 红色 (Red) - 重点突出
}

# ==========================================
# 读取数据
# ==========================================
print("正在读取 15分钟 样本数据...")
try:
    df_sample = pd.read_csv('sample_15min_data.csv')
    data = df_sample['peak_power'].values
except FileNotFoundError:
    print("错误：找不到 sample_15min_data.csv，请先运行上一步的分析代码。")
    exit()

# ==========================================
# 绘制图 2: 详细拟合分析图 (含 Rayleigh)
# ==========================================
print("正在绘制图 2: 详细拟合分析图 (含 Rayleigh)...")
fig2, ax2 = plt.subplots(figsize=(14, 9))

# 1. 绘制实测数据直方图
n, bins, patches = ax2.hist(data, bins=60, density=True, color='gray', alpha=0.3, label='实测数据 (Histogram)')

# 2. 生成拟合曲线的 X 轴
x_plot = np.linspace(data.min() - 1, data.max() + 1, 1000)

# 3. 拟合各分布并绘图 (按重要性排序)

# --- A. Weibull (红实线 - 重点) ---
params_w = stats.weibull_min.fit(data)
y_w = stats.weibull_min.pdf(x_plot, *params_w)
ks_w = stats.kstest(data, lambda x: stats.weibull_min.cdf(x, *params_w)).statistic
ax2.plot(x_plot, y_w, color=colors_dict['Weibull'], linewidth=2.5,
         label=f'Weibull (KS={ks_w:.3f})')

# --- B. Rice (绿虚线) ---
try:
    params_r = stats.rice.fit(data)
    y_r = stats.rice.pdf(x_plot, *params_r)
    ks_r = stats.kstest(data, lambda x: stats.rice.cdf(x, *params_r)).statistic
    ax2.plot(x_plot, y_r, color=colors_dict['Rice'], linestyle='--', linewidth=2,
             label=f'Rice (KS={ks_r:.3f})')
except:
    y_r = np.zeros_like(x_plot)

# --- C. Rayleigh (紫点划线 - 新增) ---
# Rayleigh 分布通常只有 loc 和 scale 参数
params_ray = stats.rayleigh.fit(data)
y_ray = stats.rayleigh.pdf(x_plot, *params_ray)
ks_ray = stats.kstest(data, lambda x: stats.rayleigh.cdf(x, *params_ray)).statistic
ax2.plot(x_plot, y_ray, color=colors_dict['Rayleigh'], linestyle='-.', linewidth=2,
         label=f'Rayleigh (KS={ks_ray:.3f})')

# --- D. Gaussian (蓝点线 - 基准) ---
params_n = stats.norm.fit(data)
y_n = stats.norm.pdf(x_plot, *params_n)
ks_n = stats.kstest(data, lambda x: stats.norm.cdf(x, *params_n)).statistic
ax2.plot(x_plot, y_n, color=colors_dict['Gaussian'], linestyle=':', linewidth=2,
         label=f'Gaussian (KS={ks_n:.3f})')

# 装饰主图
ax2.set_xlabel('Peak Power (dBm)', fontsize=12)
ax2.set_ylabel('概率密度 (PDF)', fontsize=12)
ax2.set_title(f'15分钟典型数据片段分布拟合详情\n(Weibull vs Rice vs Rayleigh vs Gaussian)', fontsize=16)
ax2.legend(fontsize=12, loc='upper right', frameon=True, shadow=True, title="Fit Models")

# ==========================================
# 制作局部放大 (Inset Zoom)
# ==========================================
# 自动聚焦于峰值区域
peak_x = bins[np.argmax(n)]
zoom_width = (data.max() - data.min()) * 0.15
x1, x2 = peak_x - zoom_width/2, peak_x + zoom_width/2
y1, y2 = 0, np.max(n) * 1.05

# 创建嵌入轴
axins = inset_axes(ax2, width="40%", height="30%", loc='center left',
                   bbox_to_anchor=(0.05, 0.45, 0.45, 0.45), bbox_transform=ax2.transAxes)

# 在嵌入轴中重画所有线条
axins.hist(data, bins=60, density=True, color='gray', alpha=0.3)
axins.plot(x_plot, y_w, color=colors_dict['Weibull'], linewidth=2.5)       # Weibull
axins.plot(x_plot, y_r, color=colors_dict['Rice'], linestyle='--', linewidth=2)  # Rice
axins.plot(x_plot, y_ray, color=colors_dict['Rayleigh'], linestyle='-.', linewidth=2) # Rayleigh (新增)
axins.plot(x_plot, y_n, color=colors_dict['Gaussian'], linestyle=':', linewidth=2)   # Gaussian

# 设置嵌入轴范围
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_title("峰值拟合细节放大", fontsize=10, fontweight='bold')
axins.grid(False)

# 画连接线
mark_inset(ax2, axins, loc1=3, loc2=4, fc="none", ec="0.5", linestyle='--')

plt.tight_layout()
save_name = 'Plot2_PDF_Analysis_With_Rayleigh.png'
plt.savefig(save_name, dpi=300)
print(f"图表已保存为: {save_name}")
plt.show()