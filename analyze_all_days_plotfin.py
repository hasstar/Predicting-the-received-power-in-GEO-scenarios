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
# 设置中文字体 (根据你的系统选择)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 定义统一的颜色方案，确保两张图对应关系一致
colors_dict = {
    'Gaussian': '#3498db',  # 蓝色
    'Rice': '#2ecc71',      # 绿色
    'Rayleigh': '#95a5a6',  # 灰色
    'Weibull': '#e74c3c'    # 红色 (重点突出)
}

# ==========================================
# 图 1: 分布占比随时间粒度变化 (堆叠面积图)
# ==========================================
print("正在读取统计汇总数据...")
try:
    df_summary = pd.read_csv('granularity_analysis_summary.csv')
except FileNotFoundError:
    print("错误：找不到 granularity_analysis_summary.csv")
    exit()

print("正在绘制图 1: 分布占比演变图...")
fig1, ax1 = plt.subplots(figsize=(8, 5))

x = df_summary['Granularity_Min']
# 准备堆叠数据 (确保顺序一致)
labels = ['Gaussian', 'Rice', 'Rayleigh', 'Weibull']
y_data = [df_summary[f'Pct_{label}'] for label in labels]
colors = [colors_dict[label] for label in labels]

# 绘制堆叠面积图
ax1.stackplot(x, y_data, labels=labels, colors=colors, alpha=0.85)

# 装饰图表
ax1.set_xlim(1, 120)
ax1.set_ylim(0, 100)
ax1.set_xlabel('时间粒度 (观测窗口大小/分钟)', fontsize=12, fontweight='bold')
ax1.set_ylabel('最佳拟合占比 (%)', fontsize=12, fontweight='bold')
#ax1.set_title('各统计分布模型的适用性随观测时间尺度的演变 (1-120 min)', fontsize=16, pad=20)
ax1.legend(loc='upper left',  fontsize=8)
ax1.grid(True, alpha=0.3, linestyle='--')
# ax1.xticks(fontsize=10)
# ax1.yticks(fontsize=10)
# 添加解释性标注 (示例，根据一般规律)
ax1.text(5, 50, '短时间尺度\n(快衰落/噪声主导)', color='white', ha='center', fontweight='bold', alpha=0.9)
ax1.text(100, 50, '长时间尺度\n(慢衰落/非平稳主导)', color='white', ha='center', fontweight='bold', alpha=0.9)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 手动调整间距
plt.savefig('Plot1_Distribution_Evolution.png', dpi=300)
print("图 1 已保存为 Plot1_Distribution_Evolution.png")

# ==========================================
# 图 2: 15分钟典型片段的 PDF 拟合详情 (含局部放大)
# ==========================================
print("\n正在读取 15分钟 样本数据...")
try:
    df_sample = pd.read_csv('sample_15min_data.csv')
    data = df_sample['peak_power'].values
except FileNotFoundError:
    print("错误：找不到 sample_15min_data.csv")
    exit()

print("正在绘制图 2: 详细拟合分析图...")
fig2, ax2 = plt.subplots(figsize=(8, 5))

# 1. 绘制实测数据直方图
# 计算直方图数据 (density=True)
n, bins, patches = ax2.hist(data, bins=60, density=True, color='gray', alpha=0.3, label='实测数据 (Histogram)')

# 2. 生成拟合曲线的 X 轴
x_plot = np.linspace(data.min() - 1, data.max() + 1, 1000)

# 3. 拟合各分布并绘图
# --- Weibull (红实线) ---
params_w = stats.weibull_min.fit(data)
y_w = stats.weibull_min.pdf(x_plot, *params_w)
ks_w = stats.kstest(data, lambda x: stats.weibull_min.cdf(x, *params_w)).statistic
ax2.plot(x_plot, y_w, color=colors_dict['Weibull'], linewidth=2.5,
         label=f'Weibull (KS={ks_w:.3f})')

# --- Rice (绿虚线) ---
try:
    params_r = stats.rice.fit(data)
    b, loc, scale = params_r
    K_factor = b  # K因子就是b参数

    y_r = stats.rice.pdf(x_plot, *params_r)
    ks_r = stats.kstest(data, lambda x: stats.rice.cdf(x, *params_r)).statistic

    ax2.plot(x_plot, y_r, color=colors_dict['Rice'], linestyle='--', linewidth=2,
             label=f'Rice (K={K_factor:.2f}, KS={ks_r:.3f})')  # 添加K因子显示
except:
    print("Rice 分布拟合失败，跳过绘制")
    y_r = np.zeros_like(x_plot)

# --- Gaussian (蓝点线) ---
params_n = stats.norm.fit(data)
y_n = stats.norm.pdf(x_plot, *params_n)
ks_n = stats.kstest(data, lambda x: stats.norm.cdf(x, *params_n)).statistic
ax2.plot(x_plot, y_n, color=colors_dict['Gaussian'], linestyle=':', linewidth=2,
         label=f'Gaussian (KS={ks_n:.3f})')

# 装饰主图
ax2.set_xlabel('Peak Power (dBm)', fontsize=12)
ax2.set_ylabel('概率密度 (PDF)', fontsize=12)
#ax2.set_title(f'15分钟典型数据片段分布拟合详情\n(Weibull vs Gaussian)', fontsize=16)
ax2.legend(fontsize=8, loc='upper right', frameon=True, shadow=True)
# ax2.xticks(fontsize=10)
# ax2.yticks(fontsize=10)
# ==========================================
# 制作局部放大 (Inset Zoom)
# ==========================================
# 策略：聚焦于概率密度最高的“峰值”区域，因为这里往往能看出分布是尖峭还是平坦
peak_x = bins[np.argmax(n)] # 找到直方图最高的那个柱子的位置
zoom_width = (data.max() - data.min()) * 0.15 # 缩放窗口宽度占总宽度的 15%
x1, x2 = peak_x - zoom_width/2, peak_x + zoom_width/2
y1, y2 = 0, np.max(n) * 1.05

# 创建嵌入轴 (位置在图的左上或左中)
axins = inset_axes(ax2, width="40%", height="30%", loc='center left',
                   bbox_to_anchor=(0.05, 0.45, 0.45, 0.45), bbox_transform=ax2.transAxes)

# 在嵌入轴中重画内容
axins.hist(data, bins=60, density=True, color='gray', alpha=0.3)
axins.plot(x_plot, y_w, color=colors_dict['Weibull'], linewidth=2.5) # Weibull
axins.plot(x_plot, y_r, color=colors_dict['Rice'], linestyle='--', linewidth=2)  # Rice
axins.plot(x_plot, y_n, color=colors_dict['Gaussian'], linestyle=':', linewidth=2)   # Gaussian

# 设置嵌入轴的显示范围
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# 隐藏嵌入轴的刻度数值（为了美观，也可以保留）
# axins.set_xticks([])
# axins.set_yticks([])
axins.set_title("峰值拟合细节放大", fontsize=10, fontweight='bold')
axins.grid(False)

# 画连接线 (将放大镜和主图区域连起来)
mark_inset(ax2, axins, loc1=3, loc2=4, fc="none", ec="0.5", linestyle='--')

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 手动调整间距
plt.savefig('Plot2_PDF_Analysis_Zoom.png', dpi=300)
print("图 2 已保存为 Plot2_PDF_Analysis_Zoom.png")

print("\n绘图完成！请查看生成的 .png 图片。")