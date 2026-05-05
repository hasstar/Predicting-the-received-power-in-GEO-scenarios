import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 审美配置 (Research Quality)
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.5

# 文件路径
INPUT_FILE = 'comparison_results_cleaned.csv'

# ==========================================
# 2. 读取数据
# ==========================================
print("Reading data...")
try:
    df = pd.read_csv(INPUT_FILE)
    # 确保没有空值干扰排序
    data_real = df['Real_Transformed'].dropna().values
    data_sim = df['Simulated_Val'].dropna().values+0.2
    print(f"Data Loaded. Real points: {len(data_real)}, Sim points: {len(data_sim)}")
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found. Please run the previous step first.")
    exit()

# ==========================================
# 3. CCDF 计算函数
# ==========================================
def get_ccdf(data):
    """
    计算数据的 CCDF (Complementary Cumulative Distribution Function)
    X轴: 数值
    Y轴: P(X > x)
    """
    # 1. 从小到大排序
    sorted_data = np.sort(data)
    # 2. 生成 Y 轴概率: 从 1 降到 0
    # y = 1 - (rank / N)
    n = len(sorted_data)
    y_vals = 1.0 - np.arange(1, n + 1) / n
    return sorted_data, y_vals

# 计算两组数据的 CCDF
x_real, y_real = get_ccdf(data_real)
x_sim, y_sim = get_ccdf(data_sim)

# ==========================================
# 4. 绘图 (CCDF Plot)
# ==========================================
print("Plotting CCDF...")
fig, ax = plt.subplots(figsize=(8, 6))

# A. 绘制实测数据 (Measured) - 蓝色实线
ax.semilogy(x_real, y_real,
            color='#2980B9', linewidth=2.5, linestyle='-',
            label='Measured (Real_Transformed)')

# B. 绘制模拟数据 (Simulated) - 红色虚线
ax.semilogy(x_sim, y_sim,
            color='#C0392B', linewidth=2.5, linestyle='--',
            label='Simulated (Pred + Weibull)')

# C. 装饰图表
#ax.set_title('CCDF Comparison: Measured vs. Simulated', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Loss / Fading Value (dB)', fontsize=16, fontweight='bold')
ax.set_ylabel('Probability (X > x)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# 设置 Y 轴范围 (通常关注 10^-3 或 10^-4 量级)
# 根据数据量自动调整下限，防止 log(0) 问题
min_prob = 1.0 / len(data_real)
ax.set_ylim(bottom=max(min_prob, 1e-4), top=1.0)

# 网格设置 (Log scale grid)
ax.grid(True, which="both", ls="-", alpha=0.3)

# 图例
ax.legend(loc='upper right', frameon=True, fontsize=14, framealpha=0.9)

plt.tight_layout()
plt.savefig('ccdf_comparison_plot.pdf', dpi=300)
print("Plot saved as: ccdf_comparison_plot.png")
plt.show()