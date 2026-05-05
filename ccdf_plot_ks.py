import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

# 忽略 scipy 在样本量过大时可能发出的 p-value 警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 审美配置
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

INPUT_FILE = 'comparison_results_cleaned.csv'

# ==========================================
# 2. 读取数据
# ==========================================
print("Reading data...")
try:
    df = pd.read_csv(INPUT_FILE)
    # 提取两组数据
    data_real = df['Real_Transformed'].dropna().values
    data_sim = df['Simulated_Val'].dropna().values

    # 简单的对齐检查 (虽然理论上已经对齐，但为了统计检验严谨性)
    # 如果长度不一致，截取最短的长度
    min_len = min(len(data_real), len(data_sim))
    data_real = data_real[:min_len]
    data_sim = data_sim[:min_len]

    print(f"Data Loaded. N={min_len}")

except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found.")
    exit()

# ==========================================
# 3. 统计显著性检验 (Statistical Tests)
# ==========================================
print("Running Statistical Tests...")

# --- A. Kolmogorov-Smirnov (K-S) Test ---
# H0: 两个样本服从同一分布
# statistic (D): 两个CDF之间的最大距离 (越小越好)
ks_stat, ks_p = stats.ks_2samp(data_real, data_sim)

# --- B. Anderson-Darling (A-D) Test (k-sample) ---
# H0: 两个样本来自同一总体
# A-D 对尾部差异更敏感，非常适合衰落模型验证
# 注意：如果样本量非常大(>5000)，anderson_ksamp 计算会很慢，且 p-value 几乎总是趋近于 0
# 为了演示和计算速度，如果数据量太大，我们进行随机抽样 (例如抽 2000 个点) 进行 A-D 检验
if len(data_real) > 5000:
    print("   (Data too large for AD-test, sampling 2000 points for metric calculation...)")
    idx = np.random.choice(len(data_real), 2000, replace=False)
    ad_res = stats.anderson_ksamp([data_real[idx], data_sim[idx]])
else:
    ad_res = stats.anderson_ksamp([data_real, data_sim])

ad_stat = ad_res.statistic
ad_p = ad_res.significance_level  # 这是一个临界值概率

print(f"   K-S Statistic: {ks_stat:.4f} (p={ks_p:.4e})")
print(f"   A-D Statistic: {ad_stat:.4f} (p={ad_p})")


# ==========================================
# 4. CCDF 计算与绘图
# ==========================================
def get_ccdf(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y_vals = 1.0 - np.arange(1, n + 1) / n
    return sorted_data, y_vals


x_real, y_real = get_ccdf(data_real)
x_sim, y_sim = get_ccdf(data_sim)

print("Plotting CCDF with Stats...")
fig, ax = plt.subplots(figsize=(9, 7))

# 绘制曲线
ax.semilogy(x_real, y_real,
            color='#2980B9', linewidth=2.5, linestyle='-',
            label='Measured Target')

ax.semilogy(x_sim, y_sim,
            color='#C0392B', linewidth=2.5, linestyle='--',
            label='Weibull Expanded')

# 装饰
ax.set_title('Distribution Validation: CCDF & Statistical Tests', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Value (dB)', fontsize=13, fontweight='bold')
ax.set_ylabel('Probability (X > x)', fontsize=13, fontweight='bold')

# 设置范围
min_prob = 1.0 / len(data_real)
ax.set_ylim(bottom=max(min_prob, 1e-4), top=1.0)
ax.grid(True, which="both", ls="-", alpha=0.3)
ax.legend(loc='upper right', frameon=True, fontsize=11)

# ==========================================
# 5. 添加统计检验结果文本框
# ==========================================
# 解释：对于大样本数据，P值通常极小(显著差异)，这是正常的。
# 科研中主要看 Statistic (统计量) 的大小来判断"拟合优度"。
# KS < 0.1 通常被认为拟合得不错。

stats_text = (f"$\\bf{{Statistical\\ Validation}}$\n"
              f"Sample Size ($N$): {len(data_real)}\n"
              f"-" * 25 + "\n"
                          f"$\\bf{{K-S\\ Test}}$ (Overall):\n"
                          f"Statistic ($D$): {ks_stat:.4f}\n"
                          f"$p$-value: {ks_p:.2e}\n"
                          f"-" * 25 + "\n"
                                      f"$\\bf{{A-D\\ Test}}$ (Tail-sensitive):\n"
                                      f"Statistic: {ad_stat:.2f}\n"
                                      f"Sig. Level: {ad_p:.4f}")

# 将文本框放在左下角 (bottom left)
ax.text(0.03, 0.03, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#7f8c8d', alpha=0.9))

plt.tight_layout()
plt.savefig('ccdf_with_statistical_tests.png', dpi=300)
print("Plot saved as: ccdf_with_statistical_tests.png")
plt.show()