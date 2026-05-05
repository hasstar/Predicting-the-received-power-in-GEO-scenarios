import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import sys

warnings.filterwarnings('ignore')

# ==========================================
# 1. 审美与参数配置
# ==========================================
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

# Weibull 参数
K_PARAM = 4.8068
LAMBDA_PARAM = 5.9156
CONST_OFFSET = -79.38

# 文件路径
FILE_DETAILED = 'detailed_hourly_secondly_data.csv'
FILE_HOURLY_PRED = 'test_set_predictions_time_only.csv'
OUTPUT_FILE = 'comparison_results.csv'

# ==========================================
# 2. 数据读取与对齐
# ==========================================
print("1. Reading and aligning data...")

try:
    # 读取数据
    df_detail = pd.read_csv(FILE_DETAILED)
    df_pred = pd.read_csv(FILE_HOURLY_PRED)

    # 转换时间格式
    df_detail['datetime'] = pd.to_datetime(df_detail['datetime'])
    df_pred['datetime'] = pd.to_datetime(df_pred['datetime'])

    # 创建对齐键：将秒级时间向下取整到小时
    df_detail['join_hour'] = df_detail['datetime'].dt.floor('H')
    df_pred['join_hour'] = df_pred['datetime']  # 假设预测文件已经是整点

    # 合并数据 (Left Join)
    df_merged = pd.merge(df_detail,
                         df_pred[['join_hour', 'Predicted_Loss']],
                         on='join_hour',
                         how='left')

    # 删除没有对应预测值的行
    initial_len = len(df_merged)
    df_merged = df_merged.dropna(subset=['Predicted_Loss'])
    if len(df_merged) < initial_len:
        print(f"   Warning: Dropped {initial_len - len(df_merged)} rows due to missing prediction data.")

except FileNotFoundError:
    print("Error: Input files not found. generating dummy data for demonstration...")
    # 模拟数据
    N = 10000
    df_merged = pd.DataFrame({
        'peak_power': np.random.normal(-90, 2, N),
        'Predicted_Loss': np.repeat([-0.075, -0.065], N // 2)
    })

# ==========================================
# 3. 计算对比信号
# ==========================================
print("2. Calculating signals...")

# A. 计算实测变换值 (Target)
# 公式: -79.38 - peak_power
df_merged['Real_Transformed'] = CONST_OFFSET - df_merged['peak_power']

# B. 计算预测扩展值 (Expanded)
# 1. 生成 Weibull 噪声
num_samples = len(df_merged)
weibull_noise = stats.weibull_min.rvs(K_PARAM, scale=LAMBDA_PARAM, size=num_samples)

# 2. 公式: Predicted_Loss + Weibull
df_merged['Weibull_Noise'] = weibull_noise
df_merged['Simulated_Val'] = df_merged['Predicted_Loss'] + df_merged['Weibull_Noise']-6

# 保存结果
df_export = df_merged[['peak_power', 'Predicted_Loss', 'Real_Transformed', 'Simulated_Val']]
df_export.to_csv(OUTPUT_FILE, index=False)
print(f"   Data saved to {OUTPUT_FILE}")

# ==========================================
# 4. 绘图 (连续点数坐标)
# ==========================================
print("3. Plotting comparison...")

# 为了图表可读性，如果数据量太大(超过2万点)，我们只画前 5000 个点来观察细节
# 或者你可以修改这里的 slice_end = len(df_merged) 来画全部
slice_end = min(len(df_merged), 50000)
df_plot = df_merged.iloc[:slice_end]

# 生成连续的横坐标 (点数)
x_axis = np.arange(len(df_plot))

fig, ax = plt.subplots(figsize=(12, 6))

# 画实测变换曲线 (蓝色)
ax.plot(x_axis, df_plot['Real_Transformed'],
        color='#2980B9', linewidth=1, alpha=0.7,
        label='Measured Target (-79.38 - PeakPower)')

# 画预测扩展曲线 (红色)
ax.plot(x_axis, df_plot['Simulated_Val'],
        color='#C0392B', linewidth=1, alpha=0.7,
        label=f'Simulated (Pred + Weibull)')

# 计算简单的统计量放在标题或图中
diff_mean = (df_plot['Simulated_Val'] - df_plot['Real_Transformed']).mean()
rmse = np.sqrt(((df_plot['Simulated_Val'] - df_plot['Real_Transformed']) ** 2).mean())

# 装饰图表
ax.set_title(f'Similarity Check: Measured Transformed vs. Weibull Expanded\n(First {slice_end} points)', fontsize=14,
             fontweight='bold')
ax.set_xlabel('Sample Index (Points)', fontsize=12, fontweight='bold')
ax.set_ylabel('Value (dB)', fontsize=12, fontweight='bold')

# 添加统计信息框
stats_text = (f"Comparison Stats:\n"
              f"Mean Diff: {diff_mean:.2f} dB\n"
              f"RMSE: {rmse:.2f} dB")
ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

# 图例
ax.legend(loc='upper right', frameon=False, fontsize=11)

# 美化坐标轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('similarity_check_plot.png', dpi=300)
print("4. Plot saved as: similarity_check_plot.png")
plt.show()

# 打印数值对比
print("\n数值统计对比:")
print(
    f"实测变换值 (Real) - 均值: {df_merged['Real_Transformed'].mean():.4f}, 方差: {df_merged['Real_Transformed'].var():.4f}")
print(
    f"预测扩展值 (Sim)  - 均值: {df_merged['Simulated_Val'].mean():.4f}, 方差: {df_merged['Simulated_Val'].var():.4f}")