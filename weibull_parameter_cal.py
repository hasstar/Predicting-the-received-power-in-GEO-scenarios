import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import make_interp_spline
import warnings
import sys

warnings.filterwarnings('ignore')

# ==========================================
# 1. 审美配置
# ==========================================
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2

# ==========================================
# 2. 数据读取
# ==========================================
print("Reading data...")
try:
    df = pd.read_csv('数据/snapshot_SinglePoint_results_2s.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.dropna(subset=['peak_power'])
    print(f"Data loaded: {len(df)} rows.")
except FileNotFoundError:
    print("Error: File not found.")
    sys.exit()

# ==========================================
# 3. 循环计算参数演变 (1min -> 120min)
# ==========================================
max_granularity = 120
results = []

print(f"Analyzing Weibull parameters for granularity 1 to {max_granularity} min...")

for n in range(1, max_granularity + 1):
    # 进度提示
    if n % 10 == 0:
        print(f"Processing: {n}/{max_granularity} min...")

    groups = df.groupby(pd.Grouper(key='datetime', freq=f'{n}min'))

    k_values = []
    lambda_values = []

    for _, group in groups:
        data_segment = group['peak_power'].values
        if len(data_segment) < 30: continue

        try:
            # 拟合
            shape, loc, scale = stats.weibull_min.fit(data_segment)

            # 【修正点】：Scale (lambda) 必须是正数。
            # Shape (k) 通常在 0.1 到 10 之间。
            if 0.1 < shape < 20 and scale > 0:
                k_values.append(shape)
                lambda_values.append(scale)
        except:
            continue

    # 只有当该粒度下计算出了有效值，才记录
    if k_values:
        results.append({
            'granularity': n,
            'k_mean': np.mean(k_values),
            'k_std': np.std(k_values),
            'lambda_mean': np.mean(lambda_values),
            'lambda_std': np.std(lambda_values)
        })

# 【安全检查】：防止没有任何数据生成导致报错
if not results:
    print("Error: No valid Weibull parameters could be fitted. Please check your data.")
    sys.exit()

df_params = pd.DataFrame(results)
# 保存数据
df_params.to_csv('weibull_params_evolution.csv', index=False)
print(f"Calculation done. Generated {len(df_params)} rows of parameter data.")

# ==========================================
# 4. 绘图 (Dual Plot)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

x = df_params['granularity']


def plot_trend(ax, x_data, y_mean, y_std, color, label, ylabel):
    # 原始散点
    ax.scatter(x_data, y_mean, color=color, alpha=0.15, s=10)

    # B-Spline 平滑
    if len(x_data) > 3:
        x_new = np.linspace(x_data.min(), x_data.max(), 300)
        try:
            spl_mean = make_interp_spline(x_data, y_mean, k=3)
            y_mean_smooth = spl_mean(x_new)

            spl_std = make_interp_spline(x_data, y_std, k=3)
            y_std_smooth = spl_std(x_new)
        except:
            # 如果插值失败，退化为线性
            x_new = x_data
            y_mean_smooth = y_mean
            y_std_smooth = y_std
    else:
        x_new, y_mean_smooth, y_std_smooth = x_data, y_mean, y_std

    # 误差带
    ax.fill_between(x_new,
                    y_mean_smooth - y_std_smooth,
                    y_mean_smooth + y_std_smooth,
                    color=color, alpha=0.15, label='$\pm$1 Std Dev')

    # 曲线
    ax.plot(x_new, y_mean_smooth, color='white', linewidth=5, alpha=1.0)
    ax.plot(x_new, y_mean_smooth, color=color, linewidth=3, label=f'Mean {label}')

    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
    ax.legend(loc='upper right', frameon=False, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


# 绘制 k (Shape)
plot_trend(ax1, x, df_params['k_mean'], df_params['k_std'],
           '#C0392B', 'Shape ($k$)', 'Shape Parameter ($k$)')
#ax1.set_title('Evolution of Weibull Parameters vs. Time Granularity', fontsize=16, fontweight='bold', pad=20)

# 绘制 lambda (Scale)
plot_trend(ax2, x, df_params['lambda_mean'], df_params['lambda_std'],
           '#2980B9', 'Scale ($\lambda$)', 'Scale Parameter ($\lambda$)')

ax2.set_xlabel('Time Granularity (min)', fontsize=18, fontweight='bold')
ax2.set_xlim(0, max_granularity)

plt.tight_layout()
plt.savefig('Weibull_Params_Evolution.pdf', dpi=300)
print("Plot saved as: Weibull_Params_Evolution.png")
plt.show()