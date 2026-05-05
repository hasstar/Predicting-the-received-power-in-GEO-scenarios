import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
import sys

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 用户配置区域
# ==========================================
print("=" * 80)
print("配置参数")
print("=" * 80)

# 在此处设置分析的时间粒度（n分钟）
try:
    input_n = input("请输入分析的时间粒度（分钟），例如 30: ")
    INTERVAL_MINUTES = int(input_n) if input_n.strip() else 60
except ValueError:
    INTERVAL_MINUTES = 60
    print("输入无效，默认使用 60 分钟")

print(f"当前分析粒度: 每 {INTERVAL_MINUTES} 分钟为一个数据点")

# 定义下雨的时间段（单位：小时，相对于数据起始时间）
# 这里保留小时是为了方便定义，代码会自动转换成对应的分钟粒度索引
rain_definitions = [
    {'start_hour': 120, 'end_hour': 140, 'label': '下雨时段1'},  # 120-140小时
    {'start_hour': 180, 'end_hour': 260, 'label': '下雨时段2'}  # 180-260小时
]

# ==========================================
# 2. 数据读取与处理
# ==========================================
print("\n正在读取数据...")
# 读取数据 (请确保路径正确)
try:
    df = pd.read_csv('数据/snapshot_SinglePoint_results_2s.csv')
except FileNotFoundError:
    print("错误：找不到文件 '数据/snapshot_SinglePoint_results_2s.csv'")
    # 为了演示代码逻辑，如果没有文件，这里生成模拟数据
    print("正在生成模拟数据以供演示...")
    dates = pd.date_range(start='2023-01-01', periods=200000, freq='2S')
    df = pd.DataFrame({'datetime': dates, 'peak_power': np.random.normal(-90, 2, 200000)})
    # 模拟下雨衰减
    mask_rain = (df.index > 120 * 1800) & (df.index < 140 * 1800)
    df.loc[mask_rain, 'peak_power'] -= 5

print(f"总共读取了 {len(df)} 行数据")

# 转换时间列
time_col = 'datetime'
df[time_col] = pd.to_datetime(df[time_col])
start_time = df[time_col].min()
print(f"时间范围: {start_time} 到 {df[time_col].max()}")

# 创建自定义分钟粒度的索引 (Period Index)
# 将总秒数除以 (INTERVAL_MINUTES * 60)
df['period_index'] = ((df[time_col] - start_time).dt.total_seconds() / (INTERVAL_MINUTES * 60)).astype(int)

# 移除NaN值
df_clean = df.dropna(subset=['peak_power'])
print(f"清洗后有效数据点: {len(df_clean)}")

# ==========================================
# 3. 标记下雨和晴天数据
# ==========================================
# 将定义的小时范围转换为 period_index 范围
rainy_indices_set = set()
rain_period_ranges = []

print(f"\n时段划分 (基于 {INTERVAL_MINUTES} 分钟粒度):")
for definition in rain_definitions:
    # 将小时转换为对应的 period index
    start_idx = int(definition['start_hour'] * 60 / INTERVAL_MINUTES)
    end_idx = int(definition['end_hour'] * 60 / INTERVAL_MINUTES)

    current_range = list(range(start_idx, end_idx))

    # 修正点：直接在这里生成字符串，供后面使用
    hour_range_str = f"{definition['start_hour']}-{definition['end_hour']}h"

    rain_period_ranges.append({
        'label': definition['label'],
        'indices': current_range,
        'hour_range': hour_range_str
    })
    rainy_indices_set.update(current_range)

    # 修正点：打印时使用上面生成的变量，而不是去字典里找不存在的键
    print(f"  {definition['label']} ({hour_range_str}): 索引 {start_idx} - {end_idx}")

rainy_indices_all = sorted(list(rainy_indices_set))

# 晴天时段（不在任何下雨索引中的时段）
all_period_indices = df_clean['period_index'].unique()
sunny_indices = [h for h in all_period_indices if h not in rainy_indices_all]

# 提取数据
rainy_data_all = df_clean[df_clean['period_index'].isin(rainy_indices_all)]['peak_power']
sunny_data = df_clean[df_clean['period_index'].isin(sunny_indices)]['peak_power']

# 提取各分段数据用于画图
rainy_data_parts = []
for rp in rain_period_ranges:
    part_data = df_clean[df_clean['period_index'].isin(rp['indices'])]['peak_power']
    rainy_data_parts.append((rp['label'], part_data))

print(f"\n数据点统计:")
print(f"  晴天: {len(sunny_data):,} 个数据点 (共 {len(sunny_indices)} 个 {INTERVAL_MINUTES}分钟段)")
print(f"  下雨天（全部）: {len(rainy_data_all):,} 个数据点 (共 {len(rainy_indices_all)} 个 {INTERVAL_MINUTES}分钟段)")

# ==========================================
# 4. 统计分析函数
# ==========================================
# 定义分布测试函数
distributions = {
    '正态分布': stats.norm,
    '对数正态': stats.lognorm,
    '威布尔': stats.weibull_min,
    '指数分布': stats.expon,
    '伽马分布': stats.gamma,
    '瑞利分布': stats.rayleigh,
}


def analyze_and_print(data, label):
    """分析并打印统计信息"""
    if len(data) == 0:
        print(f"\n{label}: 无数据")
        return None

    print(f"\n{'=' * 60}")
    print(f"{label} 分析")
    print(f"{'=' * 60}")
    print(f"均值: {data.mean():.4f} | 中位数: {data.median():.4f} | 标准差: {data.std():.4f}")
    print(f"偏度: {stats.skew(data):.4f} | 峰度: {stats.kurtosis(data):.4f}")

    # 分布拟合
    fit_results = []
    # 为了速度，如果数据量太大，进行下采样计算K-S
    calc_data = data if len(data) < 10000 else np.random.choice(data, 10000)

    for dist_name, distribution in distributions.items():
        try:
            params = distribution.fit(data)
            # KS test
            ks_stat, _ = stats.kstest(calc_data, lambda x: distribution.cdf(x, *params))
            fit_results.append({'dist': dist_name, 'ks': ks_stat, 'params': params})
        except:
            pass

    fit_results.sort(key=lambda x: x['ks'])
    if fit_results:
        best = fit_results[0]
        print(f"最佳拟合: {best['dist']} (KS={best['ks']:.4f})")
        return best['dist']
    return "Unknown"


# 分析各个时段
sunny_dist = analyze_and_print(sunny_data, "晴天")
rainy_dist = analyze_and_print(rainy_data_all, "下雨天（全部）")

# 统计检验
print(f"\n{'=' * 60}\n差异显著性检验 (晴天 vs 下雨)\n{'=' * 60}")
t_stat, t_p = stats.ttest_ind(sunny_data, rainy_data_all)
ks_stat, ks_p = stats.ks_2samp(sunny_data, rainy_data_all)
print(f"T检验 (均值差异): p-value = {t_p:.4e} {'(显著)' if t_p < 0.05 else '(不显著)'}")
print(f"K-S检验 (分布差异): p-value = {ks_p:.4e} {'(显著)' if ks_p < 0.05 else '(不显著)'}")

# ==========================================
# 5. 逐个时间段详细分析
# ==========================================
print(f"\n正在进行逐 {INTERVAL_MINUTES} 分钟时段分析...")

detailed_results = []
# 仅分析下雨时段，为了观察雨中变化
for idx in rainy_indices_all:
    segment = df_clean[df_clean['period_index'] == idx]['peak_power']
    if len(segment) < 10: continue  # 数据太少跳过

    # 确定属于哪个雨段
    period_label = "未知"
    for rp in rain_period_ranges:
        if idx in rp['indices']:
            period_label = rp['label']
            break

    # 简单拟合寻找最佳分布
    best_dist = "None"
    min_ks = 1.0
    for name, dist in distributions.items():
        try:
            params = dist.fit(segment)
            ks, _ = stats.kstest(segment, lambda x: dist.cdf(x, *params))
            if ks < min_ks:
                min_ks = ks
                best_dist = name
        except:
            pass

    detailed_results.append({
        'period_index': idx,
        'time_offset_hours': idx * INTERVAL_MINUTES / 60,  # 换算回小时方便看
        'label': period_label,
        'mean': segment.mean(),
        'std': segment.std(),
        'min': segment.min(),
        'max': segment.max(),
        'best_dist': best_dist,
        'ks_stat': min_ks
    })

res_df = pd.DataFrame(detailed_results)

# ==========================================
# 6. 可视化
# ==========================================
print("正在绘图...")
fig = plt.figure(figsize=(20, 15))
plt.suptitle(f'下雨 vs 晴天 分布特性分析 (粒度: {INTERVAL_MINUTES}分钟)', fontsize=16)

# 1. 整体分布直方图对比
ax1 = plt.subplot(3, 3, 1)
bins = np.linspace(df_clean['peak_power'].min(), df_clean['peak_power'].max(), 100)
ax1.hist(sunny_data, bins=bins, alpha=0.5, label='晴天', density=True, color='gold')
ax1.hist(rainy_data_all, bins=bins, alpha=0.5, label='下雨(全部)', density=True, color='blue')
ax1.set_title('整体概率密度直方图')
ax1.set_xlabel('Power (dBm)')
ax1.legend()

# 2. KDE 曲线对比 (观察形状)
ax2 = plt.subplot(3, 3, 2)
sns.kdeplot(sunny_data, ax=ax2, label='晴天', color='gold', fill=True, alpha=0.1)
for label, data in rainy_data_parts:
    sns.kdeplot(data, ax=ax2, label=label, fill=False, linewidth=2)
ax2.set_title('KDE 分布曲线形状对比')
ax2.legend()

# 3. 箱线图 (查看离群点和四分位数)
ax3 = plt.subplot(3, 3, 3)
plot_data = [sunny_data, rainy_data_all] + [d[1] for d in rainy_data_parts]
plot_labels = ['晴天', '下雨(全)'] + [d[0] for d in rainy_data_parts]
sns.boxplot(data=plot_data, ax=ax3, orient='v')
ax3.set_xticklabels(plot_labels, rotation=45)
ax3.set_title('箱线图：波动范围对比')

# 4. 下雨期间均值随时间变化
ax4 = plt.subplot(3, 1, 2)  # 占据中间一行
if not res_df.empty:
    # 绘制均值线
    sns.lineplot(data=res_df, x='period_index', y='mean', hue='label', marker='o', ax=ax4)
    # 绘制晴天参考线
    ax4.axhline(sunny_data.mean(), color='gold', linestyle='--', label='晴天均值', linewidth=2)
    # 绘制误差带 (Standard Deviation)
    ax4.fill_between(res_df['period_index'],
                     res_df['mean'] - res_df['std'],
                     res_df['mean'] + res_df['std'],
                     color='gray', alpha=0.2, label='±1 标准差')

    ax4.set_title(f'下雨期间信号均值变化趋势 (每点代表 {INTERVAL_MINUTES} 分钟)')
    ax4.set_xlabel(f'时间段索引 (1单位={INTERVAL_MINUTES}分钟)')
    ax4.set_ylabel('Peak Power (dBm)')
    ax4.legend(loc='upper right')
else:
    ax4.text(0.5, 0.5, '无下雨时段数据', ha='center')

# 5. 下雨期间分布类型的演变
ax5 = plt.subplot(3, 2, 5)
if not res_df.empty:
    dist_counts = res_df['best_dist'].value_counts()
    ax5.pie(dist_counts, labels=dist_counts.index, autopct='%1.1f%%', startangle=90)
    ax5.set_title(f'下雨期间各时段最佳拟合分布统计 (Total: {len(res_df)} 时段)')
else:
    ax5.text(0.5, 0.5, '无数据', ha='center')

# 6. 标准差(波动性)对比散点图
ax6 = plt.subplot(3, 2, 6)
if not res_df.empty:
    sns.scatterplot(data=res_df, x='period_index', y='std', hue='label', ax=ax6)
    ax6.axhline(sunny_data.std(), color='gold', linestyle='--', label='晴天波动水平')
    ax6.set_title('信号波动性 (标准差) 随时间变化')
    ax6.set_ylabel('Std Dev (dBm)')
else:
    ax6.text(0.5, 0.5, '无数据', ha='center')

plt.tight_layout()
filename = f'rain_analysis_{INTERVAL_MINUTES}min.png'
plt.savefig(filename, dpi=300)
print(f"\n图表已保存为: {filename}")

# 保存CSV
csv_filename = f'rain_hourly_stats_{INTERVAL_MINUTES}min.csv'
if not res_df.empty:
    res_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"详细统计数据已保存为: {csv_filename}")

print("\n分析完成。")