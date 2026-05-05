import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
print("全时段分布分析配置")
print("=" * 80)

# 1. 设置时间粒度
try:
    input_n = input("请输入分析的时间粒度（分钟），例如 10: ")
    INTERVAL_MINUTES = int(input_n) if input_n.strip() else 60
except ValueError:
    INTERVAL_MINUTES = 60
print(f"当前分析粒度: 每 {INTERVAL_MINUTES} 分钟为一个数据点")

# 2. 定义下雨时间段 (用于给全时段数据打标签)
rain_definitions = [
    {'start_hour': 120, 'end_hour': 140, 'label': '下雨时段1'},
    {'start_hour': 180, 'end_hour': 260, 'label': '下雨时段2'}
]

# ==========================================
# 2. 数据读取与预处理
# ==========================================
print("\n正在读取数据...")
try:
    df = pd.read_csv('数据/snapshot_SinglePoint_results_2s.csv')
    print(f"读取成功，共 {len(df)} 行。")
except FileNotFoundError:
    print("错误：找不到文件，生成模拟数据演示...")
    # 模拟数据：前半段晴天(Rice/Norm)，中间下雨(Weibull/Rayleigh)，后半段晴天
    dates = pd.date_range(start='2023-01-01', periods=100000, freq='10S')
    # 晴天：均值-85，类似高斯
    data = np.random.normal(-85, 1.5, 100000)
    # 下雨段1：均值下降，波动变大
    rain_mask = (np.linspace(0, 300, 100000) >= 120) & (np.linspace(0, 300, 100000) <= 140)
    data[rain_mask] = np.random.weibull(2, sum(rain_mask)) * 5 - 100
    df = pd.DataFrame({'datetime': dates, 'peak_power': data})

# 时间转换
df['datetime'] = pd.to_datetime(df['datetime'])
start_time = df['datetime'].min()

# 创建自定义分钟粒度的索引
df['period_index'] = ((df['datetime'] - start_time).dt.total_seconds() / (INTERVAL_MINUTES * 60)).astype(int)

# 清洗数据
df_clean = df.dropna(subset=['peak_power'])

# ==========================================
# 3. 准备索引映射 (用于快速查找某时刻是否下雨)
# ==========================================
rain_map = {}  # key: period_index, value: label
rain_ranges_info = []

print("\n时段标记映射:")
for definition in rain_definitions:
    s_idx = int(definition['start_hour'] * 60 / INTERVAL_MINUTES)
    e_idx = int(definition['end_hour'] * 60 / INTERVAL_MINUTES)

    # 记录每个下雨索引对应的标签
    for idx in range(s_idx, e_idx):
        rain_map[idx] = definition['label']

    hour_range_str = f"{definition['start_hour']}-{definition['end_hour']}h"
    print(f"  {definition['label']} ({hour_range_str}): Index {s_idx} -> {e_idx}")

# ==========================================
# 4. 定义分布 (包含莱斯分布)
# ==========================================
distributions = {
    '正态分布': stats.norm,
    '对数正态': stats.lognorm,
    '威布尔': stats.weibull_min,
    '指数分布': stats.expon,
    '伽马分布': stats.gamma,
    '瑞利分布': stats.rayleigh,
    '莱斯分布': stats.rice  # 新增 Rician
}

# ==========================================
# 5. 全时段逐个分析 (核心循环)
# ==========================================
print(f"\n开始全时段详细分析 (包括晴天)...")
print("这可能需要一些时间，取决于数据量...")

all_indices = sorted(df_clean['period_index'].unique())
detailed_results = []

total_steps = len(all_indices)
for i, idx in enumerate(all_indices):
    # 进度条
    if i % 50 == 0:
        sys.stdout.write(f"\r进度: {i}/{total_steps} ({(i / total_steps) * 100:.1f}%)")
        sys.stdout.flush()

    # 获取当前时间段数据
    segment = df_clean[df_clean['period_index'] == idx]['peak_power']
    if len(segment) < 10: continue

    # 判断当前标签
    current_label = rain_map.get(idx, '晴天')

    # 计算基础统计量
    stats_dict = {
        'period_index': idx,
        'time_offset_hours': idx * INTERVAL_MINUTES / 60,
        'label': current_label,
        'count': len(segment),
        'mean': segment.mean(),
        'std': segment.std(),
        'skew': stats.skew(segment),
        'kurtosis': stats.kurtosis(segment)
    }

    # 寻找最佳分布
    best_dist_name = "None"
    min_ks = 1.0

    # 为了性能，如果单段数据量过大，采样用于拟合检测
    fit_data = segment if len(segment) < 2000 else np.random.choice(segment, 2000)

    for name, dist in distributions.items():
        try:
            # 拟合参数
            params = dist.fit(fit_data)
            # K-S 检验
            ks_stat, _ = stats.kstest(fit_data, lambda x: dist.cdf(x, *params))

            if ks_stat < min_ks:
                min_ks = ks_stat
                best_dist_name = name
        except:
            continue

    stats_dict['best_dist'] = best_dist_name
    stats_dict['ks_stat'] = min_ks
    detailed_results.append(stats_dict)

print(f"\n\n分析完成! 生成了 {len(detailed_results)} 个时段的数据。")
res_df = pd.DataFrame(detailed_results)

# ==========================================
# 6. 可视化结果
# ==========================================
print("正在绘制全时段分析图表...")
fig = plt.figure(figsize=(24, 18))
plt.suptitle(f'全时段信号分布演变分析 (粒度: {INTERVAL_MINUTES}分钟, 含莱斯分布)', fontsize=16)

# 定义颜色映射
palette = {'晴天': 'gold', '下雨时段1': 'blue', '下雨时段2': 'darkblue'}

# 1. 均值全时段变化趋势
ax1 = plt.subplot(4, 1, 1)
sns.scatterplot(data=res_df, x='time_offset_hours', y='mean', hue='label', palette=palette, s=15, ax=ax1, linewidth=0)
# 加上平滑曲线
res_df['mean_smooth'] = res_df['mean'].rolling(window=5, center=True).mean()
ax1.plot(res_df['time_offset_hours'], res_df['mean_smooth'], color='black', alpha=0.3, label='5点移动平均')
ax1.set_title('全时段功率均值变化 (观察雨衰过程)')
ax1.set_ylabel('Mean Power (dBm)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. 标准差(波动)变化
ax2 = plt.subplot(4, 1, 2)
sns.scatterplot(data=res_df, x='time_offset_hours', y='std', hue='label', palette=palette, s=15, ax=ax2, linewidth=0)
ax2.set_title('全时段信号波动性(标准差)变化')
ax2.set_ylabel('Std Dev (dB)')
ax2.grid(True, alpha=0.3)

# 3. 最佳拟合分布随时间的变化 (散点图)
# 将分布名称映射为数字以便画图
dist_types = sorted(res_df['best_dist'].unique())
dist_map = {name: i for i, name in enumerate(dist_types)}
res_df['dist_code'] = res_df['best_dist'].map(dist_map)

ax3 = plt.subplot(4, 1, 3)
scatter = ax3.scatter(res_df['time_offset_hours'], res_df['dist_code'],
                      c=res_df['label'].map(
                          lambda x: {'晴天': 'gold', '下雨时段1': 'blue', '下雨时段2': 'darkblue'}.get(x, 'gray')),
                      alpha=0.6, s=20)
ax3.set_yticks(range(len(dist_types)))
ax3.set_yticklabels(dist_types)
ax3.set_title('最佳拟合分布类型随时间演变')
ax3.grid(True, axis='x', alpha=0.3)

# 4. 统计对比：晴天 vs 雨天 的分布构成 (两个饼图)
ax4_1 = plt.subplot(4, 2, 7)
ax4_2 = plt.subplot(4, 2, 8)

# 晴天分布统计
sunny_counts = res_df[res_df['label'] == '晴天']['best_dist'].value_counts()
if not sunny_counts.empty:
    ax4_1.pie(sunny_counts, labels=sunny_counts.index, autopct='%1.1f%%', startangle=90,
              colors=sns.color_palette("Pastel1"))
    ax4_1.set_title('晴天时段的最佳分布构成')
else:
    ax4_1.text(0.5, 0.5, "无晴天数据", ha='center')

# 雨天分布统计 (合并所有雨段)
rainy_counts = res_df[res_df['label'] != '晴天']['best_dist'].value_counts()
if not rainy_counts.empty:
    ax4_2.pie(rainy_counts, labels=rainy_counts.index, autopct='%1.1f%%', startangle=90,
              colors=sns.color_palette("Pastel1"))
    ax4_2.set_title('下雨时段的最佳分布构成')
else:
    ax4_2.text(0.5, 0.5, "无雨天数据", ha='center')

plt.tight_layout()
png_name = f'full_analysis_{INTERVAL_MINUTES}min_rice.png'
plt.savefig(png_name, dpi=300)
print(f"\n图表已保存为: {png_name}")

# 保存CSV
csv_name = f'full_stats_{INTERVAL_MINUTES}min_rice.csv'
res_df.drop(columns=['dist_code', 'mean_smooth'], errors='ignore').to_csv(csv_name, index=False, encoding='utf-8-sig')
print(f"详细数据已保存为: {csv_name}")

# 打印简要结论
print("\n" + "=" * 80)
print("分析摘要")
print("=" * 80)
print("分布占比 (Top 3):")
print(f"  晴天: {sunny_counts.head(3).to_dict() if not sunny_counts.empty else 'None'}")
print(f"  雨天: {rainy_counts.head(3).to_dict() if not rainy_counts.empty else 'None'}")