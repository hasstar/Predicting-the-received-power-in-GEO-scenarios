import pandas as pd
import numpy as np
from scipy import stats
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# ==========================================
# 0. 配置与数据加载
# ==========================================
# 定义粒度列表 (1到120)
granularities = list(range(1, 121))

# 定义候选分布
candidates = {
    'Gaussian': stats.norm,
    'Rice': stats.rice,
    'Rayleigh': stats.rayleigh,
    'Weibull': stats.weibull_min
}

print("正在读取数据...")
try:
    # 替换成你的真实路径
    df = pd.read_csv('数据/snapshot_SinglePoint_results_2s.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.dropna(subset=['peak_power'])
    print(f"读取成功，共 {len(df)} 行数据")
except FileNotFoundError:
    print("未找到文件，生成模拟数据用于演示逻辑...")
    dates = pd.date_range(start='2023-01-01', periods=100000, freq='2S')
    # 模拟：加上趋势和不同类型的噪声
    trend = np.sin(np.linspace(0, 20, 100000)) * 2
    # 混合 Weibull 和 Gaussian 特征
    signal = np.random.weibull(1.8, 100000) * -5 - 85 + trend
    df = pd.DataFrame({'datetime': dates, 'peak_power': signal})

# ==========================================
# 任务 1: 循环计算 1~120 分钟粒度的统计特征
# ==========================================
output_filename = 'granularity_analysis_summary.csv'
print(f"\n准备开始计算 {granularities[0]} 到 {granularities[-1]} 分钟粒度的分布特征...")
print(f"结果将实时保存至: {output_filename}")

# 初始化结果列表
all_results = []


# 定义保存函数
def save_current_results(data_list, filename):
    temp_df = pd.DataFrame(data_list)
    temp_df.to_csv(filename, index=False, encoding='utf-8-sig')


# 开始大循环
total_tasks = len(granularities)
for i, n_min in enumerate(granularities):
    print(f"[{i + 1}/{total_tasks}] 正在分析粒度: {n_min} 分钟...", end="")

    # 1. 按时间分组
    groups = df.groupby(pd.Grouper(key='datetime', freq=f'{n_min}min'))

    # 2. 初始化该粒度下的计数器
    # 记录谁是第一名
    winner_counts = {k: 0 for k in candidates.keys()}
    # 记录所有分布的 KS 值总和 (用于算平均匹配度)
    ks_sums = {k: 0.0 for k in candidates.keys()}
    # 记录该粒度下有多少个有效的时间段
    valid_segments = 0

    # 3. 遍历该粒度下的每一个时间切片
    for _, group in groups:
        data = group['peak_power'].values
        # 忽略数据过少的段
        if len(data) < 50: continue

        # 下采样优化速度 (对于KS检验，2000个点足够代表分布形状)
        fit_data = data #if len(data) < 2000 else np.random.choice(data, 2000)

        # 对当前段，测试所有分布
        segment_results = []

        for name, dist_func in candidates.items():
            try:
                params = dist_func.fit(fit_data)
                # 计算 KS 统计量
                ks_stat, _ = stats.kstest(fit_data, lambda x: dist_func.cdf(x, *params))

                segment_results.append({'name': name, 'ks': ks_stat})

                # 累加 KS 值
                ks_sums[name] += ks_stat
            except:
                # 拟合失败给一个很大的 KS 值
                segment_results.append({'name': name, 'ks': 1.0})
                ks_sums[name] += 1.0

        # 排序找到最佳分布 (KS 越小越好)
        segment_results.sort(key=lambda x: x['ks'])

        # 记录第一名 (Best Model)
        best_model = segment_results[0]['name']
        winner_counts[best_model] += 1

        # (可选) 你可以扩展逻辑记录 Top 2, Top 3，但在统计表中通常关注第一名

        valid_segments += 1

    # 4. 汇总该粒度的结果
    if valid_segments > 0:
        row = {
            'Granularity_Min': n_min,
            'Total_Segments': valid_segments
        }

        # 计算占比 (Proportion)
        for name in candidates.keys():
            row[f'Pct_{name}'] = (winner_counts[name] / valid_segments) * 100

        # 计算平均 KS 值 (Avg KS) - 反映该分布在该时间尺度下的普遍适应性
        for name in candidates.keys():
            row[f'Avg_KS_{name}'] = ks_sums[name] / valid_segments

        # 记录 Top 3 排名统计 (基于平均 KS 值排序)
        # 这里我们算出在这个粒度下，整体表现最好的前三名是谁
        avg_ks_ranking = sorted([(k, row[f'Avg_KS_{k}']) for k in candidates.keys()], key=lambda x: x[1])
        row['Rank1_Model'] = avg_ks_ranking[0][0]
        row['Rank2_Model'] = avg_ks_ranking[1][0]
        row['Rank3_Model'] = avg_ks_ranking[2][0]

        all_results.append(row)
        print(f" 完成. (最佳: {row['Rank1_Model']})")
    else:
        print(f" 跳过 (无有效数据)")

    # 每计算完一个粒度就保存一次，防止程序崩溃
    save_current_results(all_results, output_filename)

print(f"\n任务 1 完成！所有统计数据已保存至 {output_filename}")

# ==========================================
# 任务 2: 寻找并保存最具代表性的 15分钟 数据片段
# ==========================================
target_freq_min = 15
print(f"\n任务 2: 正在扫描 {target_freq_min} 分钟粒度的数据，寻找 Weibull 优于 Gaussian 的典型片段...")

groups_15 = list(df.groupby(pd.Grouper(key='datetime', freq=f'{target_freq_min}min')))

best_example_data = None
best_example_time = None
max_diff_score = -1
found = False

# 步长采样遍历，提高搜索速度
step = max(1, int(len(groups_15) / 100))

for i in range(0, len(groups_15), step):
    time_key, group = groups_15[i]
    data = group['peak_power'].values

    if len(data) < 100: continue

    try:
        # 拟合 Weibull
        p_w = stats.weibull_min.fit(data)
        ks_w, _ = stats.kstest(data, lambda x: stats.weibull_min.cdf(x, *p_w))

        # 拟合 Gaussian
        p_n = stats.norm.fit(data)
        ks_n, _ = stats.kstest(data, lambda x: stats.norm.cdf(x, *p_n))

        # 我们寻找：Gaussian 拟合得不好(KS大)，但 Weibull 拟合得好(KS小) 的差距最大的时刻
        # 这样画出来的图最有说服力
        diff = ks_n - ks_w

        if diff > max_diff_score:
            max_diff_score = diff
            best_example_data = group[['datetime', 'peak_power']].copy()  # 保存完整列
            best_example_time = time_key
            found = True
    except:
        continue

sample_filename = 'sample_15min_data.csv'
if found and best_example_data is not None:
    best_example_data.to_csv(sample_filename, index=False, encoding='utf-8-sig')
    print(f"已找到最佳片段 (时间: {best_example_time}, 差异分: {max_diff_score:.4f})")
    print(f"数据已保存至: {sample_filename}")
else:
    print("未找到满足条件的片段，保存第一段有效数据作为备选。")
    if len(groups_15) > 0:
        groups_15[0][1][['datetime', 'peak_power']].to_csv(sample_filename, index=False)

print("\n所有处理结束。")