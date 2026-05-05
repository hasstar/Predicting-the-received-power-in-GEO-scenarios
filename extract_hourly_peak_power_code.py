import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def extract_hourly_peak_power():
    """
    根据小时级时间提取秒级数据的peak_power
    """
    print("开始提取小时级对应的秒级数据...")

    try:
        # 1. 读取两个文件
        print("读取文件...")
        df_hourly = pd.read_csv('test_set_predictions_time_only.csv')
        df_secondly = pd.read_csv('数据/snapshot_SinglePoint_results_2s.csv')

        print(f"小时级文件形状: {df_hourly.shape}")
        print(f"秒级文件形状: {df_secondly.shape}")

        # 2. 检查列名
        print("\n小时级文件列名:", list(df_hourly.columns))
        print("秒级文件列名:", list(df_secondly.columns))

        # 3. 转换时间列为datetime格式
        print("\n处理时间列...")

        # 查找时间列（可能有大小写问题）
        time_col_hourly = None
        time_col_secondly = None

        for col in df_hourly.columns:
            if 'time' in col.lower():
                time_col_hourly = col
                break
        if not time_col_hourly:
            time_col_hourly = df_hourly.columns[0]  # 默认第一列

        for col in df_secondly.columns:
            if 'time' in col.lower():
                time_col_secondly = col
                break
        if not time_col_secondly:
            time_col_secondly = df_secondly.columns[0]  # 默认第一列

        print(f"使用时间列 - 小时级: {time_col_hourly}, 秒级: {time_col_secondly}")

        # 4. 转换时间格式
        df_hourly[time_col_hourly] = pd.to_datetime(df_hourly[time_col_hourly])
        df_secondly[time_col_secondly] = pd.to_datetime(df_secondly[time_col_secondly])

        # 5. 按时间排序
        df_secondly = df_secondly.sort_values(time_col_secondly)
        df_hourly = df_hourly.sort_values(time_col_hourly)

        print(f"小时级时间范围: {df_hourly[time_col_hourly].min()} 到 {df_hourly[time_col_hourly].max()}")
        print(f"秒级时间范围: {df_secondly[time_col_secondly].min()} 到 {df_secondly[time_col_secondly].max()}")

        # 6. 查找peak_power列
        peak_power_col = None
        for col in df_secondly.columns:
            if 'peak' in col.lower() and 'power' in col.lower():
                peak_power_col = col
                break
        if not peak_power_col:
            # 如果没有找到，使用包含power的列
            power_cols = [col for col in df_secondly.columns if 'power' in col.lower()]
            if power_cols:
                peak_power_col = power_cols[0]
            else:
                raise ValueError("秒级文件中未找到peak_power列")

        print(f"使用功率列: {peak_power_col}")

        # 7. 为每个小时提取对应的秒级数据
        print("\n开始提取小时对应的秒级数据...")

        results = []

        for idx, hour_row in df_hourly.iterrows():
            hour_time = hour_row[time_col_hourly]
            next_hour = hour_time + timedelta(hours=1)

            # 提取该小时内的秒级数据
            mask = (df_secondly[time_col_secondly] >= hour_time) & \
                   (df_secondly[time_col_secondly] < next_hour)

            hourly_secondly_data = df_secondly[mask]

            if len(hourly_secondly_data) > 0:
                # 计算该小时的统计信息
                peak_power_values = hourly_secondly_data[peak_power_col]

                result = {
                    'hourly_time': hour_time,
                    'secondly_data_count': len(hourly_secondly_data),
                    'peak_power_mean': peak_power_values.mean(),
                    'peak_power_std': peak_power_values.std(),
                    'peak_power_min': peak_power_values.min(),
                    'peak_power_max': peak_power_values.max(),
                    'peak_power_median': peak_power_values.median(),
                    'peak_power_first': peak_power_values.iloc[0] if len(peak_power_values) > 0 else np.nan,
                    'peak_power_last': peak_power_values.iloc[-1] if len(peak_power_values) > 0 else np.nan
                }

                # 添加原始小时级数据的所有列
                for col in df_hourly.columns:
                    if col != time_col_hourly:  # 时间列已添加
                        result[f'hourly_{col}'] = hour_row[col]

                results.append(result)

                if idx < 3:  # 显示前3个处理示例
                    print(f"小时 {hour_time}: 找到 {len(hourly_secondly_data)} 个秒级数据点")
                    print(f"  Peak Power统计 - 均值: {result['peak_power_mean']:.2f}, "
                          f"标准差: {result['peak_power_std']:.2f}")

            else:
                print(f"警告: 小时 {hour_time} 没有对应的秒级数据")

        # 8. 创建结果DataFrame
        result_df = pd.DataFrame(results)

        print(f"\n处理完成!")
        print(f"成功处理 {len(result_df)} 个小时的数据")
        print(f"结果数据形状: {result_df.shape}")

        # 9. 保存结果
        output_file = 'hourly_peak_power_statistics.csv'
        result_df.to_csv(output_file, index=False)
        print(f"结果已保存到: {output_file}")

        # 显示结果预览
        print(f"\n结果文件前5行:")
        print(result_df.head())

        return result_df

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return None


def extract_detailed_secondly_data():
    """
    提取详细的秒级数据（保留所有秒级数据点）
    """
    print("\n" + "=" * 60)
    print("提取详细秒级数据")
    print("=" * 60)

    try:
        # 读取文件
        df_hourly = pd.read_csv('test_set_predictions_time_only.csv')
        df_secondly = pd.read_csv('数据/snapshot_SinglePoint_results_2s.csv')

        # 查找时间列
        time_col_hourly = [col for col in df_hourly.columns if 'time' in col.lower()][0]
        time_col_secondly = [col for col in df_secondly.columns if 'time' in col.lower()][0]

        # 转换时间
        df_hourly[time_col_hourly] = pd.to_datetime(df_hourly[time_col_hourly])
        df_secondly[time_col_secondly] = pd.to_datetime(df_secondly[time_col_secondly])

        # 查找peak_power列
        peak_power_col = [col for col in df_secondly.columns
                          if 'peak' in col.lower() and 'power' in col.lower()][0]

        # 为每个小时提取所有秒级数据点
        detailed_results = []

        for idx, hour_row in df_hourly.iterrows():
            hour_time = hour_row[time_col_hourly]
            next_hour = hour_time + timedelta(hours=1)

            # 提取该小时内的所有秒级数据
            mask = (df_secondly[time_col_secondly] >= hour_time) & \
                   (df_secondly[time_col_secondly] < next_hour)

            hourly_data = df_secondly[mask].copy()

            if len(hourly_data) > 0:
                # 为每个秒级数据点添加对应的小时信息
                hourly_data['hourly_reference_time'] = hour_time

                # 添加小时级数据的其他列
                for col in df_hourly.columns:
                    if col != time_col_hourly:
                        hourly_data[f'hourly_{col}'] = hour_row[col]

                detailed_results.append(hourly_data)

        # 合并所有结果
        if detailed_results:
            detailed_df = pd.concat(detailed_results, ignore_index=True)

            # 保存详细结果
            detailed_output = 'detailed_hourly_secondly_data.csv'
            detailed_df.to_csv(detailed_output, index=False)
            print(f"详细秒级数据已保存到: {detailed_output}")
            print(f"详细数据形状: {detailed_df.shape}")

            return detailed_df
        else:
            print("没有找到匹配的数据")
            return None

    except Exception as e:
        print(f"详细数据提取错误: {e}")
        return None


def analyze_time_coverage():
    """
    分析时间覆盖情况
    """
    print("\n" + "=" * 60)
    print("时间覆盖分析")
    print("=" * 60)

    df_hourly = pd.read_csv('test_set_predictions_time_only.csv')
    df_secondly = pd.read_csv('数据/snapshot_SinglePoint_results_2s.csv')

    time_col_hourly = [col for col in df_hourly.columns if 'time' in col.lower()][0]
    time_col_secondly = [col for col in df_secondly.columns if 'time' in col.lower()][0]

    df_hourly[time_col_hourly] = pd.to_datetime(df_hourly[time_col_hourly])
    df_secondly[time_col_secondly] = pd.to_datetime(df_secondly[time_col_secondly])

    print(f"小时级数据点数: {len(df_hourly)}")
    print(f"秒级数据点数: {len(df_secondly)}")
    print(f"秒级数据时间跨度: {(df_secondly[time_col_secondly].max() - df_secondly[time_col_secondly].min())}")

    # 检查每个小时的数据覆盖
    coverage_stats = []
    for hour_time in df_hourly[time_col_hourly]:
        next_hour = hour_time + timedelta(hours=1)
        mask = (df_secondly[time_col_secondly] >= hour_time) & \
               (df_secondly[time_col_secondly] < next_hour)
        count = mask.sum()
        coverage_stats.append(count)

    coverage_stats = pd.Series(coverage_stats)
    print(f"\n小时数据覆盖统计:")
    print(f"平均每个小时秒级数据点数: {coverage_stats.mean():.1f}")
    print(f"最少数据点: {coverage_stats.min()}")
    print(f"最多数据点: {coverage_stats.max()}")
    print(f"无数据的小时数: {(coverage_stats == 0).sum()}")

    return coverage_stats


# 主程序
if __name__ == "__main__":
    print("小时级-秒级数据提取工具")
    print("=" * 60)

    # 1. 分析时间覆盖
    coverage = analyze_time_coverage()

    # 2. 提取统计信息
    print("\n" + "=" * 60)
    result_stats = extract_hourly_peak_power()

    # 3. 提取详细数据（可选）
    print("\n" + "=" * 60)
    detailed_data = extract_detailed_secondly_data()

    print("\n处理完成！")