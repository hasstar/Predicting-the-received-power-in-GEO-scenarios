import pandas as pd
import numpy as np
import os


def add_time_to_predictions():
    """
    修复文件路径问题
    """
    print("开始处理文件...")

    # 定义正确的文件路径
    predictions_file = 'test_set_predictions_only.csv'
    loss_file = '数据/loss_data_and_1h_time.csv'  # 注意子目录路径

    # 检查文件是否存在
    print("检查文件是否存在...")
    print(f"预测文件: {predictions_file} - 存在: {os.path.exists(predictions_file)}")
    print(f"时间文件: {loss_file} - 存在: {os.path.exists(loss_file)}")

    if not os.path.exists(predictions_file):
        # 尝试在数据目录中查找
        alt_path = '数据/test_set_predictions_only.csv'
        if os.path.exists(alt_path):
            predictions_file = alt_path
            print(f"使用替代路径: {predictions_file}")

    if not os.path.exists(loss_file):
        # 尝试在根目录中查找
        alt_path = 'loss_data_and_1h_time.csv'
        if os.path.exists(alt_path):
            loss_file = alt_path
            print(f"使用替代路径: {loss_file}")

    try:
        # 1. 读取文件（使用Excel兼容编码）
        print(f"\n读取 {predictions_file}...")
        df_predictions = pd.read_csv(predictions_file, encoding='gbk')  # Excel常用编码
        print(f"预测文件形状: {df_predictions.shape}")

        print(f"读取 {loss_file}...")
        df_loss = pd.read_csv(loss_file, encoding='gbk')  # Excel常用编码
        print(f"时间文件形状: {df_loss.shape}")

        # 2. 显示列名确认
        print("\n预测文件列名:", list(df_predictions.columns))
        print("时间文件列名:", list(df_loss.columns))

        # 3. 查找需要的列
        # 查找Actual_Loss列
        actual_loss_col = None
        for col in df_predictions.columns:
            if 'actual' in col.lower() and 'loss' in col.lower():
                actual_loss_col = col
                break
        if not actual_loss_col:
            # 如果找不到，使用第一个包含loss的列
            loss_cols = [col for col in df_predictions.columns if 'loss' in col.lower()]
            if loss_cols:
                actual_loss_col = loss_cols[0]
            else:
                actual_loss_col = df_predictions.columns[0]  # 使用第一列作为备选

        # 查找loss列
        loss_col = None
        for col in df_loss.columns:
            if 'loss' in col.lower():
                loss_col = col
                break
        if not loss_col:
            loss_col = df_loss.columns[0]  # 使用第一列作为备选

        # 查找时间列
        time_col = None
        for col in df_loss.columns:
            if any(word in col.lower() for word in ['time', 'date', 'datetime', 'timestamp']):
                time_col = col
                break
        if not time_col:
            # 如果没有时间列，使用第二列作为备选
            time_col = df_loss.columns[1] if len(df_loss.columns) > 1 else df_loss.columns[0]

        print(f"\n使用的列:")
        print(f"预测文件: {actual_loss_col}")
        print(f"时间文件: {loss_col}, {time_col}")

        # 4. 显示数据预览
        print("\n预测文件前3行:")
        print(df_predictions[[actual_loss_col]].head(3))
        print("\n时间文件前3行:")
        print(df_loss[[loss_col, time_col]].head(3))

        # 5. 数据匹配
        print(f"\n开始匹配数据...")
        result_df = df_predictions.copy()
        result_df[time_col] = np.nan  # 添加时间列

        matched_count = 0
        tolerance = 1e-8  # 浮点数容差

        for idx, row in df_predictions.iterrows():
            actual_loss = row[actual_loss_col]

            # 查找匹配的loss值
            match_mask = np.isclose(df_loss[loss_col], actual_loss, atol=tolerance)
            matches = df_loss[match_mask]

            if len(matches) > 0:
                matched_time = matches.iloc[0][time_col]
                result_df.at[idx, time_col] = matched_time
                matched_count += 1

                if matched_count <= 3:
                    print(f"匹配 #{matched_count}: {actual_loss:.6f} -> {matched_time}")

        # 6. 统计结果
        total_rows = len(result_df)
        match_rate = (matched_count / total_rows) * 100

        print(f"\n匹配结果统计:")
        print(f"总行数: {total_rows}")
        print(f"成功匹配: {matched_count}")
        print(f"匹配率: {match_rate:.2f}%")

        # 检查未匹配的行
        unmatched_count = result_df[time_col].isna().sum()
        if unmatched_count > 0:
            print(f"未匹配的行数: {unmatched_count}")
            print("前5个未匹配的值:")
            print(result_df[result_df[time_col].isna()][actual_loss_col].head().values)

        # 7. 保存结果
        output_file = 'test_set_predictions_time_only.csv'
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")

        # 显示结果预览
        print(f"\n结果文件前5行:")
        print(result_df.head())

        return result_df

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return None


def list_available_files():
    """列出当前目录下的文件"""
    print("\n当前目录文件列表:")
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            print(f"  📄 {file}")

    if os.path.exists('数据'):
        print("\n数据目录文件列表:")
        for file in os.listdir('数据'):
            if file.endswith('.csv'):
                print(f"  📄 数据/{file}")


# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("CSV文件时间匹配工具（路径修复版）")
    print("=" * 60)

    # 首先列出可用文件
    list_available_files()

    # 然后执行匹配
    result = add_time_to_predictions()

    if result is not None:
        print("\n✅ 处理完成！")
    else:
        print("\n❌ 处理失败，请检查文件路径和内容")