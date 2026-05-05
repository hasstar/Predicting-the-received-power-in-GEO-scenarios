import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# 设置绘图风格和字体
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans']  # 优先尝试Arial，没有则用黑体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 10)


def plot_prediction_vs_actual(pred_csv, actual_csv, actual_col_name, save_path='comparison_result.png'):
    """
    读取预测文件和真实值文件，进行画图对比
    :param pred_csv: 预测结果CSV路径 (包含 Predicted_Loss 列)
    :param actual_csv: 真实值CSV路径
    :param actual_col_name: 真实值CSV中对应的列名 (例如 'loss' 或 'target')
    :param save_path: 图片保存路径
    """
    print("=" * 60)
    print("准备绘图对比...")

    # 1. 读取数据
    try:
        df_pred = pd.read_csv(pred_csv)
        df_actual = pd.read_csv(actual_csv)
    except FileNotFoundError as e:
        print(f"[Error] 文件未找到: {e}")
        return

    # 2. 检查列是否存在
    if 'Predicted_Loss' not in df_pred.columns:
        # 如果你的预测结果列名不是 Predicted_Loss，请在这里修改，或者打印一下df_pred.columns查看
        print(f"[Error] 预测文件中找不到 'Predicted_Loss' 列。现有列: {list(df_pred.columns)}")
        return

    if actual_col_name not in df_actual.columns:
        print(f"[Error] 真实值文件中找不到 '{actual_col_name}' 列。现有列: {list(df_actual.columns)}")
        return

    # 3. 提取数据
    y_pred = df_pred['Predicted_Loss'].values
    y_true = df_actual[actual_col_name].values

    # 4. 数据对齐 (非常重要！)
    # 如果两个文件行数不一样，取最小的行数进行截断，或者报错提示
    min_len = min(len(y_pred), len(y_true))
    if len(y_pred) != len(y_true):
        print(f"[Warning] 数据长度不一致! 预测值: {len(y_pred)}, 真实值: {len(y_true)}")
        print(f"          将自动截取前 {min_len} 行进行对比...")
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]

    # 5. 计算评估指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"评估指标 -> RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # 6. 开始绘图 (创建 2x1 的子图)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # --- 子图 1: 折线走势对比 (Line Plot) ---
    # 为了防止数据太多挤在一起，如果数据超过200个点，只画前200个或者采样，这里默认画全部
    x_axis = range(len(y_true))

    ax1.plot(x_axis, y_true, label='Actual Value', color='black', alpha=0.7, linewidth=1.5)
    ax1.plot(x_axis, y_pred, label='Predicted Value', color='red', linestyle='--', alpha=0.8, linewidth=1.5)

    ax1.set_title(f'Prediction vs Actual Trend (RMSE={rmse:.4f})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # --- 子图 2: 散点回归图 (Scatter Plot) ---
    ax2.scatter(y_true, y_pred, color='blue', alpha=0.5, s=30, label='Data Point')

    # 画对角线 (完美预测线 y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Fit (y=x)')

    ax2.set_title(f'Scatter Plot: Actual vs Predicted (R^2={r2:.4f})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Actual Value', fontsize=12)
    ax2.set_ylabel('Predicted Value', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 7. 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Success] 图片已保存至: {save_path}")

    # 如果你在本地运行IDE (如Pycharm/VSCode)，可以取消下面这行的注释来直接显示
    plt.show()


if __name__ == "__main__":
    # ================= 配置区域 =================

    # 1. 刚才生成的预测结果文件
    my_pred_file = 'final_predictions.csv'

    # 2. 包含真实值(Ground Truth)的CSV文件
    #    (这里假设它在 'merged_data.csv' 里，你需要改成你实际的文件名)
    my_actual_file = '数据/merged_result_loss.csv'

    # 3. 真实值文件里的那一列叫什么名字？
    #    (请打开你的csv确认列头，比如 'loss', 'signal_loss', 'target' 等)
    my_target_column = 'loss'

    # ===========================================

    plot_prediction_vs_actual(
        pred_csv=my_pred_file,
        actual_csv=my_actual_file,
        actual_col_name=my_target_column,
        save_path='prediction_comparison_plot.png'
    )