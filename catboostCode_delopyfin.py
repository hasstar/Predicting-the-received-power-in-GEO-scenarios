import pandas as pd
import numpy as np
import json
from catboost import CatBoostRegressor
import os


class SatellitePredictor:
    def __init__(self, model_path='final_model.cbm', feature_path='final_features.json'):
        """
        初始化：加载模型和特征列表
        """
        self.model_path = model_path
        self.feature_path = feature_path
        self.model = None
        self.required_features = []

        self._load_resources()

    def _load_resources(self):
        # 1. 加载特征列表
        if not os.path.exists(self.feature_path):
            raise FileNotFoundError(f"找不到特征文件: {self.feature_path}")

        with open(self.feature_path, 'r', encoding='utf-8') as f:
            self.required_features = json.load(f)
        print(f"[Init] 模型需要 {len(self.required_features)} 个特征: {self.required_features}")

        # 2. 加载模型
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"找不到模型文件: {self.model_path}")

        self.model = CatBoostRegressor()
        self.model.load_model(self.model_path)
        print(f"[Init] 模型加载成功")

    def process_files(self, ground_file, layer_files):
        """
        读取6个CSV文件，合并并重命名列，使其与训练时的格式一致
        """
        print("\n[Step 1] 读取并处理原始数据文件...")

        # 1. 读取地面数据
        print(f"  - 读取地面数据: {ground_file}")
        df_ground = pd.read_csv(ground_file)

        # 2. 读取高空分层数据并重命名列 (关键步骤！)
        # 必须与训练时的后缀保持完全一致，否则模型找不到特征
        layer_names = ['0-8km', '8-17km', '17-28km', '28-37km', '37-50km']

        if len(layer_files) != 5:
            raise ValueError(f"高空层文件必须是5个，当前提供了 {len(layer_files)} 个")

        layer_dfs = []
        for i, file_path in enumerate(layer_files):
            suffix = layer_names[i]
            print(f"  - 读取 {suffix} 层: {file_path}")
            df = pd.read_csv(file_path)

            # 重命名列：除了 pressure_level (如果有的话)，其他都加上后缀
            # 逻辑复现：col -> col_suffix
            new_columns = {col: f"{col}_{suffix}" for col in df.columns if col != 'pressure_level'}
            df = df.rename(columns=new_columns)
            layer_dfs.append(df)

        # 3. 横向合并所有数据 (假设所有文件的行是一一对应的)
        # axis=1 表示按列合并
        print("  - 合并所有数据...")
        df_all = pd.concat([df_ground] + layer_dfs, axis=1)
        print(f"  [OK] 合并后总维度: {df_all.shape}")

        return df_all

    def extract_features_and_predict(self, df_merged, output_file='prediction_result.csv'):
        """
        从合并的大表中提取模型所需的特定特征，并进行预测
        """
        print("\n[Step 2] 提取特征并预测...")

        # 1. 检查缺失特征
        missing_features = [f for f in self.required_features if f not in df_merged.columns]

        if missing_features:
            print(f"  [WARNING] 数据中缺少以下 {len(missing_features)} 个特征 (将自动补0):")
            print(f"  {missing_features}")
            # 缺失特征补0，保证程序不崩溃
            for f in missing_features:
                df_merged[f] = 0.0

        # 2. 提取并排序 (必须严格按照 required_features 的顺序)
        X_input = df_merged[self.required_features]
        print(f"  [OK] 已提取输入矩阵，维度: {X_input.shape}")
        print(f"       使用的特征: {list(X_input.columns)}")

        # 3. 预测
        predictions = self.model.predict(X_input)

        # 4. 保存结果
        # 我们创建一个结果DataFrame，包含预测值，如果原数据有时间戳或ID，也可以加进来
        df_result = pd.DataFrame()
        # 如果原数据有 'time' 或 'id' 列，可以取消下面注释保留它以便对照
        # if 'time' in df_merged.columns:
        #     df_result['time'] = df_merged['time']

        df_result['Predicted_Loss'] = predictions

        df_result.to_csv(output_file, index=False)
        print(f"\n[Step 3] 预测完成！")
        print(f"  结果已保存至: {output_file}")
        print(f"  预测值预览 (前5行):\n{df_result.head()}")


# ========================================================
# 主程序入口
# ========================================================
if __name__ == "__main__":
    # 1. 定义你的文件路径 (请修改为你实际的文件名)
    ground_csv = '数据/all_ground_atmosphere_data_merged_new_processed.csv'  # 地面数据

    # 这一组文件的顺序必须严格按照高度从低到高排列！
    # 对应: 0-8km, 8-17km, 17-28km, 28-37km, 37-50km
    layer_csvs = [
        '数据/atmosphere_region_39_9_116_3_0-8km_processed.csv',
        '数据/atmosphere_region_39_8_116_3_8-17km_processed.csv',
        '数据/atmosphere_region_39_7_116_4_17-28km_processed.csv',
        '数据/atmosphere_region_39_6_116_4_28-37km_processed.csv',
        '数据/atmosphere_region_39_5_116_4_37-50km_processed.csv'
    ]

    # 为了演示，我先生成一些假数据文件 (你实际使用时不需要这段代码)
    # -----------------------------------------------------------
    # create_dummy_files(ground_csv, layer_csvs) # <--- 你运行时请注释掉这一行
    # -----------------------------------------------------------

    try:
        # 2. 初始化预测器
        predictor = SatellitePredictor(
            model_path='final_model.cbm',  # 你的模型文件
            feature_path='final_features.json'  # 你的特征列表文件
        )

        # 3. 处理文件 (读取 -> 重命名 -> 合并)
        df_merged = predictor.process_files(ground_csv, layer_csvs)

        # 4. 提取特征并预测
        predictor.extract_features_and_predict(df_merged, output_file='final_predictions.csv')

    except Exception as e:
        print(f"\n[ERROR] 发生错误: {str(e)}")
        # 提示：如果是文件找不到，请检查文件名
        # 提示：如果报错 KeyError，说明JSON里的特征名在CSV里找不到，检查列名是否一致