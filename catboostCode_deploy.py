# --- START OF FILE predict_model.py ---

import pandas as pd
import numpy as np
import json
import sys
from catboost import CatBoostRegressor


class SatelliteLossPredictor:
    def __init__(self, model_path='final_model.cbm', feature_path='final_features.json'):
        """
        初始化预测器：加载模型和特征列表
        """
        print(f"Loading model from {model_path}...")
        try:
            self.model = CatBoostRegressor()
            self.model.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        print(f"Loading feature list from {feature_path}...")
        try:
            with open(feature_path, 'r', encoding='utf-8') as f:
                self.features = json.load(f)
            print(f"Model expects {len(self.features)} features.")
        except Exception as e:
            print(f"Error loading feature list: {e}")
            sys.exit(1)

    def predict(self, input_data):
        """
        执行预测
        :param input_data: 字典 (单个样本) 或 DataFrame (批量样本)
        :return: 预测结果 (numpy array)
        """
        # 1. 将输入转换为 DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input data must be a dictionary or a pandas DataFrame")

        # 2. 检查并填充缺失特征 (防止输入漏了某些特征)
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            print(f"Warning: Input data is missing {len(missing_features)} features. Filling with 0.")
            for f in missing_features:
                df[f] = 0.0

        # 3. 严格按照训练时的特征顺序排列列 (关键步骤！)
        try:
            df_sorted = df[self.features]
        except KeyError as e:
            print(f"Error: Data alignment failed. {e}")
            return None

        # 4. 执行预测
        predictions = self.model.predict(df_sorted)
        return predictions


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 实例化预测器
    predictor = SatelliteLossPredictor()

    print("\n--- 单样本预测示例 ---")
    # 假设这是你从其他系统或手动输入的参数
    # 注意：这里的键名(Key)必须和你特征工程里的列名一致
    # 你可以打开 final_features.json 查看具体的特征名称
    single_input = {
        'lspf': 1013.25,
        'o3_17-28km': 25.0,
        'o3_8-17km': 12.5,
        "vilwn": 12.5,
        "viwve": 12.5
        # ... 不需要输入所有特征，代码会自动补0，但建议尽可能全
    }

    result = predictor.predict(single_input)
    print(f"预测的信号损耗 (Loss): {result[0]:.4f}")

    print("\n--- 批量预测示例 (从CSV读取) ---")
    # 如果你有一个包含新数据的CSV文件
    try:
        # 假设你有一个 new_data.csv，里面包含了很多环境参数
        # df_new = pd.read_csv('new_data.csv')

        # 这里为了演示，我们模拟创建一个DataFrame
        df_new = pd.DataFrame([
            {'Ground_Pressure': 1010, 'wind_speed_0-8km': 10},
            {'Ground_Pressure': 1005, 'wind_speed_0-8km': 20},
            {'Ground_Pressure': 1020, 'wind_speed_0-8km': 5}
        ])

        results = predictor.predict(df_new)

        # 将结果拼接到原数据后面
        df_new['Predicted_Loss'] = results
        print(df_new[['Ground_Pressure', 'wind_speed_0-8km', 'Predicted_Loss']])

        # 保存结果
        # df_new.to_csv('prediction_results.csv', index=False)

    except Exception as e:
        print(f"Batch prediction example skipped: {e}")