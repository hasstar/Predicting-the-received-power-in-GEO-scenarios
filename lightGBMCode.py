"""
===================================================================================
高轨卫星信号损耗预测 - LightGBM模型训练与递归特征消除(RFE)
===================================================================================

【任务描述】
使用大气参数预测高轨卫星接收信号的损耗值，通过递归特征消除(RFE)找到最优特征子集

【理论基础】
1. LightGBM (Light Gradient Boosting Machine)
   - 是一种基于决策树的梯度提升框架
   - 采用histogram算法，训练速度快，内存占用低
   - 支持类别特征，自动处理缺失值
   - 适合大规模数据和高维特征

2. 递归特征消除 (Recursive Feature Elimination, RFE)
   - 从所有特征开始，训练模型
   - 根据特征重要性排序，每次消除最不重要的特征
   - 重复上述过程，直到剩余目标数量的特征
   - 帮助找到最小且最优的特征子集

3. 评估指标
   - R² (决定系数): 衡量模型对数据变异的解释程度，范围[0,1]，越接近1越好
   - RMSE (均方根误差): 预测值与真实值的偏差，越小越好
   - MAE (平均绝对误差): 预测误差的平均值，越小越好
   - MAPE (平均绝对百分比误差): 相对误差的百分比，越小越好

【作者】AI Assistant
【日期】2025-11-12
===================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import warnings
from typing import List, Dict, Tuple
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置中文字体，避免图表中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class LightGBMRFETrainer:
    """
    LightGBM递归特征消除训练器

    该类实现了完整的训练流程，包括：
    - 数据加载与预处理
    - 特征集成（并集方式）
    - 递归特征消除
    - 模型训练与评估
    - 结果可视化
    """

    def __init__(self, test_size=0.4, random_state=42):
        """
        初始化训练器

        参数:
            test_size: 测试集比例，默认0.4 (训练集60%)
            random_state: 随机种子，确保结果可复现
        """
        self.test_size = test_size
        self.random_state = random_state
        self.all_features = []  # 所有特征的并集
        self.feature_sources = {}  # 记录每个特征来自哪些方法
        self.rfe_results = []  # 存储RFE每一步的结果

        print("=" * 80)
        print("LightGBM递归特征消除训练器初始化完成")
        print(f"训练集比例: {1 - test_size:.0%}, 测试集比例: {test_size:.0%}")
        print(f"随机种子: {random_state}")
        print("=" * 80)

    def load_feature_selection_results(self, feature_dir='特征'):
        """
        加载所有特征选择方法的结果，并取并集

        【理论】特征集成 - 并集方式
        - 将5种特征选择方法的结果合并
        - 只要某个特征在任一方法中出现，就保留
        - 优点：覆盖全面，不遗漏重要特征
        - 缺点：可能包含冗余特征，但通过RFE可以消除

        参数:
            feature_dir: 特征文件所在目录
        """
        print("\n【步骤1】加载特征选择结果")
        print("-" * 80)

        # 定义5个特征选择方法的文件
        feature_files = {
            'Lasso': 'Lasso_top_35_features.csv',
            'RF': 'RF_selected_features_top35.csv',
            'MIC_minepy': 'MIC_minepy_top35_features.csv',
            'MIC_sklearn': 'MIC_sklearn_top35_features.csv',
            'Spearman': 'Spearman_top35_features.csv'
        }

        features_by_method = {}

        # 逐个读取每种方法的特征
        for method_name, filename in feature_files.items():
            filepath = f"{feature_dir}/{filename}"

            try:
                df = pd.read_csv(filepath)

                # 根据不同文件格式提取特征名
                if 'feature' in df.columns:
                    features = df['feature'].tolist()
                elif 'Feature' in df.columns:
                    features = df['Feature'].tolist()
                else:
                    print(f"  警告: {filename} 没有找到特征列")
                    continue

                # 过滤掉非字符串或空值
                features = [f for f in features if isinstance(f, str) and f.strip()]

                features_by_method[method_name] = features
                print(f"  [OK] {method_name:15s}: {len(features):2d} 个特征")

            except Exception as e:
                print(f"  [ERROR] 读取 {filename} 失败: {str(e)}")

        # 计算并集
        all_features_set = set()
        for features in features_by_method.values():
            all_features_set.update(features)

        self.all_features = sorted(list(all_features_set))

        # 记录每个特征来自哪些方法
        for feature in self.all_features:
            sources = [method for method, features in features_by_method.items()
                       if feature in features]
            self.feature_sources[feature] = sources

        print(f"\n  特征并集统计:")
        print(f"  - 总特征数: {len(self.all_features)}")
        print(f"  - 各方法特征数: {[len(f) for f in features_by_method.values()]}")

        # 统计特征被多少方法选中
        selection_counts = {}
        for feature, sources in self.feature_sources.items():
            count = len(sources)
            if count not in selection_counts:
                selection_counts[count] = []
            selection_counts[count].append(feature)

        print(f"\n  特征选择频次分布:")
        for count in sorted(selection_counts.keys(), reverse=True):
            print(f"  - 被 {count} 种方法选中: {len(selection_counts[count])} 个特征")

        return features_by_method

    def load_data(self, loss_file='数据/merged_result_loss.csv',
                  ground_file='数据/all_ground_atmosphere_data_merged_new_processed.csv',
                  layer_files=None):
        """
        加载并合并所有数据

        【数据说明】
        - 目标变量: loss (卫星信号损耗值)
        - 特征变量: 地面大气参数 + 5层高空大气参数

        参数:
            loss_file: 损耗数据文件
            ground_file: 地面大气数据文件
            layer_files: 高空大气数据文件列表
        """
        print("\n【步骤2】加载原始数据")
        print("-" * 80)

        if layer_files is None:
            layer_files = [
                '数据/atmosphere_region_39_9_116_3_0-8km_processed.csv',
                '数据/atmosphere_region_39_8_116_3_8-17km_processed.csv',
                '数据/atmosphere_region_39_7_116_4_17-28km_processed.csv',
                '数据/atmosphere_region_39_6_116_4_28-37km_processed.csv',
                '数据/atmosphere_region_39_5_116_4_37-50km_processed.csv'
            ]

        # 1. 加载损耗数据（目标变量）
        print("\n  加载目标变量 (loss)...")
        df_loss = pd.read_csv(loss_file)
        y = df_loss['loss'].values
        print(f"  [OK] 样本数: {len(y)}")
        print(f"  [OK] 损耗范围: [{y.min():.3f}, {y.max():.3f}]")
        print(f"  [OK] 损耗均值: {y.mean():.3f} ± {y.std():.3f}")

        # 2. 加载地面大气数据
        print("\n  加载地面大气参数...")
        df_ground = pd.read_csv(ground_file)
        print(f"  [OK] 地面特征数: {len(df_ground.columns)}")

        # 3. 加载高空大气数据
        print("\n  加载高空大气参数...")
        layer_dfs = []
        layer_names = ['0-8km', '8-17km', '17-28km', '28-37km', '37-50km']

        for i, layer_file in enumerate(layer_files):
            df_layer = pd.read_csv(layer_file)
            # 给列名添加后缀，区分不同层
            df_layer.columns = [f"{col}_{layer_names[i]}" if col != 'pressure_level'
                                else col for col in df_layer.columns]
            layer_dfs.append(df_layer)
            print(f"  [OK] {layer_names[i]:10s} 层: {len(df_layer.columns)} 个特征")

        # 4. 合并所有数据
        print("\n  合并所有数据...")
        df_all = pd.concat([df_ground] + layer_dfs, axis=1)

        # 5. 只保留我们选择的特征
        available_features = [f for f in self.all_features if f in df_all.columns]
        missing_features = [f for f in self.all_features if f not in df_all.columns]

        if missing_features:
            print(f"\n  警告: {len(missing_features)} 个特征在数据中不存在:")
            for feat in missing_features[:5]:  # 只显示前5个
                print(f"    - {feat}")
            if len(missing_features) > 5:
                print(f"    ... 还有 {len(missing_features) - 5} 个")

        X = df_all[available_features].values
        self.all_features = available_features  # 更新为实际可用的特征

        print(f"\n  数据合并完成:")
        print(f"  [OK] 样本数: {X.shape[0]}")
        print(f"  [OK] 可用特征数: {X.shape[1]}")
        print(f"  [OK] 缺失值数量: {np.isnan(X).sum()}")

        # 处理缺失值（如果有）
        if np.isnan(X).any():
            print(f"  处理缺失值...")
            X = np.nan_to_num(X, nan=0.0)

        self.X = X
        self.y = y

        return X, y

    def split_data(self):
        """
        划分训练集和测试集

        【理论】数据划分
        - 使用随机划分，确保训练集和测试集的分布相似
        - 设置random_state确保结果可复现
        - 训练集用于模型学习，测试集用于评估泛化能力
        """
        print("\n【步骤3】划分训练集和测试集")
        print("-" * 80)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        print(f"  训练集: {len(self.X_train)} 样本 ({len(self.X_train) / len(self.X):.1%})")
        print(f"  测试集: {len(self.X_test)} 样本 ({len(self.X_test) / len(self.X):.1%})")
        print(f"\n  训练集损耗统计: {self.y_train.mean():.3f} ± {self.y_train.std():.3f}")
        print(f"  测试集损耗统计: {self.y_test.mean():.3f} ± {self.y_test.std():.3f}")

    def calculate_metrics(self, y_true, y_pred):
        """
        计算评估指标

        【评估指标详解】
        1. R² (决定系数)
           - 公式: R² = 1 - (SS_res / SS_tot)
           - SS_res: 残差平方和, SS_tot: 总平方和
           - 含义: 模型解释了多少数据变异性
           - 范围: (-∞, 1], 1表示完美拟合

        2. RMSE (均方根误差)
           - 公式: RMSE = sqrt(mean((y_true - y_pred)²))
           - 含义: 预测值与真实值的典型偏差
           - 单位: 与目标变量相同

        3. MAE (平均绝对误差)
           - 公式: MAE = mean(|y_true - y_pred|)
           - 含义: 平均预测误差
           - 对异常值不敏感

        4. MAPE (平均绝对百分比误差)
           - 公式: MAPE = mean(|y_true - y_pred| / |y_true|) × 100%
           - 含义: 相对误差百分比
           - 便于不同量级数据的比较
        """
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # 计算MAPE，避免除零
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        return {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

    def train_lightgbm(self, X_train, y_train, X_test, y_test, feature_names):
        """
        训练LightGBM模型

        【LightGBM参数说明】
        - objective: 'regression' - 回归任务
        - metric: 'rmse' - 使用RMSE作为优化目标
        - num_leaves: 31 - 树的叶子节点数，控制模型复杂度
        - learning_rate: 0.05 - 学习率，越小训练越慢但可能效果更好
        - feature_fraction: 0.8 - 每次迭代随机选择80%的特征，防止过拟合
        - bagging_fraction: 0.8 - 每次迭代使用80%的数据，防止过拟合
        - bagging_freq: 5 - 每5次迭代进行一次bagging
        - verbose: -1 - 不输出训练日志
        - n_estimators: 500 - 最大迭代次数
        - early_stopping_rounds: 50 - 50轮无提升则停止训练
        """
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data,
                                 feature_name=feature_names)

        # 设置参数
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'force_col_wise': True
        }

        # 训练模型
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)  # 不输出日志
            ]
        )

        return model

    def recursive_feature_elimination(self):
        """
        执行递归特征消除(RFE)

        【RFE算法流程】
        1. 使用所有特征训练模型
        2. 评估模型性能
        3. 获取特征重要性排名
        4. 移除最不重要的特征
        5. 重复步骤1-4，直到只剩1个特征

        【特征重要性说明】
        LightGBM的特征重要性基于:
        - split: 特征被用于分裂的次数
        - gain: 特征带来的信息增益
        这里我们使用gain，因为它更能反映特征的实际贡献
        """
        print("\n【步骤4】递归特征消除(RFE)")
        print("-" * 80)
        print("\n  RFE算法说明:")
        print("  1. 从所有特征开始训练模型")
        print("  2. 评估模型在训练集和测试集上的性能")
        print("  3. 根据特征重要性，移除最不重要的特征")
        print("  4. 重复上述过程，直到只剩1个特征")
        print("  5. 找到性能最优的特征子集")
        print()

        current_features = self.all_features.copy()
        current_X_train = self.X_train.copy()
        current_X_test = self.X_test.copy()

        total_iterations = len(current_features)

        # 从所有特征开始，逐步减少到1个
        for iteration in range(total_iterations):
            n_features = len(current_features)

            print(f"  迭代 {iteration + 1}/{total_iterations}: 使用 {n_features} 个特征")

            # 训练模型
            model = self.train_lightgbm(
                current_X_train, self.y_train,
                current_X_test, self.y_test,
                current_features
            )

            # 预测
            y_train_pred = model.predict(current_X_train)
            y_test_pred = model.predict(current_X_test)

            # 计算指标
            train_metrics = self.calculate_metrics(self.y_train, y_train_pred)
            test_metrics = self.calculate_metrics(self.y_test, y_test_pred)

            # 获取特征重要性
            importance_dict = dict(zip(current_features,
                                       model.feature_importance(importance_type='gain')))

            # 保存结果
            result = {
                'iteration': iteration + 1,
                'n_features': n_features,
                'features': current_features.copy(),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': importance_dict.copy(),
                'model': model
            }
            self.rfe_results.append(result)

            # 输出当前结果
            print(f"    训练集 R^2={train_metrics['R2']:.4f}, "
                  f"RMSE={train_metrics['RMSE']:.4f}")
            print(f"    测试集 R^2={test_metrics['R2']:.4f}, "
                  f"RMSE={test_metrics['RMSE']:.4f}")

            # 如果只剩1个特征，停止
            if n_features == 1:
                break

            # 找到最不重要的特征并移除
            least_important_feature = min(importance_dict, key=importance_dict.get)
            least_important_idx = current_features.index(least_important_feature)

            print(f"    移除特征: {least_important_feature} "
                  f"(重要性: {importance_dict[least_important_feature]:.2f})")

            # 更新特征列表和数据
            current_features.pop(least_important_idx)
            current_X_train = np.delete(current_X_train, least_important_idx, axis=1)
            current_X_test = np.delete(current_X_test, least_important_idx, axis=1)
            print()

        print(f"  RFE完成! 共进行了 {len(self.rfe_results)} 次迭代")

    def find_best_model(self):
        """
        找到性能最优的模型

        【模型选择策略】
        主要看测试集R²，因为它反映了模型的泛化能力
        同时考虑特征数量，在性能相近的情况下，优先选择特征少的模型
        """
        print("\n【步骤5】寻找最优模型")
        print("-" * 80)

        # 按测试集R²排序
        sorted_results = sorted(self.rfe_results,
                                key=lambda x: x['test_metrics']['R2'],
                                reverse=True)

        best_result = sorted_results[0]

        print(f"\n  最优模型 (测试集R^2最高):")
        print(f"  - 特征数量: {best_result['n_features']}")
        print(f"  - 测试集R^2: {best_result['test_metrics']['R2']:.4f}")
        print(f"  - 测试集RMSE: {best_result['test_metrics']['RMSE']:.4f}")
        print(f"  - 测试集MAE: {best_result['test_metrics']['MAE']:.4f}")
        print(f"  - 测试集MAPE: {best_result['test_metrics']['MAPE']:.2f}%")

        print(f"\n  使用的特征:")
        for i, feat in enumerate(best_result['features'], 1):
            importance = best_result['feature_importance'][feat]
            print(f"    {i:2d}. {feat:30s} (重要性: {importance:8.2f})")

        self.best_result = best_result
        return best_result

    def plot_rfe_curves(self, save_path='RFE_performance_curves.png'):
        """
        绘制RFE性能曲线

        【图表说明】
        展示随着特征数量减少，模型性能的变化趋势
        包括4个子图：R², RMSE, MAE, MAPE
        """
        print("\n【步骤6】生成RFE性能曲线图")
        print("-" * 80)

        # 提取数据
        n_features_list = [r['n_features'] for r in self.rfe_results]

        train_r2 = [r['train_metrics']['R2'] for r in self.rfe_results]
        test_r2 = [r['test_metrics']['R2'] for r in self.rfe_results]

        train_rmse = [r['train_metrics']['RMSE'] for r in self.rfe_results]
        test_rmse = [r['test_metrics']['RMSE'] for r in self.rfe_results]

        train_mae = [r['train_metrics']['MAE'] for r in self.rfe_results]
        test_mae = [r['test_metrics']['MAE'] for r in self.rfe_results]

        train_mape = [r['train_metrics']['MAPE'] for r in self.rfe_results]
        test_mape = [r['test_metrics']['MAPE'] for r in self.rfe_results]

        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. R² 曲线
        ax1 = axes[0, 0]
        ax1.plot(n_features_list, train_r2, 'o-', label='训练集', linewidth=2, markersize=4)
        ax1.plot(n_features_list, test_r2, 's-', label='测试集', linewidth=2, markersize=4)
        ax1.set_xlabel('特征数量', fontsize=12)
        ax1.set_ylabel('R² (决定系数)', fontsize=12)
        ax1.set_title('R² vs 特征数量\n(越高越好，表示模型解释数据变异性的能力)',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(right=max(n_features_list) + 2)

        # 标记最优点
        best_idx = test_r2.index(max(test_r2))
        ax1.plot(n_features_list[best_idx], test_r2[best_idx], 'r*',
                 markersize=20, label=f'最优 ({n_features_list[best_idx]}特征)')
        ax1.legend(fontsize=11)

        # 2. RMSE 曲线
        ax2 = axes[0, 1]
        ax2.plot(n_features_list, train_rmse, 'o-', label='训练集', linewidth=2, markersize=4)
        ax2.plot(n_features_list, test_rmse, 's-', label='测试集', linewidth=2, markersize=4)
        ax2.set_xlabel('特征数量', fontsize=12)
        ax2.set_ylabel('RMSE (均方根误差)', fontsize=12)
        ax2.set_title('RMSE vs 特征数量\n(越低越好，表示预测值与真实值的典型偏差)',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(right=max(n_features_list) + 2)

        # 标记最优点
        best_idx_rmse = test_rmse.index(min(test_rmse))
        ax2.plot(n_features_list[best_idx_rmse], test_rmse[best_idx_rmse], 'r*',
                 markersize=20, label=f'最优 ({n_features_list[best_idx_rmse]}特征)')
        ax2.legend(fontsize=11)

        # 3. MAE 曲线
        ax3 = axes[1, 0]
        ax3.plot(n_features_list, train_mae, 'o-', label='训练集', linewidth=2, markersize=4)
        ax3.plot(n_features_list, test_mae, 's-', label='测试集', linewidth=2, markersize=4)
        ax3.set_xlabel('特征数量', fontsize=12)
        ax3.set_ylabel('MAE (平均绝对误差)', fontsize=12)
        ax3.set_title('MAE vs 特征数量\n(越低越好，表示平均预测误差)',
                      fontsize=13, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(right=max(n_features_list) + 2)

        # 4. MAPE 曲线
        ax4 = axes[1, 1]
        ax4.plot(n_features_list, train_mape, 'o-', label='训练集', linewidth=2, markersize=4)
        ax4.plot(n_features_list, test_mape, 's-', label='测试集', linewidth=2, markersize=4)
        ax4.set_xlabel('特征数量', fontsize=12)
        ax4.set_ylabel('MAPE (%)', fontsize=12)
        ax4.set_title('MAPE vs 特征数量\n(越低越好，表示相对预测误差百分比)',
                      fontsize=13, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(right=max(n_features_list) + 2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] 性能曲线图已保存: {save_path}")
        plt.close()

    def plot_feature_importance(self, save_path='feature_importance_top20.png'):
        """
        绘制最优模型的特征重要性
        """
        print("\n【步骤7】生成特征重要性图")
        print("-" * 80)

        best_result = self.best_result
        importance_dict = best_result['feature_importance']

        # 按重要性排序
        sorted_features = sorted(importance_dict.items(),
                                 key=lambda x: x[1], reverse=True)

        # 只显示前20个特征
        top_n = min(20, len(sorted_features))
        top_features = sorted_features[:top_n]

        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]

        # 绘制水平条形图
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        bars = ax.barh(range(len(features)), importances, color=colors)

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('特征重要性 (Gain)', fontsize=12)
        ax.set_title(f'Top {top_n} 特征重要性\n(基于最优模型: {best_result["n_features"]}个特征)',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # 反转y轴，使最重要的特征在顶部
        ax.invert_yaxis()

        # 在条形上添加数值
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            ax.text(imp, i, f' {imp:.1f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] 特征重要性图已保存: {save_path}")
        plt.close()

    def plot_prediction_scatter(self, save_path='prediction_scatter.png'):
        """
        绘制预测值vs真实值散点图

        【图表解读】
        - 理想情况：所有点都在y=x线上
        - 点越接近y=x线，预测越准确
        - 点的分布反映了模型的预测误差
        """
        print("\n【步骤8】生成预测散点图")
        print("-" * 80)

        best_model = self.best_result['model']

        # 获取最优模型使用的特征索引
        feature_indices = [self.all_features.index(f)
                           for f in self.best_result['features']]

        X_train_selected = self.X_train[:, feature_indices]
        X_test_selected = self.X_test[:, feature_indices]

        # 预测
        y_train_pred = best_model.predict(X_train_selected)
        y_test_pred = best_model.predict(X_test_selected)

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # 训练集散点图
        ax1.scatter(self.y_train, y_train_pred, alpha=0.6, s=50,
                    c='blue', edgecolors='black', linewidth=0.5)
        ax1.plot([self.y_train.min(), self.y_train.max()],
                 [self.y_train.min(), self.y_train.max()],
                 'r--', linewidth=2, label='理想预测线 (y=x)')

        train_r2 = self.best_result['train_metrics']['R2']
        train_rmse = self.best_result['train_metrics']['RMSE']

        ax1.set_xlabel('真实损耗值', fontsize=12)
        ax1.set_ylabel('预测损耗值', fontsize=12)
        ax1.set_title(f'训练集预测结果\nR²={train_r2:.4f}, RMSE={train_rmse:.4f}',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 测试集散点图
        ax2.scatter(self.y_test, y_test_pred, alpha=0.6, s=50,
                    c='green', edgecolors='black', linewidth=0.5)
        ax2.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 'r--', linewidth=2, label='理想预测线 (y=x)')

        test_r2 = self.best_result['test_metrics']['R2']
        test_rmse = self.best_result['test_metrics']['RMSE']

        ax2.set_xlabel('真实损耗值', fontsize=12)
        ax2.set_ylabel('预测损耗值', fontsize=12)
        ax2.set_title(f'测试集预测结果\nR²={test_r2:.4f}, RMSE={test_rmse:.4f}',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] 预测散点图已保存: {save_path}")
        plt.close()

    def plot_residual_analysis(self, save_path='residual_analysis.png'):
        """
        绘制残差分析图

        【残差分析】
        残差 = 真实值 - 预测值
        - 残差应该随机分布在0附近
        - 如果残差有明显模式，说明模型存在系统性误差
        - 残差的正态分布检验可以验证模型假设
        """
        print("\n【步骤9】生成残差分析图")
        print("-" * 80)

        best_model = self.best_result['model']
        feature_indices = [self.all_features.index(f)
                           for f in self.best_result['features']]

        X_test_selected = self.X_test[:, feature_indices]
        y_test_pred = best_model.predict(X_test_selected)

        # 计算残差
        residuals = self.y_test - y_test_pred

        # 创建2x2子图
        fig = plt.figure(figsize=(16, 12))

        # 1. 残差vs预测值
        ax1 = plt.subplot(2, 2, 1)
        ax1.scatter(y_test_pred, residuals, alpha=0.6, s=50,
                    c='blue', edgecolors='black', linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('预测值', fontsize=12)
        ax1.set_ylabel('残差 (真实值 - 预测值)', fontsize=12)
        ax1.set_title('残差 vs 预测值\n(残差应随机分布在0附近)',
                      fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. 残差直方图
        ax2 = plt.subplot(2, 2, 2)
        ax2.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('残差', fontsize=12)
        ax2.set_ylabel('频数', fontsize=12)
        ax2.set_title(f'残差分布直方图\n均值={residuals.mean():.4f}, 标准差={residuals.std():.4f}',
                      fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Q-Q图 (检验残差是否服从正态分布)
        ax3 = plt.subplot(2, 2, 3)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q图 (正态性检验)\n(点越接近红线，越符合正态分布)',
                      fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. 残差绝对值排序
        ax4 = plt.subplot(2, 2, 4)
        sorted_abs_residuals = np.sort(np.abs(residuals))
        ax4.plot(sorted_abs_residuals, 'o-', linewidth=1, markersize=4)
        ax4.set_xlabel('样本索引 (按残差绝对值排序)', fontsize=12)
        ax4.set_ylabel('残差绝对值', fontsize=12)
        ax4.set_title('残差绝对值排序图\n(识别异常样本)',
                      fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # 标记最大的几个残差
        top_n = 5
        top_indices = np.argsort(np.abs(residuals))[-top_n:]
        for idx in top_indices:
            ax1.scatter(y_test_pred[idx], residuals[idx],
                        color='red', s=200, marker='x', linewidths=3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] 残差分析图已保存: {save_path}")
        plt.close()

    def save_results_summary(self, save_path='training_results_summary.txt'):
        """
        保存详细的结果摘要
        """
        print("\n【步骤10】保存结果摘要")
        print("-" * 80)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("高轨卫星信号损耗预测 - LightGBM训练结果摘要\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 1. 数据集信息
            f.write("【1. 数据集信息】\n")
            f.write("-" * 80 + "\n")
            f.write(f"总样本数: {len(self.X)}\n")
            f.write(f"训练集样本数: {len(self.X_train)} ({len(self.X_train) / len(self.X):.1%})\n")
            f.write(f"测试集样本数: {len(self.X_test)} ({len(self.X_test) / len(self.X):.1%})\n")
            f.write(f"初始特征数: {len(self.all_features)}\n")
            f.write(f"目标变量范围: [{self.y.min():.3f}, {self.y.max():.3f}]\n")
            f.write(f"目标变量均值±标准差: {self.y.mean():.3f} ± {self.y.std():.3f}\n\n")

            # 2. 最优模型信息
            f.write("【2. 最优模型】\n")
            f.write("-" * 80 + "\n")
            best = self.best_result
            f.write(f"最优特征数量: {best['n_features']}\n\n")

            f.write("训练集性能:\n")
            for metric, value in best['train_metrics'].items():
                if metric == 'MAPE':
                    f.write(f"  {metric:6s}: {value:8.2f}%\n")
                else:
                    f.write(f"  {metric:6s}: {value:8.4f}\n")

            f.write("\n测试集性能:\n")
            for metric, value in best['test_metrics'].items():
                if metric == 'MAPE':
                    f.write(f"  {metric:6s}: {value:8.2f}%\n")
                else:
                    f.write(f"  {metric:6s}: {value:8.4f}\n")

            f.write("\n最优特征列表:\n")
            sorted_features = sorted(best['feature_importance'].items(),
                                     key=lambda x: x[1], reverse=True)
            for i, (feat, imp) in enumerate(sorted_features, 1):
                sources = ', '.join(self.feature_sources.get(feat, ['未知']))
                f.write(f"  {i:2d}. {feat:30s} (重要性: {imp:8.2f}) [来源: {sources}]\n")

            # 3. RFE全过程结果
            f.write("\n【3. RFE全过程结果】\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'迭代':>4s} {'特征数':>6s} {'训练R²':>9s} {'测试R²':>9s} "
                    f"{'训练RMSE':>10s} {'测试RMSE':>10s} {'训练MAE':>9s} {'测试MAE':>9s}\n")
            f.write("-" * 80 + "\n")

            for result in self.rfe_results:
                f.write(f"{result['iteration']:4d} {result['n_features']:6d} "
                        f"{result['train_metrics']['R2']:9.4f} "
                        f"{result['test_metrics']['R2']:9.4f} "
                        f"{result['train_metrics']['RMSE']:10.4f} "
                        f"{result['test_metrics']['RMSE']:10.4f} "
                        f"{result['train_metrics']['MAE']:9.4f} "
                        f"{result['test_metrics']['MAE']:9.4f}\n")

            # 4. 性能Top10模型
            f.write("\n【4. 性能Top10模型 (按测试集R²排序)】\n")
            f.write("-" * 80 + "\n")
            sorted_results = sorted(self.rfe_results,
                                    key=lambda x: x['test_metrics']['R2'],
                                    reverse=True)[:10]

            f.write(f"{'排名':>4s} {'特征数':>6s} {'测试R²':>9s} {'测试RMSE':>10s} "
                    f"{'测试MAE':>9s} {'测试MAPE':>10s}\n")
            f.write("-" * 80 + "\n")

            for rank, result in enumerate(sorted_results, 1):
                f.write(f"{rank:4d} {result['n_features']:6d} "
                        f"{result['test_metrics']['R2']:9.4f} "
                        f"{result['test_metrics']['RMSE']:10.4f} "
                        f"{result['test_metrics']['MAE']:9.4f} "
                        f"{result['test_metrics']['MAPE']:9.2f}%\n")

            # 5. 特征来源统计
            f.write("\n【5. 特征来源统计】\n")
            f.write("-" * 80 + "\n")
            source_count = {}
            for sources in self.feature_sources.values():
                for source in sources:
                    source_count[source] = source_count.get(source, 0) + 1

            for source, count in sorted(source_count.items(),
                                        key=lambda x: x[1], reverse=True):
                f.write(f"  {source:15s}: {count:3d} 个特征\n")

            # 6. 结论与建议
            f.write("\n【6. 结论与建议】\n")
            f.write("-" * 80 + "\n")

            best_r2 = best['test_metrics']['R2']
            best_rmse = best['test_metrics']['RMSE']

            if best_r2 >= 0.9:
                performance = "优秀"
            elif best_r2 >= 0.8:
                performance = "良好"
            elif best_r2 >= 0.7:
                performance = "中等"
            else:
                performance = "待提升"

            f.write(f"1. 模型性能评价: {performance}\n")
            f.write(f"   - 测试集R²={best_r2:.4f}，说明模型解释了{best_r2 * 100:.1f}%的数据变异性\n")
            f.write(f"   - 测试集RMSE={best_rmse:.4f}，预测误差约为真实值的"
                    f"{best_rmse / self.y_test.mean() * 100:.1f}%\n\n")

            f.write(f"2. 特征选择结果:\n")
            f.write(f"   - 从{len(self.all_features)}个特征优化到{best['n_features']}个特征\n")
            f.write(f"   - 特征减少了{(1 - best['n_features'] / len(self.all_features)) * 100:.1f}%\n")
            f.write(f"   - 保持了{best_r2 * 100:.1f}%的预测准确性\n\n")

            f.write("3. 改进建议:\n")
            if best_r2 < 0.9:
                f.write("   - 可以尝试更多的特征工程（交互特征、多项式特征等）\n")
                f.write("   - 调整模型超参数（网格搜索或贝叶斯优化）\n")
                f.write("   - 尝试其他算法（XGBoost、CatBoost、神经网络等）\n")
            f.write("   - 收集更多数据样本，提高模型泛化能力\n")
            f.write("   - 分析误差较大的样本，寻找数据质量问题\n")

        print(f"  [OK] 结果摘要已保存: {save_path}")

    def save_results_to_excel(self, save_path='training_results_detailed.xlsx'):
        """
        将所有结果保存到Excel文件
        """
        print("\n【步骤11】保存详细结果到Excel")
        print("-" * 80)

        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            # Sheet 1: RFE全过程
            rfe_data = []
            for result in self.rfe_results:
                row = {
                    '迭代': result['iteration'],
                    '特征数': result['n_features'],
                    '训练集R²': result['train_metrics']['R2'],
                    '测试集R²': result['test_metrics']['R2'],
                    '训练集RMSE': result['train_metrics']['RMSE'],
                    '测试集RMSE': result['test_metrics']['RMSE'],
                    '训练集MAE': result['train_metrics']['MAE'],
                    '测试集MAE': result['test_metrics']['MAE'],
                    '训练集MAPE': result['train_metrics']['MAPE'],
                    '测试集MAPE': result['test_metrics']['MAPE'],
                }
                rfe_data.append(row)

            df_rfe = pd.DataFrame(rfe_data)
            df_rfe.to_excel(writer, sheet_name='RFE过程', index=False)

            # Sheet 2: 最优模型特征
            best_features = []
            for feat, imp in sorted(self.best_result['feature_importance'].items(),
                                    key=lambda x: x[1], reverse=True):
                best_features.append({
                    '特征名': feat,
                    '重要性': imp,
                    '来源方法': ', '.join(self.feature_sources.get(feat, ['未知']))
                })

            df_features = pd.DataFrame(best_features)
            df_features.to_excel(writer, sheet_name='最优模型特征', index=False)

            # Sheet 3: 所有特征信息
            all_features_info = []
            for feat in self.all_features:
                all_features_info.append({
                    '特征名': feat,
                    '来源方法': ', '.join(self.feature_sources.get(feat, ['未知'])),
                    '选择次数': len(self.feature_sources.get(feat, []))
                })

            df_all_features = pd.DataFrame(all_features_info)
            df_all_features = df_all_features.sort_values('选择次数', ascending=False)
            df_all_features.to_excel(writer, sheet_name='所有特征信息', index=False)

            # Sheet 4: 性能汇总
            summary_data = {
                '指标': ['样本总数', '训练集样本', '测试集样本', '初始特征数',
                         '最优特征数', '测试集R²', '测试集RMSE', '测试集MAE', '测试集MAPE'],
                '数值': [
                    len(self.X),
                    len(self.X_train),
                    len(self.X_test),
                    len(self.all_features),
                    self.best_result['n_features'],
                    self.best_result['test_metrics']['R2'],
                    self.best_result['test_metrics']['RMSE'],
                    self.best_result['test_metrics']['MAE'],
                    self.best_result['test_metrics']['MAPE']
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='性能汇总', index=False)

        print(f"  [OK] 详细结果已保存到Excel: {save_path}")

    def run_complete_pipeline(self):
        """
        运行完整的训练流程
        """
        print("\n")
        print("=" * 80)
        print(" " * 20 + "LightGBM RFE 训练流程启动")
        print("=" * 80)

        # 执行所有步骤
        self.load_feature_selection_results()
        self.load_data()
        self.split_data()
        self.recursive_feature_elimination()
        self.find_best_model()

        # 生成所有图表
        self.plot_rfe_curves()
        self.plot_feature_importance()
        self.plot_prediction_scatter()
        self.plot_residual_analysis()

        # 保存结果
        self.save_results_summary()
        self.save_results_to_excel()

        print("\n")
        print("=" * 80)
        print(" " * 25 + "训练流程完成!")
        print("=" * 80)
        print("\n已生成以下文件:")
        print("  1. RFE_performance_curves.png - RFE性能曲线图")
        print("  2. feature_importance_top20.png - 特征重要性图")
        print("  3. prediction_scatter.png - 预测散点图")
        print("  4. residual_analysis.png - 残差分析图")
        print("  5. training_results_summary.txt - 结果摘要文本")
        print("  6. training_results_detailed.xlsx - 详细结果Excel表")
        print("\n请查看这些文件了解详细的训练结果!")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    """
    主程序执行流程
    """
    # 创建训练器实例
    trainer = LightGBMRFETrainer(
        test_size=0.4,  # 测试集40%，训练集60%
        random_state=42  # 随机种子
    )

    # 运行完整流程
    trainer.run_complete_pipeline()

    print("\n" + "=" * 80)
    print("感谢使用! 如有任何问题，请查看代码中的详细注释。")
    print("=" * 80)

