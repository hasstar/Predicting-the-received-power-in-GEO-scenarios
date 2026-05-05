"""
===================================================================================
High-Orbit Satellite Signal Loss Prediction - CatBoost Model with RFE
===================================================================================

[Changes from original]
- Replaced CatBoost with CatBoostRegressor
- Implemented RFE using CatBoost feature importances
- Kept original pipeline (load features, merge data, split, RFE, find best, plot, save)
- Avoided using red/green/blue as distinguishing plot colors
- Saved same output files as original

[Author] AI Assistant (converted to CatBoost)
[Date] 2025-11-21
===================================================================================
"""
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
import warnings
from typing import List, Dict, Tuple
import json
from datetime import datetime
import sys
import io
from scipy import stats


warnings.filterwarnings('ignore')

# Set output encoding to UTF-8 (retain your original wrapper)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set font for matplotlib (use English in plots)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class CatBoostRFETrainer:
    """
    NOTE: Class name kept for compatibility with your scripts and saved file names.
    This version uses CatBoost internally.
    """

    def __init__(self, test_size=0.4, random_state=42, catboost_iterations=500, catboost_depth=6):
        """
        Initialize trainer

        Parameters:
            test_size: test set ratio, default 0.4 (60% training)
            random_state: random seed for reproducibility
            catboost_iterations: default iterations for CatBoost
            catboost_depth: default tree depth for CatBoost
        """
        self.test_size = test_size
        self.random_state = random_state
        self.all_features = []
        self.feature_sources = {}
        self.rfe_results = []
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_result = None
        self.catboost_iterations = catboost_iterations
        self.catboost_depth = catboost_depth

        print("=" * 80)
        print("CatBoost Recursive Feature Elimination Trainer Initialized")
        print(f"Training set ratio: {1 - test_size:.0%}, Test set ratio: {test_size:.0%}")
        print(f"Random seed: {random_state}")
        print("=" * 80)

    def load_feature_selection_results(self, feature_dir='特征'):
        """
        Load feature selection results from all methods and take union
        """
        print("\n[Step 1] Load Feature Selection Results")
        print("-" * 80)

        feature_files = {
            'Lasso': 'Lasso_top_35_features.csv',
            'RF': 'RF_selected_features_top35.csv',
            'MIC_minepy': 'MIC_minepy_top35_features.csv',
            'MIC_sklearn': 'MIC_sklearn_top35_features.csv',
            'Spearman': 'Spearman_top35_features.csv'
        }

        features_by_method = {}

        for method_name, filename in feature_files.items():
            filepath = f"{feature_dir}/{filename}"

            try:
                df = pd.read_csv(filepath)

                if 'feature' in df.columns:
                    features = df['feature'].tolist()
                elif 'Feature' in df.columns:
                    features = df['Feature'].tolist()
                else:
                    print(f"  [WARNING] No feature column in {filename}")
                    continue

                features = [f for f in features if isinstance(f, str) and f.strip()]
                features_by_method[method_name] = features
                print(f"  [OK] {method_name:15s}: {len(features):2d} features")

            except Exception as e:
                print(f"  [ERROR] Failed to read {filename}: {str(e)}")

        # Calculate union
        all_features_set = set()
        for features in features_by_method.values():
            all_features_set.update(features)

        self.all_features = sorted(list(all_features_set))

        # Track source methods for each feature
        for feature in self.all_features:
            sources = [method for method, features in features_by_method.items()
                       if feature in features]
            self.feature_sources[feature] = sources

        print(f"\n  Feature Union Statistics:")
        print(f"  - Total features: {len(self.all_features)}")
        print(f"  - Individual method counts: {[len(f) for f in features_by_method.values()]}")

        selection_counts = {}
        for feature, sources in self.feature_sources.items():
            count = len(sources)
            if count not in selection_counts:
                selection_counts[count] = []
            selection_counts[count].append(feature)

        print(f"\n  Feature Selection Frequency:")
        for count in sorted(selection_counts.keys(), reverse=True):
            print(f"  - Selected by {count} methods: {len(selection_counts[count])} features")

        return features_by_method

    def load_data(self, loss_file='数据/merged_result_loss.csv',
                  ground_file='数据/all_ground_atmosphere_data_merged_new_processed.csv',
                  layer_files=None):
        """
        Load and merge all data
        """
        print("\n[Step 2] Load Raw Data")
        print("-" * 80)

        if layer_files is None:
            layer_files = [
                '数据/atmosphere_region_39_9_116_3_0-8km_processed.csv',
                '数据/atmosphere_region_39_8_116_3_8-17km_processed.csv',
                '数据/atmosphere_region_39_7_116_4_17-28km_processed.csv',
                '数据/atmosphere_region_39_6_116_4_28-37km_processed.csv',
                '数据/atmosphere_region_39_5_116_4_37-50km_processed.csv'
            ]

        print("\n  Loading target variable (loss)...")
        df_loss = pd.read_csv(loss_file)
        if 'loss' not in df_loss.columns:
            raise ValueError(f"'loss' column not found in {loss_file}")
        y = df_loss['loss'].values
        print(f"  [OK] Samples: {len(y)}")
        print(f"  [OK] Loss range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"  [OK] Loss mean: {y.mean():.3f} +/- {y.std():.3f}")

        print("\n  Loading ground atmosphere...")
        df_ground = pd.read_csv(ground_file)
        print(f"  [OK] Ground features: {len(df_ground.columns)}")

        print("\n  Loading upper atmosphere...")
        layer_dfs = []
        layer_names = ['0-8km', '8-17km', '17-28km', '28-37km', '37-50km']

        for i, layer_file in enumerate(layer_files):
            df_layer = pd.read_csv(layer_file)
            df_layer.columns = [f"{col}_{layer_names[i]}" if col != 'pressure_level'
                                else col for col in df_layer.columns]
            layer_dfs.append(df_layer)
            print(f"  [OK] {layer_names[i]:10s} layer: {len(df_layer.columns)} features")

        print("\n  Merging all data...")
        df_all = pd.concat([df_ground] + layer_dfs, axis=1)

        available_features = [f for f in self.all_features if f in df_all.columns]
        missing_features = [f for f in self.all_features if f not in df_all.columns]

        if missing_features:
            print(f"\n  [WARNING] {len(missing_features)} features not found:")
            for feat in missing_features[:5]:
                print(f"    - {feat}")
            if len(missing_features) > 5:
                print(f"    ... and {len(missing_features) - 5} more")

        X = df_all[available_features].values
        self.all_features = available_features

        print(f"\n  Data merge completed:")
        print(f"  [OK] Samples: {X.shape[0]}")
        print(f"  [OK] Available features: {X.shape[1]}")
        print(f"  [OK] Missing values: {np.isnan(X).sum()}")

        if np.isnan(X).any():
            print(f"  Handling missing values...")
            X = np.nan_to_num(X, nan=0.0)

        self.X = X
        self.y = y

        # Also keep a DataFrame version for CatBoost-friendly RFE manipulations
        self.df_all = pd.DataFrame(df_all[available_features].copy())

        return X, y

    def split_data(self):
        """
        Split training and test sets
        """
        print("\n[Step 3] Split Training and Test Sets")
        print("-" * 80)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Also keep DataFrame versions aligned with self.all_features for RFE
        df = self.df_all.copy()
        n = len(df)
        # train_test_split preserves order? We must re-split the DataFrame in the same manner:
        # We'll use indices split via sklearn with shuffle controlled by random_state
        indices = np.arange(n)
        train_idx, test_idx = train_test_split(indices, test_size=self.test_size, random_state=self.random_state)
        self.df_train = df.iloc[train_idx].reset_index(drop=True)
        self.df_test = df.iloc[test_idx].reset_index(drop=True)

        print(f"  Training: {len(self.X_train)} samples ({len(self.X_train) / len(self.X):.1%})")
        print(f"  Test: {len(self.X_test)} samples ({len(self.X_test) / len(self.X):.1%})")
        print(f"\n  Training loss: {self.y_train.mean():.3f} +/- {self.y_train.std():.3f}")
        print(f"  Test loss: {self.y_test.mean():.3f} +/- {self.y_test.std():.3f}")

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics
        """
        # Ensure arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        mask = y_true != 0
        if mask.sum() == 0:
            mape = np.nan
        else:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        return {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

    def train_catboost(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, use_early_stopping=False):
        """
        Train CatBoost model with default parameters
        """
        params = dict(
            iterations=self.catboost_iterations,
            depth=self.catboost_depth,
            learning_rate=0.05,
            loss_function='RMSE',
            random_seed=self.random_state,
            verbose=False,
            task_type='CPU'  # change to 'GPU' if you have GPU and CatBoost configured
        )

        model = CatBoostRegressor(**params)

        if use_early_stopping and X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)

        return model

    def recursive_feature_elimination(self):
        """
        Execute Recursive Feature Elimination (RFE) with CatBoost feature importances

        This function:
        - works on dataframe self.df_train/self.df_test (keeps feature names)
        - in each iteration trains a CatBoost on the current feature set,
          computes feature importances, removes the least important feature,
          and records metrics.
        """
        print("\n[Step 4] Recursive Feature Elimination (RFE) using CatBoost")
        print("-" * 80)
        print("\n  RFE Algorithm:")
        print("  1. Start with all features")
        print("  2. Train model and evaluate")
        print("  3. Get feature importance ranking from CatBoost")
        print("  4. Remove least important feature")
        print("  5. Repeat until 1 feature remains")
        print("  6. Track all metrics and features at each step")
        print()

        # Work with DataFrame copies so we retain column names
        current_features = self.all_features.copy()
        current_X_train_df = self.df_train[current_features].copy()
        current_X_test_df = self.df_test[current_features].copy()

        total_iterations = len(current_features)

        # Create detailed log file
        log_file = open('CatBoost_RFE_detailed_log.txt', 'w', encoding='utf-8')
        log_file.write("=" * 100 + "\n")
        log_file.write("CatBoost RFE Detailed Log - Feature List for Each Iteration\n")
        log_file.write("=" * 100 + "\n\n")

        for iteration in range(total_iterations):
            n_features = len(current_features)

            print(f"  Iteration {iteration + 1}/{total_iterations}: {n_features} features")

            # Write to log file
            log_file.write(f"\n{'=' * 100}\n")
            log_file.write(f"ITERATION {iteration + 1}/{total_iterations}\n")
            log_file.write(f"{'=' * 100}\n")
            log_file.write(f"Number of features: {n_features}\n\n")
            log_file.write(f"Feature list:\n")
            for idx, feat in enumerate(current_features, 1):
                sources = ', '.join(self.feature_sources.get(feat, ['Unknown']))
                log_file.write(f"  {idx:2d}. {feat:40s} [Source: {sources}]\n")
            log_file.write("\n")

            # Train CatBoost on current features
            model = self.train_catboost(
                current_X_train_df.values, self.y_train,
                X_val=current_X_test_df.values, y_val=self.y_test,
                feature_names=current_features,
                use_early_stopping=False
            )

            # Predict
            y_train_pred = model.predict(current_X_train_df.values)
            y_test_pred = model.predict(current_X_test_df.values)

            # Calculate metrics
            train_metrics = self.calculate_metrics(self.y_train, y_train_pred)
            test_metrics = self.calculate_metrics(self.y_test, y_test_pred)

            # Get feature importance from CatBoost (FeatureImportance)
            importance = model.get_feature_importance(type='FeatureImportance')
            importance_dict = dict(zip(current_features, importance))

            # Write metrics to log
            log_file.write(f"Performance Metrics:\n")
            log_file.write(f"  Training Set:\n")
            log_file.write(f"    R2:   {train_metrics['R2']:.6f}\n")
            log_file.write(f"    RMSE: {train_metrics['RMSE']:.6f}\n")
            log_file.write(f"    MAE:  {train_metrics['MAE']:.6f}\n")
            log_file.write(f"    MAPE: {train_metrics['MAPE']:.2f}%\n")
            log_file.write(f"\n  Test Set:\n")
            log_file.write(f"    R2:   {test_metrics['R2']:.6f}\n")
            log_file.write(f"    RMSE: {test_metrics['RMSE']:.6f}\n")
            log_file.write(f"    MAE:  {test_metrics['MAE']:.6f}\n")
            log_file.write(f"    MAPE: {test_metrics['MAPE']:.2f}%\n")
            log_file.write(f"\n")

            # Write feature importance to log (sorted)
            log_file.write(f"Feature Importance (sorted):\n")
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            for idx, (feat, imp) in enumerate(sorted_importance, 1):
                log_file.write(f"  {idx:2d}. {feat:40s} : {imp:.6f}\n")
            log_file.write("\n")

            # Save result
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

            # Console output
            print(f"    Train R^2={train_metrics['R2']:.4f}, RMSE={train_metrics['RMSE']:.4f}")
            print(f"    Test  R^2={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.4f}")

            if n_features == 1:
                log_file.write(f"RFE Complete - Reached 1 feature\n")
                break

            # Remove least important feature
            least_important_feature = min(importance_dict, key=importance_dict.get)
            least_important_idx = current_features.index(least_important_feature)

            print(f"    Remove: {least_important_feature} (importance: {importance_dict[least_important_feature]:.6f})")
            log_file.write(f"Feature to remove: {least_important_feature} (importance: {importance_dict[least_important_feature]:.6f})\n")

            # Update current feature lists and DataFrames
            current_features.pop(least_important_idx)
            current_X_train_df = current_X_train_df[current_features]
            current_X_test_df = current_X_test_df[current_features]
            print()

        log_file.write("\n" + "=" * 100 + "\n")
        log_file.write(f"RFE COMPLETED - Total {len(self.rfe_results)} iterations\n")
        log_file.write("=" * 100 + "\n")
        log_file.close()

        print(f"  RFE completed! {len(self.rfe_results)} iterations")
        print(f"  Detailed feature log saved: CatBoost_RFE_detailed_log.txt")

    def find_best_model(self, target_iteration=None):
        """
        Find optimal model based on test R^2 OR select a specific iteration provided by user.
        """
        print("\n[Step 5] Select Final Model")
        print("-" * 80)

        # check valid range
        total_iterations = len(self.rfe_results)

        if target_iteration is not None:
            # === 修改逻辑开始：用户指定了具体的循环次数 ===
            if 1 <= target_iteration <= total_iterations:
                # 列表索引是从0开始的，所以要减1
                best_result = self.rfe_results[target_iteration - 1]
                print(f"\n  [USER OVERRIDE] Selected specific iteration: {target_iteration}")
            else:
                raise ValueError(f"Target iteration {target_iteration} is out of range (1-{total_iterations})")
            # === 修改逻辑结束 ===
        else:
            # === 原有逻辑：自动寻找 R2 最高的 ===
            sorted_results = sorted(self.rfe_results,
                                    key=lambda x: x['test_metrics']['R2'],
                                    reverse=True)
            best_result = sorted_results[0]
            print(f"\n  [AUTO SELECT] Selected best model based on highest Test R^2")

        print(f"  - Iteration: {best_result['iteration']}")
        print(f"  - Number of features: {best_result['n_features']}")
        print(f"  - Test R^2: {best_result['test_metrics']['R2']:.4f}")
        print(f"  - Test RMSE: {best_result['test_metrics']['RMSE']:.4f}")
        print(f"  - Test MAE: {best_result['test_metrics']['MAE']:.4f}")
        print(f"  - Test MAPE: {best_result['test_metrics']['MAPE']:.2f}%")

        print(f"\n  Features used in this iteration:")
        sorted_features = sorted(best_result['feature_importance'].items(),
                                 key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(sorted_features, 1):
            sources = ', '.join(self.feature_sources.get(feat, ['Unknown']))
            print(f"    {i:2d}. {feat:35s} (imp: {imp:6.4f}) [Source: {sources}]")

        self.best_result = best_result
        return best_result

    def plot_rfe_curves(self, save_path='CatBoost_RFE_performance_curves.png'):
        """
        Plot RFE performance curves
        """
        print("\n[Step 6] Generate RFE Performance Curves")
        print("-" * 80)

        n_features_list = [r['n_features'] for r in self.rfe_results]

        train_r2 = [r['train_metrics']['R2'] for r in self.rfe_results]
        test_r2 = [r['test_metrics']['R2'] for r in self.rfe_results]

        train_rmse = [r['train_metrics']['RMSE'] for r in self.rfe_results]
        test_rmse = [r['test_metrics']['RMSE'] for r in self.rfe_results]

        train_mae = [r['train_metrics']['MAE'] for r in self.rfe_results]
        test_mae = [r['test_metrics']['MAE'] for r in self.rfe_results]

        train_mape = [r['train_metrics']['MAPE'] for r in self.rfe_results]
        test_mape = [r['test_metrics']['MAPE'] for r in self.rfe_results]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # R^2
        ax1 = axes[0, 0]
        ax1.plot(n_features_list, train_r2, 'o-', label='Training Set', linewidth=2, markersize=4, color='tab:purple')
        ax1.plot(n_features_list, test_r2, 's-', label='Test Set', linewidth=2, markersize=4, color='tab:orange')
        ax1.set_xlabel('Number of Features', fontsize=12)
        ax1.set_ylabel('R^2 Score', fontsize=12)
        ax1.set_title('R^2 vs Number of Features\n(Higher is better)',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        best_idx = test_r2.index(max(test_r2))
        ax1.plot(n_features_list[best_idx], test_r2[best_idx], marker='*', color='k',
                 markersize=14, label=f'Best ({n_features_list[best_idx]} features)')
        ax1.legend(fontsize=11)

        # RMSE
        ax2 = axes[0, 1]
        ax2.plot(n_features_list, train_rmse, 'o-', label='Training Set', linewidth=2, markersize=4, color='tab:purple')
        ax2.plot(n_features_list, test_rmse, 's-', label='Test Set', linewidth=2, markersize=4, color='tab:orange')
        ax2.set_xlabel('Number of Features', fontsize=12)
        ax2.set_ylabel('RMSE', fontsize=12)
        ax2.set_title('RMSE vs Number of Features\n(Lower is better)',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # MAE
        ax3 = axes[1, 0]
        ax3.plot(n_features_list, train_mae, 'o-', label='Training Set', linewidth=2, markersize=4, color='tab:purple')
        ax3.plot(n_features_list, test_mae, 's-', label='Test Set', linewidth=2, markersize=4, color='tab:orange')
        ax3.set_xlabel('Number of Features', fontsize=12)
        ax3.set_ylabel('MAE', fontsize=12)
        ax3.set_title('MAE vs Number of Features\n(Lower is better)',
                      fontsize=13, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)

        # MAPE
        ax4 = axes[1, 1]
        ax4.plot(n_features_list, train_mape, 'o-', label='Training Set', linewidth=2, markersize=4, color='tab:purple')
        ax4.plot(n_features_list, test_mape, 's-', label='Test Set', linewidth=2, markersize=4, color='tab:orange')
        ax4.set_xlabel('Number of Features', fontsize=12)
        ax4.set_ylabel('MAPE (%)', fontsize=12)
        ax4.set_title('MAPE vs Number of Features\n(Lower is better)',
                      fontsize=13, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Curves saved: {save_path}")
        plt.close()

    def plot_feature_importance(self, save_path='CatBoost_feature_importance_top20.png'):
        """
        Plot feature importance
        """
        print("\n[Step 7] Generate Feature Importance Plot")
        print("-" * 80)

        importance_dict = self.best_result['feature_importance']
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        top_n = min(20, len(sorted_features))
        top_features = sorted_features[:top_n]

        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]

        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))

        # Use a perceptually-uniform colormap for bars (not raw RGB)
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(features)))
        bars = ax.barh(range(len(features)), importances, color=colors)

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance (CatBoost)\nBest model with {self.best_result["n_features"]} features',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        for i, (bar, imp) in enumerate(zip(bars, importances)):
            ax.text(imp, i, f' {imp:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Importance plot saved: {save_path}")
        plt.close()

    def plot_prediction_scatter(self, save_path='CatBoost_prediction_scatter.png'):
        """
        Plot prediction scatter
        """
        print("\n[Step 8] Generate Prediction Scatter Plot")
        print("-" * 80)

        best_model = self.best_result['model']
        feature_indices = [self.all_features.index(f) for f in self.best_result['features']]

        X_train_selected = self.X_train[:, feature_indices]
        X_test_selected = self.X_test[:, feature_indices]

        y_train_pred = best_model.predict(X_train_selected)
        y_test_pred = best_model.predict(X_test_selected)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Training
        ax1.scatter(self.y_train, y_train_pred, alpha=0.6, s=50,
                    c='tab:purple', edgecolors='k', linewidth=0.5)
        ax1.plot([self.y_train.min(), self.y_train.max()],
                 [self.y_train.min(), self.y_train.max()],
                 'k--', linewidth=2, label='Perfect Prediction (y=x)')

        train_r2 = self.best_result['train_metrics']['R2']
        train_rmse = self.best_result['train_metrics']['RMSE']

        ax1.set_xlabel('Actual Loss', fontsize=12)
        ax1.set_ylabel('Predicted Loss', fontsize=12)
        ax1.set_title(f'Training Set\nR^2={train_r2:.4f}, RMSE={train_rmse:.4f}',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Test
        ax2.scatter(self.y_test, y_test_pred, alpha=0.6, s=50,
                    c='tab:orange', edgecolors='k', linewidth=0.5)
        ax2.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 'k--', linewidth=2, label='Perfect Prediction (y=x)')

        test_r2 = self.best_result['test_metrics']['R2']
        test_rmse = self.best_result['test_metrics']['RMSE']

        ax2.set_xlabel('Actual Loss', fontsize=12)
        ax2.set_ylabel('Predicted Loss', fontsize=12)
        ax2.set_title(f'Test Set\nR^2={test_r2:.4f}, RMSE={test_rmse:.4f}',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Scatter plot saved: {save_path}")
        plt.close()

    def plot_residual_analysis(self, save_path='CatBoost_residual_analysis.png'):
        """
        Plot residual analysis
        """
        print("\n[Step 9] Generate Residual Analysis")
        print("-" * 80)

        best_model = self.best_result['model']
        feature_indices = [self.all_features.index(f) for f in self.best_result['features']]
        X_test_selected = self.X_test[:, feature_indices]
        y_test_pred = best_model.predict(X_test_selected)

        residuals = self.y_test - y_test_pred

        fig = plt.figure(figsize=(16, 12))

        # 1. Residuals vs Predicted
        ax1 = plt.subplot(2, 2, 1)
        ax1.scatter(y_test_pred, residuals, alpha=0.6, s=50,
                    c='tab:purple', edgecolors='k', linewidth=0.5)
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Value', fontsize=18)
        ax1.set_ylabel('Residual', fontsize=18)
        ax1.set_title('Residuals vs Predicted Values', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Histogram
        ax2 = plt.subplot(2, 2, 2)
        ax2.hist(residuals, bins=30, color='lightgray', edgecolor='k', alpha=0.9)
        ax2.axvline(x=0, color='k', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residual', fontsize=18)
        ax2.set_ylabel('Frequency', fontsize=18)
        ax2.set_title(f'Residual Distribution\nMean={residuals.mean():.4f}, Std={residuals.std():.4f}',
                      fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Q-Q Plot
        ax3 = plt.subplot(2, 2, 3)
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Sorted Residuals
        ax4 = plt.subplot(2, 2, 4)
        sorted_abs_residuals = np.sort(np.abs(residuals))
        ax4.plot(sorted_abs_residuals, 'o-', linewidth=1, markersize=4, color='tab:orange')
        ax4.set_xlabel('Sample Index (sorted)', fontsize=18)
        ax4.set_ylabel('Absolute Residual', fontsize=18)
        ax4.set_title('Sorted Absolute Residuals', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Residual plot saved: {save_path}")
        plt.close()

    def save_results_summary(self, save_path='CatBoost_training_results_summary.txt'):
        """
        Save results summary
        """
        print("\n[Step 10] Save Results Summary")
        print("-" * 80)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CatBoost Training Results Summary\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("[1. Dataset Information]\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total samples: {len(self.X)}\n")
            f.write(f"Training: {len(self.X_train)} ({len(self.X_train) / len(self.X):.1%})\n")
            f.write(f"Test: {len(self.X_test)} ({len(self.X_test) / len(self.X):.1%})\n")
            f.write(f"Initial features: {len(self.all_features)}\n\n")

            f.write("[2. Best Model]\n")
            f.write("-" * 80 + "\n")
            best = self.best_result
            f.write(f"Optimal features: {best['n_features']}\n\n")

            f.write("Training performance:\n")
            for metric, value in best['train_metrics'].items():
                if metric == 'MAPE':
                    f.write(f"  {metric:6s}: {value:8.2f}%\n")
                else:
                    f.write(f"  {metric:6s}: {value:8.4f}\n")

            f.write("\nTest performance:\n")
            for metric, value in best['test_metrics'].items():
                if metric == 'MAPE':
                    f.write(f"  {metric:6s}: {value:8.2f}%\n")
                else:
                    f.write(f"  {metric:6s}: {value:8.4f}\n")

            f.write("\nOptimal features:\n")
            sorted_features = sorted(best['feature_importance'].items(),
                                     key=lambda x: x[1], reverse=True)
            for i, (feat, imp) in enumerate(sorted_features, 1):
                sources = ', '.join(self.feature_sources.get(feat, ['Unknown']))
                f.write(f"  {i:2d}. {feat:35s} {imp:8.4f} [{sources}]\n")

            f.write("\n[3. Complete RFE Results]\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Iter':>4s} {'#Feat':>6s} {'TrainR2':>9s} {'TestR2':>9s} "
                    f"{'TrainRMSE':>10s} {'TestRMSE':>10s}\n")
            f.write("-" * 80 + "\n")

            for result in self.rfe_results:
                f.write(f"{result['iteration']:4d} {result['n_features']:6d} "
                        f"{result['train_metrics']['R2']:9.4f} "
                        f"{result['test_metrics']['R2']:9.4f} "
                        f"{result['train_metrics']['RMSE']:10.4f} "
                        f"{result['test_metrics']['RMSE']:10.4f}\n")

        print(f"  [OK] Summary saved: {save_path}")

    def save_results_to_excel(self, save_path='CatBoost_training_results_detailed.xlsx'):
        """
        Save to Excel
        """
        print("\n[Step 11] Save Detailed Results to Excel")
        print("-" * 80)

        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            # Sheet 1: RFE Process
            rfe_data = []
            for result in self.rfe_results:
                row = {
                    'Iteration': result['iteration'],
                    'Number of Features': result['n_features'],
                    'Train R2': result['train_metrics']['R2'],
                    'Test R2': result['test_metrics']['R2'],
                    'Train RMSE': result['train_metrics']['RMSE'],
                    'Test RMSE': result['test_metrics']['RMSE'],
                    'Train MAE': result['train_metrics']['MAE'],
                    'Test MAE': result['test_metrics']['MAE'],
                    'Train MAPE': result['train_metrics']['MAPE'],
                    'Test MAPE': result['test_metrics']['MAPE'],
                }
                rfe_data.append(row)

            df_rfe = pd.DataFrame(rfe_data)
            df_rfe.to_excel(writer, sheet_name='RFE Process', index=False)

            # Sheet 2: Best Model Features
            best_features = []
            for feat, imp in sorted(self.best_result['feature_importance'].items(),
                                    key=lambda x: x[1], reverse=True):
                best_features.append({
                    'Feature': feat,
                    'Importance': imp,
                    'Source Methods': ', '.join(self.feature_sources.get(feat, ['Unknown']))
                })

            df_features = pd.DataFrame(best_features)
            df_features.to_excel(writer, sheet_name='Best Model Features', index=False)

            # Sheet 3: Feature List for Each Iteration
            iter_features_data = []
            for result in self.rfe_results:
                for feat in result['features']:
                    iter_features_data.append({
                        'Iteration': result['iteration'],
                        'Feature': feat,
                        'Importance': result['feature_importance'][feat],
                        'Source Methods': ', '.join(self.feature_sources.get(feat, ['Unknown']))
                    })

            df_iter_features = pd.DataFrame(iter_features_data)
            df_iter_features.to_excel(writer, sheet_name='Features Per Iteration', index=False)

        print(f"  [OK] Excel saved: {save_path}")

    def run_complete_pipeline(self, target_iteration=None):
        """
        Run complete training pipeline
        :param target_iteration: (Optional) The specific iteration number to select as the final model.
                                 If None, automatically selects the best R2 score.
        """
        print("\n")
        print("=" * 80)
        print(" " * 20 + "CatBoost RFE Training Pipeline Started")
        print("=" * 80)

        self.load_feature_selection_results()
        self.load_data()
        self.split_data()
        self.recursive_feature_elimination()

        # Pass the target_iteration to find_best_model
        self.find_best_model(target_iteration=target_iteration)

        self.plot_rfe_curves()
        self.plot_feature_importance()
        self.plot_prediction_scatter()
        self.plot_residual_analysis()

        self.save_results_summary()
        self.save_results_to_excel()

        # === 新增：保存模型用于部署 ===
        self.save_model_artifact()

        # === 新增：单独提取并分析测试集 ===
        self.analyze_test_set()

        # === 新增：绘制 SHAP 图 ===
        # 这里设置 top_n=15，表示只看最重要的15个特征
        self.plot_shap_analysis(top_n=15)
        # =========================
        print("\n")
        print("=" * 80)
        print(" " * 25 + "Training Completed!")
        print("=" * 80)
        # ... (后续打印语句保持不变)

    def plot_shap_analysis(self, top_n=20, save_name='CatBoost_SHAP_summary.pdf'):
        """
        生成 SHAP 摘要图 (Beeswarm plot)，展示特征对模型输出的影响。
        :param top_n: 只展示影响力最大的前 n 个特征
        :param save_name: 图片保存文件名
        """
        print("\n[Step Extra] Generating SHAP Explainer Plot")
        print("-" * 80)

        # 1. 检查是否有 shap 库
        try:
            import shap
        except ImportError:
            print("  [Error] 'shap' library not installed. Please run: pip install shap")
            return

        if self.best_result is None:
            print("  [Error] No model selected. Please run find_best_model first.")
            return

        # 2. 获取模型和对应的测试集特征
        model = self.best_result['model']
        feature_names = self.best_result['features']

        # 提取对应特征的测试集数据 (DataFrame格式，保留列名)
        X_test_shap = self.df_test[feature_names]

        print(f"  Calculating SHAP values for {X_test_shap.shape[0]} samples...")

        # 3. 计算 SHAP 值
        # TreeExplainer 专门针对树模型 (CatBoost/XGBoost/LightGBM) 优化，速度快
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_shap)

        # 4. 绘图
        plt.figure(figsize=(8, 5))

        # summary_plot 是最经典的 SHAP 图
        # max_display 控制只显示前 n 个特征
        # show=False 允许我们后续手动保存
        shap.summary_plot(shap_values, X_test_shap, max_display=top_n, show=False)

        ax = plt.gca()
        #ax.tick_params(axis='both', which='major', labelsize=10)  # 刻度字号
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=16)  # x轴标题
        ax.set_ylabel('Features', fontsize=16)  # y轴标题

        # 5. 保存并关闭
        # 由于 shap 内部会自动调整 margin，我们使用 bbox_inches='tight' 确保不切边
        plt.tight_layout()
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] SHAP summary plot saved to: {save_name}")
    def analyze_test_set(self, save_csv='test_set_predictions_only.csv', save_plot='test_set_analysis.pdf'):
        """
        专门提取 40% 测试集的数据进行预测、导出和绘图
        """
        print("\n[Step Extra] Analyzing Test Set (40%) Only")
        print("-" * 80)


        if self.best_result is None:
            print("  [Error] No model selected yet. Please run find_best_model first.")
            return

        # 1. 获取选定迭代的模型和特征列表
        model = self.best_result['model']
        selected_features = self.best_result['features']

        print(f"  Selected Iteration: {self.best_result['iteration']}")
        print(f"  Features count: {len(selected_features)}")

        # 2. 准备测试数据
        # self.df_test 是在 split_data 阶段生成的，包含了所有原始特征
        # 我们只提取当前模型需要的特征
        X_test_selected = self.df_test[selected_features].copy()
        y_test_actual = self.y_test  # 这是对应的真实标签

        # 3. 进行预测
        print("  Predicting on test set...")
        y_test_pred = model.predict(X_test_selected)

        # 4. 导出数据到 CSV
        # 创建一个结果 DataFrame
        df_export = X_test_selected.copy()
        df_export['Actual_Loss'] = y_test_actual
        df_export['Predicted_Loss'] = y_test_pred
        df_export['Error'] = df_export['Actual_Loss'] - df_export['Predicted_Loss']

        df_export.to_csv(save_csv, index=False)
        print(f"  [OK] Test set predictions saved to: {save_csv}")

        # 5. 计算指标
        metrics = self.calculate_metrics(y_test_actual, y_test_pred)
        title_str = f"Test Set Analysis (n={len(y_test_actual)})\nR2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}"

        # 6. 单独绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))# 14 12

        # --- 子图1：折线走势对比 ---
        # 如果数据点太多，只画前 150 个以便观察细节，或者画全部
        display_limit = 200
        if len(y_test_actual) > display_limit:
            plot_indices = range(display_limit)
            y_act_plot = y_test_actual[:display_limit]
            y_pred_plot = y_test_pred[:display_limit]
            subtitle = f"(Displaying first {display_limit} samples)"
        else:
            plot_indices = range(len(y_test_actual))
            y_act_plot = y_test_actual
            y_pred_plot = y_test_pred
            subtitle = "(All samples)"

        ax1.plot(plot_indices, y_act_plot, 'k-', alpha=0.7, label='Actual Loss(dB)', linewidth=1.5)
        ax1.plot(plot_indices, y_pred_plot, 'r--', alpha=0.8, label='Predicted Loss(dB)', linewidth=1.5)
        #ax1.set_title(f"Test Set Prediction Trend {subtitle}", fontsize=13, fontweight='bold')
        ax1.set_xlabel("Sample Index",fontsize=12)
        ax1.set_ylabel("Signal Loss(dB)",fontsize=12)
        ax1.legend(fontsize=8,loc="upper right")
        ax1.grid(True, alpha=0.3)

        # --- 子图2：真实值 vs 预测值 散点图 ---
        ax2.scatter(y_test_actual, y_test_pred, c='royalblue', alpha=0.6, edgecolors='w', s=40)

        # 画对角线
        min_val = min(y_test_actual.min(), y_test_pred.min())
        max_val = max(y_test_actual.max(), y_test_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Fit')

        #ax2.set_title(f"Test Set Scatter Plot\n{title_str}", fontsize=13, fontweight='bold')
        ax2.set_xlabel("Actual Loss(dB)",fontsize=12)
        ax2.set_ylabel("Predicted Loss(dB)",fontsize=12)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_plot, dpi=300)
        print(f"  [OK] Test set plot saved to: {save_plot}")
        plt.close()
    def save_model_artifact(self, model_name='final_model.cbm', feature_name='final_features.json'):
        """
        保存用于部署的模型文件和特征列表
        """
        print("\n[Step 12] Save Model Artifacts for Deployment")
        print("-" * 80)

        # 1. 保存模型实体 (CatBoost格式)
        model = self.best_result['model']
        model.save_model(model_name)
        model.save_model('best_model_catboost.cpp', format='cpp')
        model.save_model("best_model_catboost.onnx", format="onnx")
        model.save_model("best_model_catboost.json", format="json")
        print(f"  [OK] Model saved to: {model_name}")
        print(f"  [OK] Model saved to: best_model_catboost.cpp")
        print(f"  [OK] Model saved to: best_model_catboost.onnx")
        # 2. 保存对应的特征列表 (非常重要，预测时顺序必须一致)
        features = self.best_result['features']
        with open(feature_name, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=4)
        print(f"  [OK] Feature list saved to: {feature_name}")
        print(f"  [INFO] Ready for deployment! Use these two files in your prediction script.")
if __name__ == "__main__":
    trainer = CatBoostRFETrainer(
        test_size=0.4,
        random_state=42,
        catboost_iterations=500,
        catboost_depth=6
    )

    # 修改这里：传入 target_iteration = 50
    # 如果想恢复自动选择，只需去掉这个参数或设为 None
    trainer.run_complete_pipeline(target_iteration=67)



    print("\n" + "=" * 80)
    print("Thank you! Check CatBoost_RFE_detailed_log.txt for complete feature lists.")
    print("=" * 80)
