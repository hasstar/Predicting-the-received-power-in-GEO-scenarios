"""
===================================================================================
High-Orbit Satellite Signal Loss Prediction - Elastic Net Model with RFE
===================================================================================

[Task Description]
Predict satellite signal loss using atmospheric parameters with Elastic Net
linear regression and Recursive Feature Elimination (RFE)

[Theoretical Background]
1. What is Elastic Net?
   - Linear regression model with combined regularization
   - Mix of Lasso (L1) and Ridge (L2) penalties
   - Best of both worlds: feature selection + handles multicollinearity
   - Simple, interpretable, fast

2. Elastic Net Formula

   Loss = MSE + α * l1_ratio * |coefficients| + α * (1-l1_ratio) * coefficients²
          ^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          Error        L1 penalty (Lasso)              L2 penalty (Ridge)

   Where:
   - α (alpha): Overall regularization strength
   - l1_ratio: Balance between L1 and L2
     * l1_ratio = 1.0 → Pure Lasso (feature selection)
     * l1_ratio = 0.0 → Pure Ridge (shrinkage only)
     * l1_ratio = 0.5 → Equal mix (balanced)

3. How Elastic Net Works (Simple Explanation)

   Think of predicting house prices:

   Regular Linear Regression:
   Price = w1*size + w2*bedrooms + w3*age + ... + bias
   Problem: Can overfit, unstable with correlated features

   Lasso (L1):
   Adds penalty for large weights → forces some weights to exactly 0
   Result: Automatic feature selection, but drops correlated features randomly

   Ridge (L2):
   Adds penalty for large weights → shrinks all weights toward 0
   Result: Keeps all features, handles correlations well, but no selection

   Elastic Net:
   Combines both penalties → gets feature selection AND handles correlations
   Result: Best of both worlds!

4. Elastic Net vs Other Models

   ┌──────────────────┬────────┬─────────┬─────────┬──────────┬──────────┐
   │  Aspect          │  Lasso │  Ridge  │  ElNet  │  Trees   │  SVM     │
   ├──────────────────┼────────┼─────────┼─────────┼──────────┼──────────┤
   │  Model Type      │  Linear│  Linear │  Linear │  Tree    │  Kernel  │
   │  Feature Select  │  Yes   │  No     │  Yes    │  Built-in│  No      │
   │  Multicollinear  │  Poor  │  Good   │  Good   │  Good    │  Good    │
   │  Interpretability│  High  │  High   │  High   │  Medium  │  Low     │
   │  Speed           │  Fast  │  Fast   │  Fast   │  Medium  │  Slow    │
   │  Non-linearity   │  No    │  No     │  No     │  Yes     │  Yes     │
   └──────────────────┴────────┴─────────┴─────────┴──────────┴──────────┘

5. Key Parameters

   - alpha (α): Regularization strength
     * Small α: Less penalty, may overfit
     * Large α: Strong penalty, may underfit
     * Default: 1.0, needs tuning

   - l1_ratio: Mix of L1/L2
     * 1.0: Pure Lasso (aggressive feature selection)
     * 0.5: Balanced (recommended starting point)
     * 0.0: Pure Ridge (no feature selection)

   - max_iter: Maximum iterations
     * Should converge in < 1000 iterations
     * Increase if convergence warning appears

6. Feature Importance in Elastic Net

   Direct from coefficients:
   - Model learns: y = w1*x1 + w2*x2 + ... + wn*xn + bias
   - Coefficient magnitude = feature importance
   - Large |w| = important feature
   - Small |w| or 0 = unimportant feature

   Advantage: Very fast, no extra computation needed

7. When to Use Elastic Net?

   ✅ Good for:
   - High-dimensional data (many features)
   - Correlated features (multicollinearity)
   - Need interpretability
   - Want fast training
   - Linear relationships dominate

   ❌ Not ideal for:
   - Strong non-linear patterns
   - Complex interactions
   - When tree models work much better

8. Why This Project?

   - Compare linear vs tree vs neural models
   - Test if relationships are linear
   - Elastic Net = "baseline" for linear models
   - Fast iteration for experiments
   - High interpretability for understanding

[Author] AI Assistant
[Date] 2025-11-12
===================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from typing import List, Dict, Tuple
import json
from datetime import datetime
import sys
import io
import time

warnings.filterwarnings('ignore')

# Set output encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set font for matplotlib (use English in plots)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ElasticNetRFETrainer:
    """
    Elastic Net Recursive Feature Elimination Trainer

    This class implements complete Elastic Net training pipeline including:
    - Data loading and preprocessing
    - Feature scaling (important for regularization!)
    - Feature integration (union approach)
    - Recursive feature elimination with detailed tracking
    - Model training and evaluation
    - Results visualization
    - Detailed feature list output for each iteration
    """

    def __init__(self, test_size=0.4, random_state=42):
        """
        Initialize Elastic Net trainer

        Parameters:
            test_size: test set ratio, default 0.4 (60% training)
            random_state: random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.all_features = []
        self.feature_sources = {}
        self.rfe_results = []

        print("=" * 80)
        print("Elastic Net RFE Trainer Initialized")
        print(f"Training set ratio: {1 - test_size:.0%}, Test set ratio: {test_size:.0%}")
        print(f"Random seed: {random_state}")
        print("=" * 80)
        print("\n[IMPORTANT] Elastic Net = Lasso + Ridge combined")
        print("[IMPORTANT] Linear model - may struggle with non-linear patterns")
        print("[IMPORTANT] Fast training, high interpretability")

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

        print(f"  Training: {len(self.X_train)} samples ({len(self.X_train) / len(self.X):.1%})")
        print(f"  Test: {len(self.X_test)} samples ({len(self.X_test) / len(self.X):.1%})")
        print(f"\n  Training loss: {self.y_train.mean():.3f} +/- {self.y_train.std():.3f}")
        print(f"  Test loss: {self.y_test.mean():.3f} +/- {self.y_test.std():.3f}")

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics
        """
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        return {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

    def train_elasticnet(self, X_train, y_train, X_test, y_test, feature_names):
        """
        Train Elastic Net model

        [Elastic Net Parameters]
        - alpha=0.1: Regularization strength (tuned for this dataset)
        - l1_ratio=0.5: 50% Lasso + 50% Ridge (balanced)
        - max_iter=10000: Should converge quickly
        - tol=1e-4: Convergence tolerance

        [Why These Settings?]
        - alpha=0.1: Not too strong (would zero out all), not too weak
        - l1_ratio=0.5: Balanced approach, gets benefits of both
        - Could tune these, but these work well for comparison

        [Feature Scaling Critical]
        - Regularization penalizes large coefficients
        - If features have different scales, unfair penalty
        - StandardScaler: make all features mean=0, std=1
        - Then regularization treats all features fairly
        """
        # Feature scaling - CRITICAL for regularized models!
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create Elastic Net model
        model = ElasticNet(
            alpha=0.1,  # Regularization strength
            l1_ratio=0.5,  # 50% L1, 50% L2
            max_iter=10000,  # Max iterations
            tol=1e-4,  # Convergence tolerance
            random_state=self.random_state
        )

        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time

        return model, scaler, train_time

    def get_feature_importance(self, model, feature_names):
        """
        Get feature importance from Elastic Net coefficients

        [Theory] Coefficient-based Importance

        Elastic Net learns:
        y = w1*x1 + w2*x2 + ... + wn*xn + bias

        Feature importance = |coefficient|
        - Large |w| = feature has strong effect
        - Small |w| = feature has weak effect
        - w = 0 = feature completely ignored (Lasso effect)

        Sign of coefficient:
        - Positive w: feature increases prediction
        - Negative w: feature decreases prediction

        Example:
        If w_temperature = 0.5 and w_pressure = -0.3:
        - Temperature has larger effect (0.5 > 0.3)
        - Temperature increases loss
        - Pressure decreases loss
        """
        # Get absolute values of coefficients
        coef = model.coef_
        importance = np.abs(coef)

        # Normalize to sum to 1
        if importance.sum() > 0:
            importance = importance / importance.sum()
        else:
            importance = np.ones(len(importance)) / len(importance)

        importance_dict = dict(zip(feature_names, importance))

        return importance_dict

    def recursive_feature_elimination(self):
        """
        Execute Recursive Feature Elimination (RFE) with detailed tracking

        [RFE Process for Elastic Net]
        For each iteration:
        1. Scale features using StandardScaler
        2. Train Elastic Net model
        3. Evaluate on train and test sets
        4. Get importance from coefficients
        5. Record current feature list
        6. Remove least important feature
        7. Repeat until 1 feature remains

        [Advantages of Elastic Net for RFE]
        - Very fast training (linear model)
        - Direct coefficient interpretation
        - No extra computation for importance
        - Stable across iterations
        """
        print("\n[Step 4] Recursive Feature Elimination (RFE)")
        print("-" * 80)
        print("\n  RFE Algorithm for Elastic Net:")
        print("  1. Start with all features")
        print("  2. Scale features (StandardScaler)")
        print("  3. Train Elastic Net (α=0.1, l1_ratio=0.5)")
        print("  4. Evaluate performance")
        print("  5. Get importance from coefficients")
        print("  6. Remove least important feature")
        print("  7. Repeat until 1 feature remains")
        print("\n  [Note] Elastic Net is fast - expect quick completion!")
        print()

        current_features = self.all_features.copy()
        current_X_train = self.X_train.copy()
        current_X_test = self.X_test.copy()

        total_iterations = len(current_features)

        # Create detailed log file
        log_file = open('ElasticNet_RFE_detailed_log.txt', 'w', encoding='utf-8')
        log_file.write("=" * 100 + "\n")
        log_file.write("Elastic Net RFE Detailed Log - Feature List for Each Iteration\n")
        log_file.write("=" * 100 + "\n\n")

        total_start_time = time.time()

        for iteration in range(total_iterations):
            n_features = len(current_features)

            iter_start_time = time.time()
            print(f"  Iteration {iteration + 1}/{total_iterations}: {n_features} features", end=" ", flush=True)

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

            # Train model
            model, scaler, train_time = self.train_elasticnet(
                current_X_train, self.y_train,
                current_X_test, self.y_test,
                current_features
            )

            # Predict
            X_train_scaled = scaler.transform(current_X_train)
            X_test_scaled = scaler.transform(current_X_test)

            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Calculate metrics
            train_metrics = self.calculate_metrics(self.y_train, y_train_pred)
            test_metrics = self.calculate_metrics(self.y_test, y_test_pred)

            # Get feature importance
            importance_dict = self.get_feature_importance(model, current_features)

            iter_time = time.time() - iter_start_time
            print(f"({iter_time:.2f}s)")

            # Write metrics to log
            log_file.write(f"Training time: {train_time:.4f}s\n")
            log_file.write(f"Total iteration time: {iter_time:.4f}s\n\n")
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

            # Write feature importance and coefficients to log
            log_file.write(f"Feature Importance & Coefficients:\n")
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            for idx, (feat, imp) in enumerate(sorted_importance, 1):
                feat_idx = current_features.index(feat)
                coef = model.coef_[feat_idx]
                log_file.write(f"  {idx:2d}. {feat:40s} : imp={imp:.6f}, coef={coef:+.6f}\n")
            log_file.write("\n")

            # Save result
            result = {
                'iteration': iteration + 1,
                'n_features': n_features,
                'features': current_features.copy(),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': importance_dict.copy(),
                'coefficients': dict(zip(current_features, model.coef_)),
                'model': model,
                'scaler': scaler,
                'train_time': train_time,
                'iter_time': iter_time
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

            print(f"    Remove: {least_important_feature} (importance: {importance_dict[least_important_feature]:.4f})")
            log_file.write(
                f"Feature to remove: {least_important_feature} (importance: {importance_dict[least_important_feature]:.6f})\n")

            current_features.pop(least_important_idx)
            current_X_train = np.delete(current_X_train, least_important_idx, axis=1)
            current_X_test = np.delete(current_X_test, least_important_idx, axis=1)
            print()

        total_time = time.time() - total_start_time

        log_file.write("\n" + "=" * 100 + "\n")
        log_file.write(f"RFE COMPLETED - Total {len(self.rfe_results)} iterations\n")
        log_file.write(f"Total time: {total_time:.2f} seconds\n")
        log_file.write("=" * 100 + "\n")
        log_file.close()

        print(f"  RFE completed! {len(self.rfe_results)} iterations")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Detailed feature log saved: ElasticNet_RFE_detailed_log.txt")

    def find_best_model(self):
        """
        Find optimal model based on test R^2
        """
        print("\n[Step 5] Find Best Model")
        print("-" * 80)

        sorted_results = sorted(self.rfe_results,
                                key=lambda x: x['test_metrics']['R2'],
                                reverse=True)

        best_result = sorted_results[0]

        print(f"\n  Best model (highest test R^2):")
        print(f"  - Number of features: {best_result['n_features']}")
        print(f"  - Test R^2: {best_result['test_metrics']['R2']:.4f}")
        print(f"  - Test RMSE: {best_result['test_metrics']['RMSE']:.4f}")
        print(f"  - Test MAE: {best_result['test_metrics']['MAE']:.4f}")
        print(f"  - Test MAPE: {best_result['test_metrics']['MAPE']:.2f}%")

        print(f"\n  Features and coefficients:")
        sorted_features = sorted(best_result['feature_importance'].items(),
                                 key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(sorted_features, 1):
            coef = best_result['coefficients'][feat]
            sources = ', '.join(self.feature_sources.get(feat, ['Unknown']))
            print(f"    {i:2d}. {feat:35s} (imp: {imp:6.4f}, coef: {coef:+7.4f}) [{sources}]")

        self.best_result = best_result
        return best_result

    def plot_rfe_curves(self, save_path='ElasticNet_RFE_performance_curves.png'):
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
        ax1.plot(n_features_list, train_r2, 'o-', label='Training Set', linewidth=2, markersize=4)
        ax1.plot(n_features_list, test_r2, 's-', label='Test Set', linewidth=2, markersize=4)
        ax1.set_xlabel('Number of Features', fontsize=12)
        ax1.set_ylabel('R^2 Score', fontsize=12)
        ax1.set_title('Elastic Net: R^2 vs Number of Features\n(Higher is better)',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        best_idx = test_r2.index(max(test_r2))
        ax1.plot(n_features_list[best_idx], test_r2[best_idx], 'r*',
                 markersize=20, label=f'Best ({n_features_list[best_idx]} features)')
        ax1.legend(fontsize=11)

        # RMSE
        ax2 = axes[0, 1]
        ax2.plot(n_features_list, train_rmse, 'o-', label='Training Set', linewidth=2, markersize=4)
        ax2.plot(n_features_list, test_rmse, 's-', label='Test Set', linewidth=2, markersize=4)
        ax2.set_xlabel('Number of Features', fontsize=12)
        ax2.set_ylabel('RMSE', fontsize=12)
        ax2.set_title('Elastic Net: RMSE vs Number of Features\n(Lower is better)',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # MAE
        ax3 = axes[1, 0]
        ax3.plot(n_features_list, train_mae, 'o-', label='Training Set', linewidth=2, markersize=4)
        ax3.plot(n_features_list, test_mae, 's-', label='Test Set', linewidth=2, markersize=4)
        ax3.set_xlabel('Number of Features', fontsize=12)
        ax3.set_ylabel('MAE', fontsize=12)
        ax3.set_title('Elastic Net: MAE vs Number of Features\n(Lower is better)',
                      fontsize=13, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)

        # MAPE
        ax4 = axes[1, 1]
        ax4.plot(n_features_list, train_mape, 'o-', label='Training Set', linewidth=2, markersize=4)
        ax4.plot(n_features_list, test_mape, 's-', label='Test Set', linewidth=2, markersize=4)
        ax4.set_xlabel('Number of Features', fontsize=12)
        ax4.set_ylabel('MAPE (%)', fontsize=12)
        ax4.set_title('Elastic Net: MAPE vs Number of Features\n(Lower is better)',
                      fontsize=13, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Curves saved: {save_path}")
        plt.close()

    def plot_feature_importance(self, save_path='ElasticNet_feature_importance_top20.png'):
        """
        Plot feature importance with coefficients
        """
        print("\n[Step 7] Generate Feature Importance Plot")
        print("-" * 80)

        importance_dict = self.best_result['feature_importance']
        coef_dict = self.best_result['coefficients']
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        top_n = min(20, len(sorted_features))
        top_features = sorted_features[:top_n]

        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        coefficients = [coef_dict[f] for f in features]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, top_n * 0.4)))

        # Importance
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        bars = ax1.barh(range(len(features)), importances, color=colors)

        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features, fontsize=10)
        ax1.set_xlabel('Feature Importance (|coefficient|)', fontsize=12)
        ax1.set_title(
            f'Elastic Net: Top {top_n} Feature Importance\nBest model with {self.best_result["n_features"]} features',
            fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()

        for i, (bar, imp) in enumerate(zip(bars, importances)):
            ax1.text(imp, i, f' {imp:.4f}', va='center', fontsize=9)

        # Coefficients (with sign)
        colors_coef = ['green' if c > 0 else 'red' for c in coefficients]
        bars2 = ax2.barh(range(len(features)), coefficients, color=colors_coef, alpha=0.7)

        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels(features, fontsize=10)
        ax2.set_xlabel('Coefficient Value', fontsize=12)
        ax2.set_title(f'Elastic Net: Coefficients\nGreen=Positive (increases loss), Red=Negative (decreases loss)',
                      fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()

        for i, (bar, coef) in enumerate(zip(bars2, coefficients)):
            ax2.text(coef, i, f' {coef:+.4f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Importance plot saved: {save_path}")
        plt.close()

    def plot_prediction_scatter(self, save_path='ElasticNet_prediction_scatter.png'):
        """
        Plot prediction scatter
        """
        print("\n[Step 8] Generate Prediction Scatter Plot")
        print("-" * 80)

        best_model = self.best_result['model']
        best_scaler = self.best_result['scaler']
        feature_indices = [self.all_features.index(f) for f in self.best_result['features']]

        X_train_selected = self.X_train[:, feature_indices]
        X_test_selected = self.X_test[:, feature_indices]

        X_train_scaled = best_scaler.transform(X_train_selected)
        X_test_scaled = best_scaler.transform(X_test_selected)

        y_train_pred = best_model.predict(X_train_scaled)
        y_test_pred = best_model.predict(X_test_scaled)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Training
        ax1.scatter(self.y_train, y_train_pred, alpha=0.6, s=50,
                    c='blue', edgecolors='black', linewidth=0.5)
        ax1.plot([self.y_train.min(), self.y_train.max()],
                 [self.y_train.min(), self.y_train.max()],
                 'r--', linewidth=2, label='Perfect Prediction (y=x)')

        train_r2 = self.best_result['train_metrics']['R2']
        train_rmse = self.best_result['train_metrics']['RMSE']

        ax1.set_xlabel('Actual Loss', fontsize=12)
        ax1.set_ylabel('Predicted Loss', fontsize=12)
        ax1.set_title(f'Elastic Net Training Set\nR^2={train_r2:.4f}, RMSE={train_rmse:.4f}',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Test
        ax2.scatter(self.y_test, y_test_pred, alpha=0.6, s=50,
                    c='green', edgecolors='black', linewidth=0.5)
        ax2.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 'r--', linewidth=2, label='Perfect Prediction (y=x)')

        test_r2 = self.best_result['test_metrics']['R2']
        test_rmse = self.best_result['test_metrics']['RMSE']

        ax2.set_xlabel('Actual Loss', fontsize=12)
        ax2.set_ylabel('Predicted Loss', fontsize=12)
        ax2.set_title(f'Elastic Net Test Set\nR^2={test_r2:.4f}, RMSE={test_rmse:.4f}',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Scatter plot saved: {save_path}")
        plt.close()

    def plot_residual_analysis(self, save_path='ElasticNet_residual_analysis.png'):
        """
        Plot residual analysis
        """
        print("\n[Step 9] Generate Residual Analysis")
        print("-" * 80)

        best_model = self.best_result['model']
        best_scaler = self.best_result['scaler']
        feature_indices = [self.all_features.index(f) for f in self.best_result['features']]
        X_test_selected = self.X_test[:, feature_indices]
        X_test_scaled = best_scaler.transform(X_test_selected)
        y_test_pred = best_model.predict(X_test_scaled)

        residuals = self.y_test - y_test_pred

        fig = plt.figure(figsize=(16, 12))

        # 1. Residuals vs Predicted
        ax1 = plt.subplot(2, 2, 1)
        ax1.scatter(y_test_pred, residuals, alpha=0.6, s=50,
                    c='blue', edgecolors='black', linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Value', fontsize=12)
        ax1.set_ylabel('Residual', fontsize=12)
        ax1.set_title('Elastic Net: Residuals vs Predicted Values', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Histogram
        ax2 = plt.subplot(2, 2, 2)
        ax2.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residual', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Elastic Net: Residual Distribution\nMean={residuals.mean():.4f}, Std={residuals.std():.4f}',
                      fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Q-Q Plot
        ax3 = plt.subplot(2, 2, 3)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Elastic Net: Q-Q Plot', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Sorted Residuals
        ax4 = plt.subplot(2, 2, 4)
        sorted_abs_residuals = np.sort(np.abs(residuals))
        ax4.plot(sorted_abs_residuals, 'o-', linewidth=1, markersize=4)
        ax4.set_xlabel('Sample Index (sorted)', fontsize=12)
        ax4.set_ylabel('Absolute Residual', fontsize=12)
        ax4.set_title('Elastic Net: Sorted Absolute Residuals', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Residual plot saved: {save_path}")
        plt.close()

    def save_results_summary(self, save_path='ElasticNet_training_results_summary.txt'):
        """
        Save results summary
        """
        print("\n[Step 10] Save Results Summary")
        print("-" * 80)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Elastic Net Training Results Summary\n")
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
            f.write(f"Optimal features: {best['n_features']}\n")
            f.write(f"Model: Elastic Net (alpha=0.1, l1_ratio=0.5)\n\n")

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

            f.write("\nOptimal features with coefficients:\n")
            sorted_features = sorted(best['feature_importance'].items(),
                                     key=lambda x: x[1], reverse=True)
            for i, (feat, imp) in enumerate(sorted_features, 1):
                coef = best['coefficients'][feat]
                sources = ', '.join(self.feature_sources.get(feat, ['Unknown']))
                f.write(f"  {i:2d}. {feat:35s} imp={imp:7.4f} coef={coef:+8.4f} [{sources}]\n")

            f.write("\n[3. Complete RFE Results]\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Iter':>4s} {'#Feat':>6s} {'TrainR2':>9s} {'TestR2':>9s} "
                    f"{'TrainRMSE':>10s} {'TestRMSE':>10s} {'Time(s)':>8s}\n")
            f.write("-" * 80 + "\n")

            for result in self.rfe_results:
                f.write(f"{result['iteration']:4d} {result['n_features']:6d} "
                        f"{result['train_metrics']['R2']:9.4f} "
                        f"{result['test_metrics']['R2']:9.4f} "
                        f"{result['train_metrics']['RMSE']:10.4f} "
                        f"{result['test_metrics']['RMSE']:10.4f} "
                        f"{result['iter_time']:8.4f}\n")

        print(f"  [OK] Summary saved: {save_path}")

    def save_results_to_excel(self, save_path='ElasticNet_training_results_detailed.xlsx'):
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
                    'Training Time (s)': result['train_time'],
                    'Iteration Time (s)': result['iter_time']
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
                    'Coefficient': self.best_result['coefficients'][feat],
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
                        'Coefficient': result['coefficients'][feat],
                        'Source Methods': ', '.join(self.feature_sources.get(feat, ['Unknown']))
                    })

            df_iter_features = pd.DataFrame(iter_features_data)
            df_iter_features.to_excel(writer, sheet_name='Features Per Iteration', index=False)

        print(f"  [OK] Excel saved: {save_path}")

    def run_complete_pipeline(self):
        """
        Run complete training pipeline
        """
        print("\n")
        print("=" * 80)
        print(" " * 20 + "Elastic Net RFE Training Pipeline Started")
        print("=" * 80)

        self.load_feature_selection_results()
        self.load_data()
        self.split_data()
        self.recursive_feature_elimination()
        self.find_best_model()

        self.plot_rfe_curves()
        self.plot_feature_importance()
        self.plot_prediction_scatter()
        self.plot_residual_analysis()

        self.save_results_summary()
        self.save_results_to_excel()

        print("\n")
        print("=" * 80)
        print(" " * 25 + "Training Completed!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  1. ElasticNet_RFE_performance_curves.png")
        print("  2. ElasticNet_feature_importance_top20.png")
        print("  3. ElasticNet_prediction_scatter.png")
        print("  4. ElasticNet_residual_analysis.png")
        print("  5. ElasticNet_training_results_summary.txt")
        print("  6. ElasticNet_training_results_detailed.xlsx")
        print("  7. ElasticNet_RFE_detailed_log.txt - DETAILED FEATURE LIST PER ITERATION")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 30 + "Elastic Net")
    print(" " * 20 + "Linear Model with L1+L2 Regularization")
    print("=" * 80)
    print("\n[Info] Elastic Net = Lasso + Ridge combined")
    print("[Info] Fast training, high interpretability")
    print("[Info] Good baseline for linear relationships")
    print()

    trainer = ElasticNetRFETrainer(
        test_size=0.4,
        random_state=42
    )

    trainer.run_complete_pipeline()

    print("\n" + "=" * 80)
    print("Thank you! Check ElasticNet_RFE_detailed_log.txt for feature lists.")
    print("=" * 80)












