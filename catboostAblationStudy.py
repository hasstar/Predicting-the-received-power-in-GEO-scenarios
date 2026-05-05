"""
===================================================================================
High-Orbit Satellite Signal Loss Prediction - Iterative Ablation Study (Full Log)
===================================================================================

[Goal]
Conduct a rigorous Iterative Ablation Study (RFE) comparing:
  1. Surface Only Model
  2. Stratified (Proposed) Model

[Output]
Saves DETAILED logs for EVERY iteration step, including:
  - Metrics (Overall, Rainy, Cloudy RMSE)
  - The feature removed at this step
  - The full list of features used at this step

[Author] AI Assistant
[Date] 2025-12-04
===================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import warnings
import sys
import io
import os

warnings.filterwarnings('ignore')
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Plot styling
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

class CatBoostIterativeAblation:
    def __init__(self, test_size=0.4, random_state=42, catboost_iterations=500, catboost_depth=6):
        self.test_size = test_size
        self.random_state = random_state
        self.catboost_iterations = catboost_iterations
        self.catboost_depth = catboost_depth

        self.all_features_union = []
        self.surface_features = []

        self.df_train = None
        self.df_test = None
        self.y_train = None
        self.y_test = None

        self.history = {}

    def load_and_parse_features(self, feature_dir='特征'):
        """Load features and split into Surface vs Stratified candidates."""
        print("\n[Step 1] Loading & Splitting Feature Candidates")
        feature_files = [
            'Lasso_top_35_features.csv', 'RF_selected_features_top35.csv',
            'MIC_minepy_top35_features.csv', 'MIC_sklearn_top35_features.csv',
            'Spearman_top35_features.csv'
        ]

        all_feats = set()
        for filename in feature_files:
            filepath = os.path.join(feature_dir, filename)
            if not os.path.exists(filepath): continue
            try:
                try: df = pd.read_csv(filepath, encoding='utf-8')
                except: df = pd.read_csv(filepath, encoding='gbk')

                col = 'feature' if 'feature' in df.columns else 'Feature' if 'Feature' in df.columns else None
                if col: all_feats.update([str(f).strip() for f in df[col].dropna().tolist()])
            except: pass

        self.all_features_union = sorted(list(all_feats))

        # Split logic: Check for altitude suffixes
        suffix_indicators = ['-8km', '-17km', '-28km', '-37km', '-50km']
        self.surface_features = [f for f in self.all_features_union
                                 if not any(s in f for s in suffix_indicators)]

        print(f"  > Total Candidate Features (Stratified): {len(self.all_features_union)}")
        print(f"  > Surface-Only Candidate Features:       {len(self.surface_features)}")
        return self.all_features_union

    def load_data(self, loss_file='数据/merged_result_loss.csv',
                  ground_file='数据/all_ground_atmosphere_data_merged_new_processed.csv',
                  layer_files=None):
        print("\n[Step 2] Loading Data")
        if layer_files is None:
            layer_files = [
                '数据/atmosphere_region_39_9_116_3_0-8km_processed.csv',
                '数据/atmosphere_region_39_8_116_3_8-17km_processed.csv',
                '数据/atmosphere_region_39_7_116_4_17-28km_processed.csv',
                '数据/atmosphere_region_39_6_116_4_28-37km_processed.csv',
                '数据/atmosphere_region_39_5_116_4_37-50km_processed.csv'
            ]

        df_loss = pd.read_csv(loss_file)
        y = df_loss['loss'].values
        df_ground = pd.read_csv(ground_file)
        layer_names = ['0-8km', '8-17km', '17-28km', '28-37km', '37-50km']
        layer_dfs = []
        for i, fpath in enumerate(layer_files):
            df_l = pd.read_csv(fpath)
            df_l.columns = [f"{col}_{layer_names[i]}" if col != 'pressure_level' else col for col in df_l.columns]
            layer_dfs.append(df_l)

        df_all = pd.concat([df_ground] + layer_dfs, axis=1)

        if 'tp' not in df_all.columns:
            df_all['tp'] = 0
            print("  [WARNING] 'tp' missing, set to 0.")

        if 'tclw' not in df_all.columns:
            if 'tclw_0-8km' in df_all.columns: df_all['tclw'] = df_all['tclw_0-8km']
            else: df_all['tclw'] = 0

        missing = [f for f in self.all_features_union if f not in df_all.columns]
        for m in missing: df_all[m] = 0
        df_all = df_all.fillna(0)

        self.df_all = df_all
        self.y = y

    def split_data(self):
        indices = np.arange(len(self.df_all))
        train_idx, test_idx = train_test_split(indices, test_size=self.test_size, random_state=self.random_state)

        self.df_train = self.df_all.iloc[train_idx].reset_index(drop=True)
        self.df_test = self.df_all.iloc[test_idx].reset_index(drop=True)
        self.y_train = self.y[train_idx]
        self.y_test = self.y[test_idx]

    def run_rfe_process(self, initial_features, scenario_name):
        """
        Runs RFE and records FULL history.
        """
        print(f"\n  >>> Starting RFE for Scenario: {scenario_name}")
        print(f"      Initial Features: {len(initial_features)}")

        current_features = initial_features.copy()
        history = []

        rain_mask = self.df_test['tp'] > 0.00001
        cloud_mask = (self.df_test['tp'] <= 0.00001) & (self.df_test['tclw'] > 0.02)

        total_steps = len(current_features)

        # Loop until 1 feature remains
        for step in range(total_steps):
            n_feats = len(current_features)
            if n_feats == 0: break

            # 1. Train
            X_train = self.df_train[current_features].values
            X_test = self.df_test[current_features].values

            model = CatBoostRegressor(
                iterations=self.catboost_iterations,
                depth=self.catboost_depth,
                learning_rate=0.05,
                loss_function='RMSE',
                random_seed=self.random_state,
                verbose=False,
                allow_writing_files=False
            )
            model.fit(X_train, self.y_train)

            # 2. Evaluate
            y_pred = model.predict(X_test)
            rmse_overall = np.sqrt(mean_squared_error(self.y_test, y_pred))
            rmse_rain = np.sqrt(mean_squared_error(self.y_test[rain_mask], y_pred[rain_mask])) if rain_mask.sum() > 0 else np.nan
            rmse_cloud = np.sqrt(mean_squared_error(self.y_test[cloud_mask], y_pred[cloud_mask])) if cloud_mask.sum() > 0 else np.nan

            # 3. Determine Removal (if not last step)
            removed_feat_name = "None"
            if n_feats > 1:
                importances = model.get_feature_importance()
                min_idx = np.argmin(importances)
                removed_feat_name = current_features[min_idx]

            # 4. Record History
            history.append({
                'n_features': n_feats,
                'RMSE_Overall': rmse_overall,
                'RMSE_Rainy': rmse_rain,
                'RMSE_Cloudy': rmse_cloud,
                'Removed_Feature': removed_feat_name,
                'Remaining_Features': ";".join(current_features) # Join list to string
            })

            # Print progress
            if step % 5 == 0 or n_feats == 1:
                print(f"      [Iter {step+1}/{total_steps}] Feats={n_feats:2d} | Overall={rmse_overall:.4f} | Cloud={rmse_cloud:.4f}")

            # 5. Execute Removal
            if n_feats > 1:
                current_features.pop(min_idx)

        return history

    def save_detailed_history(self, history, filename):
        """
        Saves the full iteration history to CSV.
        """
        df = pd.DataFrame(history)
        # Reorder columns for better readability
        cols = ['n_features', 'RMSE_Overall', 'RMSE_Rainy', 'RMSE_Cloudy', 'Removed_Feature', 'Remaining_Features']
        df = df[cols]
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"  > Detailed log saved to '{filename}'")

    def run_full_study(self):
        print("\n[Step 3] Running Iterative Ablation Study")

        # --- 1. Run Surface Only ---
        print("\n  [A] Processing Surface-Only Baseline...")
        hist_surface = self.run_rfe_process(self.surface_features, "Surface Only")
        self.save_detailed_history(hist_surface, "Ablation_Detail_Surface.csv")

        # --- 2. Run Stratified (Proposed) ---
        print("\n  [B] Processing Stratified (Proposed)...")
        hist_stratified = self.run_rfe_process(self.all_features_union, "Stratified")
        self.save_detailed_history(hist_stratified, "Ablation_Detail_Stratified.csv")

        # --- 3. Best Model Comparison ---
        best_surface = min(hist_surface, key=lambda x: x['RMSE_Overall'])
        best_stratified = min(hist_stratified, key=lambda x: x['RMSE_Overall'])

        print("\n" + "="*80)
        print("ABLATION RESULTS (Best Iterations)")
        print("="*80)

        # Save summary
        data = [
            {'Model': 'Surface Only', 'Best_Feats': best_surface['n_features'],
             'RMSE_Overall': best_surface['RMSE_Overall'], 'RMSE_Cloudy': best_surface['RMSE_Cloudy']},
            {'Model': 'Stratified', 'Best_Feats': best_stratified['n_features'],
             'RMSE_Overall': best_stratified['RMSE_Overall'], 'RMSE_Cloudy': best_stratified['RMSE_Cloudy']}
        ]
        pd.DataFrame(data).to_csv('Ablation_Summary_Best.csv', index=False)
        print("  > Summary saved to 'Ablation_Summary_Best.csv'")

        # Plot
        self.plot_curves(hist_surface, hist_stratified)

    def plot_curves(self, hist_surf, hist_strat):
        # Extract data
        n_surf = [h['n_features'] for h in hist_surf]
        rmse_surf_cloud = [h['RMSE_Cloudy'] for h in hist_surf]

        n_strat = [h['n_features'] for h in hist_strat]
        rmse_strat_cloud = [h['RMSE_Cloudy'] for h in hist_strat]

        plt.figure(figsize=(10, 6))
        plt.plot(n_surf, rmse_surf_cloud, 'o--', color='gray', label='Surface Only', alpha=0.7)
        plt.plot(n_strat, rmse_strat_cloud, 's-', color='#d62728', label='Stratified (Proposed)')

        plt.gca().invert_xaxis()
        plt.xlabel('Number of Features')
        plt.ylabel('Cloudy (No-Rain) RMSE (dB)')
        plt.title('Impact of Vertical Layers on Cloudy Condition Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('Ablation_Cloudy_Comparison.png', dpi=300)
        print("  > Plot saved to 'Ablation_Cloudy_Comparison.png'")

if __name__ == "__main__":
    trainer = CatBoostIterativeAblation()
    trainer.load_and_parse_features()
    trainer.load_data()
    trainer.split_data()
    trainer.run_full_study()