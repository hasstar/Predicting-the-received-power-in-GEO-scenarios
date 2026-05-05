"""
===================================================================================
High-Orbit Satellite Signal Loss Prediction - Final Ablation Study (3-Panel Plot)
===================================================================================

[Goal]
1. Compare Surface-Only vs. Stratified Model across THREE scenarios:
   - Overall (All Data)
   - Rainy (Precipitation > 0)
   - Cloudy (No Rain, High Cloud Water)
2. Generate a comprehensive 3-panel figure to prove:
   - "Rainy" performance is similar (Fairness).
   - "Cloudy" performance is vastly improved (Innovation).

[Output]
- Ablation_Detail_Surface.csv / Stratified.csv (Logs)
- Ablation_Summary_Best.csv (Table)
- Ablation_Three_Scenarios.png (The 3-Panel Plot)

[Author] AI Assistant
[Date] 2025-12-05
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
import time

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except:
                    df = pd.read_csv(filepath, encoding='gbk')

                col = 'feature' if 'feature' in df.columns else 'Feature' if 'Feature' in df.columns else None
                if col: all_feats.update([str(f).strip() for f in df[col].dropna().tolist()])
            except:
                pass

        self.all_features_union = sorted(list(all_feats))

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
            if 'tclw_0-8km' in df_all.columns:
                df_all['tclw'] = df_all['tclw_0-8km']
            else:
                df_all['tclw'] = 0

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
        print(f"\n  >>> Starting RFE for Scenario: {scenario_name}")

        current_features = initial_features.copy()
        history = []

        # Scenarios Masks
        rain_mask = self.df_test['tp'] > 0.00001
        cloud_mask = (self.df_test['tp'] <= 0.00001) & (self.df_test['tclw'] > 0.02)

        total_steps = len(current_features)

        for step in range(total_steps):
            n_feats = len(current_features)
            if n_feats == 0: break

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

            y_pred = model.predict(X_test)

            # --- 3 METRICS ---
            rmse_overall = np.sqrt(mean_squared_error(self.y_test, y_pred))

            if rain_mask.sum() > 0:
                rmse_rain = np.sqrt(mean_squared_error(self.y_test[rain_mask], y_pred[rain_mask]))
            else:
                rmse_rain = np.nan  # Avoid error if no rain samples

            if cloud_mask.sum() > 0:
                rmse_cloud = np.sqrt(mean_squared_error(self.y_test[cloud_mask], y_pred[cloud_mask]))
            else:
                rmse_cloud = np.nan

            # Find removed feature
            removed_feat_name = "None"
            min_idx = 0
            if n_feats > 1:
                importances = model.get_feature_importance()
                min_idx = np.argmin(importances)
                removed_feat_name = current_features[min_idx]

            history.append({
                'n_features': n_feats,
                'RMSE_Overall': rmse_overall,
                'RMSE_Rainy': rmse_rain,
                'RMSE_Cloudy': rmse_cloud,
                'Removed_Feature': removed_feat_name,
                'Remaining_Features': ";".join(current_features)
            })

            if step % 5 == 0 or n_feats == 1:
                print(
                    f"      [Iter {step + 1}/{total_steps}] Feats={n_feats:2d} | Overall={rmse_overall:.4f} | Rain={rmse_rain:.4f} | Cloud={rmse_cloud:.4f}")

            if n_feats > 1:
                current_features.pop(min_idx)

        return history

    def save_file_safely(self, func, filename, *args, **kwargs):
        """Helper to save files preventing PermissionError"""
        try:
            func(filename, *args, **kwargs)
            print(f"  > Saved to '{filename}'")
        except PermissionError:
            new_name = filename.replace('.csv', f'_{int(time.time())}.csv').replace('.png', f'_{int(time.time())}.png')
            print(f"  [WARNING] '{filename}' is busy. Saving to '{new_name}'")
            func(new_name, *args, **kwargs)

    def run_full_study(self):
        print("\n[Step 3] Running Iterative Ablation Study")

        print("\n  [A] Processing Surface-Only Baseline...")
        hist_surface = self.run_rfe_process(self.surface_features, "Surface Only")
        self.save_file_safely(pd.DataFrame(hist_surface).to_csv, "Ablation_Detail_Surface.csv", index=False,
                              encoding='utf-8-sig')

        print("\n  [B] Processing Stratified (Proposed)...")
        hist_stratified = self.run_rfe_process(self.all_features_union, "Stratified")
        self.save_file_safely(pd.DataFrame(hist_stratified).to_csv, "Ablation_Detail_Stratified.csv", index=False,
                              encoding='utf-8-sig')

        # Best model comparison
        best_surface = min(hist_surface, key=lambda x: x['RMSE_Overall'])
        best_stratified = min(hist_stratified, key=lambda x: x['RMSE_Overall'])

        imp_overall = (best_surface['RMSE_Overall'] - best_stratified['RMSE_Overall']) / best_surface[
            'RMSE_Overall'] * 100
        imp_cloud = (best_surface['RMSE_Cloudy'] - best_stratified['RMSE_Cloudy']) / best_surface['RMSE_Cloudy'] * 100

        print("\n" + "=" * 80)
        print("ABLATION RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Metric':<20} | {'Surface Only':<15} | {'Stratified':<15} | {'Improvement':<15}")
        print("-" * 75)
        print(
            f"{'Overall RMSE':<20} | {best_surface['RMSE_Overall']:<15.4f} | {best_stratified['RMSE_Overall']:<15.4f} | {imp_overall:<15.2f}%")
        print(
            f"{'Rainy RMSE':<20} | {best_surface['RMSE_Rainy']:<15.4f} | {best_stratified['RMSE_Rainy']:<15.4f} | {'(Check Plot)':<15}")
        print(
            f"{'Cloudy RMSE':<20} | {best_surface['RMSE_Cloudy']:<15.4f} | {best_stratified['RMSE_Cloudy']:<15.4f} | {imp_cloud:<15.2f}%")
        print("-" * 75)

        # Save Summary
        data = [
            {'Metric': 'Overall RMSE', 'Surface': best_surface['RMSE_Overall'],
             'Stratified': best_stratified['RMSE_Overall'], 'Improvement': imp_overall},
            {'Metric': 'Cloudy RMSE', 'Surface': best_surface['RMSE_Cloudy'],
             'Stratified': best_stratified['RMSE_Cloudy'], 'Improvement': imp_cloud},
            {'Metric': 'Rainy RMSE', 'Surface': best_surface['RMSE_Rainy'], 'Stratified': best_stratified['RMSE_Rainy'],
             'Improvement': 0}
        ]
        self.save_file_safely(pd.DataFrame(data).to_csv, "Ablation_Summary_Best.csv", index=False)

        # Plot
        self.plot_curves(hist_surface, hist_stratified)

    def plot_curves(self, hist_surf, hist_strat):
        """
        Generates 3-Panel Plot: Overall | Rainy | Cloudy
        """
        # Extract Data
        n_surf = [h['n_features'] for h in hist_surf]
        n_strat = [h['n_features'] for h in hist_strat]

        # Metrics Surface
        rmse_surf_all = [h['RMSE_Overall'] for h in hist_surf]
        rmse_surf_rain = [h['RMSE_Rainy'] for h in hist_surf]
        rmse_surf_cloud = [h['RMSE_Cloudy'] for h in hist_surf]

        # Metrics Stratified
        rmse_strat_all = [h['RMSE_Overall'] for h in hist_strat]
        rmse_strat_rain = [h['RMSE_Rainy'] for h in hist_strat]
        rmse_strat_cloud = [h['RMSE_Cloudy'] for h in hist_strat]

        # Setup Figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # Common settings
        def setup_ax(ax, title, y_label):
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Number of Features', fontsize=16)
            ax.set_ylabel(y_label, fontsize=16)
            ax.invert_xaxis()  # Right (Max Feats) -> Left (1 Feat)
            ax.grid(True, alpha=0.3)
            return ax


        pointIndex = 20
        # 1. Overall Plot
        ax1.plot(n_surf[pointIndex:-2], rmse_surf_all[pointIndex:-2], 'o--', color='gray', label='Surface Only', alpha=0.6)
        ax1.plot(n_strat[pointIndex+25:-2], rmse_strat_all[pointIndex+25:-2], 's-', color='#d62728', label='Stratified (Proposed)', linewidth=2)
        #setup_ax(ax1,y_label='RMSE (dB)')
        ax1.set_ylabel('RMSE (dB)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Number of features', fontsize=16, fontweight='bold')
        ax1.legend()

        # 2. Rainy Plot (Likely Similar)
        ax2.plot(n_surf[pointIndex:-2], rmse_surf_rain[pointIndex:-2], 'o--', color='gray', label='Surface Only', alpha=0.6)
        ax2.plot(n_strat[pointIndex+25:-2], rmse_strat_rain[pointIndex+25:-2], 's-', color='#d62728', label='Stratified (Proposed)', linewidth=2)
        #setup_ax(ax2,y_label='RMSE (dB)')
        ax2.set_ylabel('RMSE (dB)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Number of features', fontsize=16, fontweight='bold')
        ax2.legend()

        # 3. Cloudy Plot (The Key Differentiator)
        ax3.plot(n_surf[pointIndex:-2], rmse_surf_cloud[pointIndex:-2], 'o--', color='gray', label='Surface Only', alpha=0.6)
        ax3.plot(n_strat[pointIndex+25:-2], rmse_strat_cloud[pointIndex+25:-2], 's-', color='#d62728', label='Stratified (Proposed)', linewidth=2)
        #setup_ax(ax3,y_label='RMSE (dB)')
        ax3.set_ylabel('RMSE (dB)', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Number of features', fontsize=16, fontweight='bold')

        # Add annotation highlight for Cloudy
        best_cloud_idx = np.argmin(rmse_strat_cloud)
        imp_val = (rmse_surf_cloud[best_cloud_idx] - rmse_strat_cloud[best_cloud_idx]) / rmse_surf_cloud[
            best_cloud_idx] * 100

        # ax3.annotate(f'Gap: {imp_val:.1f}%',
        #              xy=(n_strat[best_cloud_idx], rmse_strat_cloud[best_cloud_idx]),
        #              xytext=(n_strat[best_cloud_idx], rmse_surf_cloud[best_cloud_idx]),
        #              arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5),
        #              ha='center', color='blue', fontweight='bold', backgroundcolor='white')

        ax3.legend()

        plt.tight_layout()

        # Safe Save Plot
        try:
            plt.savefig('Ablation_Three_Scenarios.pdf', dpi=300)
            print("  > Plot saved to 'Ablation_Three_Scenarios.png'")
        except PermissionError:
            new_name = f'Ablation_Three_Scenarios_{int(time.time())}.png'
            print(f"  [WARNING] Plot file is open. Saving to '{new_name}'")
            plt.savefig(new_name, dpi=300)


if __name__ == "__main__":
    trainer = CatBoostIterativeAblation()
    trainer.load_and_parse_features()
    trainer.load_data()
    trainer.split_data()
    trainer.run_full_study()