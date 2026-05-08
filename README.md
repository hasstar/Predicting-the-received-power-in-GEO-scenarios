# Predicting-the-received-power-in-GEO-scenarios
It includes some data and related machine learning code, using different models to learn and predict power data.3


以下文件均可直接运行（在满足输入情况的条件下）：

（0）6个模型的训练文件
catboostCode.py
xgBoostCode.py
lightGBMCode.py
SVMCode.py
MLPCode.py
elasticnetCode.py
---->都不能部署

输入：“特征”文件夹所有csv文件
输入：“数据”文件夹所有csv文件

（1）catboost模型的部署文件
catboostCode_suitable_testdata_shap.py
输入：“特征”文件夹所有csv文件
输入：“数据”文件夹所有csv文件

输出：final_model.cbm         ；   final_features.json
输出：shap物理解释图
输出：测试集上的表现图“test_set_analysis.png”
输出：测试集的预测值与实际值以及相关大气参数表”test_set_predictions_only.csv“


（2）catboost模型的实际应用文件
catboostCode_delopyfin.py
输入：”数据“文件夹里所有csv文件
输出：预测值 final_predictions.csv

catboostCode_deloyfin_plot.py
输入：预测值 final_predictions.csv
输入：实际值 数据文件夹里的merged_result_loss.csv
输出：对比图 prediction_comparison_plot.png

（3）小尺度每分钟最优的拟合分布图
analyze_all_days_plot1.py
输入granularity_analysis_summary.csv文件

（4）小尺度时间粒度为15分钟的分布与实测对比图
analyze_all_days_plot2.py
输入sample_15min_data.csv文件
输出：Plot2_PDF_Analysis_English.pdf


analyze_all_days_plot1.py
输入sample_15min_data.csv文件
输出：Plot1_With_90Benchmark.pdf

（5）不同的模型R2随着迭代次数的变化图
plotPaperV2.py
输入：allResultDetailed.xlsx（根据不同模型的训练结果手动整理的）
输出： R2_Clean_{timestamp}.pdf

（6）测试集加上时间戳
add_time_T_predictions.py
输入：测试集的预测值与实际值以及相关大气参数表”test_set_predictions_only.csv“
输入：带有时间戳的全部每小时loss数据文件”数据/loss_data_and_1h_time.csv“
输出：带有时间戳的测试集数据文件test_set_predictions_time_only.csv

（7）测试集的小时数据与秒级数据对齐
extract_hourly
_peak_power_code.py
输入：测试及小时数据 test_set_predictions_time_only.csv
输入：秒级数据 数据/snapshot_SinglePoint_results_2s.csv
输出：测试集对应的秒级数据 detailed_hourly_secondly_data.csv

（8）weibull分布的最坏情况和典型情况
weibull_parameter_seak.py
输入：秒级数据 数据/snapshot_SinglePoint_results_2s.csv
输出：Worst_Case_Analysis_Fixed.pdf

（9）weibull分布参数随着时间粒度的变化情况
weibull_parameter_cal.py
输入：秒级数据 数据/snapshot_SinglePoint_results_2s.csv
输出：weibull_params_evolution.csv、Weibull_Params_Evolution.png

（10）大尺度+小尺度在测试集上的结果对比图
multiScaleTimeFusionv2.py
输入：测试集对应的秒级数据 detailed_hourly_secondly_data.csv
输入：带有时间戳的测试集数据文件 test_set_predictions_time_only.csv
输出：大小尺度在测试集上的实测数据与预测数据对比 comparison_results_cleaned.csv
输出：similarity_check_plot_cleaned.png

（11）大小尺度预测结果与实测的分布ccdf与ks值
ccdf_plot_ks.py、ccdf_plot.py
输入：大小尺度在测试集上的实测数据与预测数据对 comparison_results_cleaned.csv
输出：ccdf_comparison_plot.png

（12）消融实验
catboostAblationStudyv2.py
输入：“特征”文件夹所有csv文件
输入：“数据”文件夹所有csv文件

输出：带有高空气象参数的训练迭代结果，以及只有地面气象参数的训练迭代结果
输出：Ablation_Overall_and_Cloudy.png
输出：Ablation_Detail_Stratified.csv、Ablation_Detail_Surface.csv
输出：Ablation_Summary_Best.csv

（13）ITU计算，在”zhbCodePython“文件夹
total_attenuation_analysis.py
输入：”数据“文件夹
输出：comparison_results.csv是ITU汇总计算结果

plot.py
输入：comparison_results.csv
输出：实测与ITU计算对比图片

(14)三个对比图multiScaleTimeFusionV3.py

输出：Ablation_Three_Scenarios.pdf
