
```
cellPIV
├─ README.md
├─ _00a_dataPreparation
│  ├─ _01_extract_images.py
│  ├─ _02_extract_equatore.py
│  └─ _mainDataPreparation.py
├─ _00b_preprocessing_excels
│  ├─ _01_prepareExcelAndAddID.py
│  ├─ _02_calculateAndPlotStatistics.py
│  ├─ _03_checkAgeSpermCulture.py
│  ├─ _04_plot_blast_proportions.py
│  ├─ blast_proportions_analysis
│  │  ├─ analysis_metadata.txt
│  │  ├─ interactive_plot.html
│  │  └─ statistical_analysis.txt
│  └─ mainStatsAndExcel.py
├─ _00c_preprocessing_images
│  ├─ _01_check_and_prepare_images.py
│  └─ _mainPreprocessingImages.py
├─ _01_opticalFlows
│  ├─ _opticalFlow_functions.py
│  ├─ _process_optical_flow.py
│  ├─ mainOpticalFlow.py
│  ├─ metrics_examples
│  │  ├─ Farneback
│  │  │  ├─ blasto
│  │  │  └─ no_blasto
│  │  └─ LucasKanade
│  │     ├─ blasto
│  │     └─ no_blasto
│  └─ optical_flow_complete_analysis_Farneback.log
├─ _02_temporalData
│  ├─ _01_fromPklToCsv.py
│  ├─ dim_reduction_files
│  │  └─ Farneback
│  ├─ files_7Days_Farneback
│  │  ├─ hybrid_dict_Farneback.pkl
│  │  ├─ mean_magnitude_dict_Farneback.pkl
│  │  ├─ sum_mean_mag_dict_Farneback.pkl
│  │  └─ vorticity_dict_Farneback.pkl
│  ├─ files_all_days_Farneback
│  │  ├─ hybrid_Farneback.pkl
│  │  ├─ mean_magnitude_Farneback.pkl
│  │  ├─ sum_mean_mag_Farneback.pkl
│  │  └─ vorticity_Farneback.pkl
│  ├─ files_all_days_LucasKanade
│  │  ├─ hybrid_LucasKanade.pkl
│  │  ├─ mean_magnitude_LucasKanade.pkl
│  │  ├─ sum_mean_mag_LucasKanade.pkl
│  │  └─ vorticity_LucasKanade.pkl
│  ├─ final_series_csv
│  │  ├─ sum_mean_mag_Farneback.csv
│  │  └─ sum_mean_mag_LucasKanade.csv
│  └─ mainTimeSeriesProcessing.py
├─ _02b_normalization
│  ├─ _01_split_normalization.py
│  ├─ _02_visualization.py
│  ├─ dim_reduction_files
│  ├─ examples
│  ├─ examples_stratified
│  └─ mainSplitAndNorm.py
├─ _03_train
│  ├─ _a_ROCKET.py
│  ├─ _b_LSTMFCN.py
│  ├─ _c_ConvTran.py
│  ├─ _c_ConvTranTraining.py
│  ├─ _c_ConvTranUtils.py
│  ├─ logging_files
│  │  ├─ day1
│  │  │  ├─ train_ConvTran_based_on_Farneback
│  │  │  ├─ train_LSTMFCN_based_on_Farneback
│  │  │  └─ train_ROCKET_based_on_Farneback
│  │  └─ day3
│  │     ├─ train_ConvTran_based_on_Farneback
│  │     ├─ train_LSTMFCN_based_on_Farneback
│  │     └─ train_ROCKET_based_on_Farneback
│  └─ mainTraining.py
├─ _04_test
│  ├─ _00_test_metrics_roc_umap_barPlots_cm.py
│  ├─ _01_mcNemarTest.py
│  ├─ _02_bootstrap_evaluation.py
│  ├─ _03_stratified_evaluation.py
│  ├─ _testFunctions.py
│  ├─ best_models
│  │  ├─ Farneback
│  │  │  ├─ best_convtran_model_1Days.pkl
│  │  │  ├─ best_convtran_model_3Days.pkl
│  │  │  ├─ best_lstmfcn_model_1Days.pth
│  │  │  ├─ best_lstmfcn_model_3Days.pth
│  │  │  ├─ best_rocket_model_1Days.joblib
│  │  │  └─ best_rocket_model_3Days.joblib
│  │  └─ LucasKanade
│  │     ├─ best_convtran_model_1Days.pkl
│  │     ├─ best_convtran_model_3Days.pkl
│  │     ├─ best_lstmfcn_model_1Days.pth
│  │     ├─ best_lstmfcn_model_3Days.pth
│  │     ├─ best_rocket_model_1Days.joblib
│  │     └─ best_rocket_model_3Days.joblib
│  ├─ bootstrap_test_metrics
│  │  ├─ Farneback
│  │  │  └─ pairwise_comparisons_bootstrap_metrics.csv
│  │  └─ LucasKanade
│  │     └─ pairwise_comparisons_bootstrap_metrics.csv
│  ├─ mainTest.py
│  └─ plots_and_metrics_test
│     ├─ Farneback
│     └─ LucasKanade
│        ├─ day 1
│        └─ day 3
├─ _99_ConvTranModel
│  ├─ AbsolutePositionalEncoding.py
│  ├─ Attention.py
│  ├─ analysis.py
│  ├─ loss.py
│  ├─ model.py
│  ├─ optimizers.py
│  └─ utils.py
├─ _utils_
│  ├─ _utils.py
│  ├─ dimReduction.py
│  └─ plot_and_save_stratified_distribution.py
├─ config.py
├─ datasets
│  ├─ BlastoLabels.xlsx
│  ├─ DB morpheus UniPV.xlsx
│  ├─ DB_Morpheus_withID.csv
│  ├─ Farneback
│  │  ├─ FinalDataset.csv
│  │  └─ subsets
│  │     ├─ Normalized_sum_mean_mag_1Days_test.csv
│  │     ├─ Normalized_sum_mean_mag_1Days_train.csv
│  │     ├─ Normalized_sum_mean_mag_1Days_val.csv
│  │     ├─ Normalized_sum_mean_mag_3Days_test.csv
│  │     ├─ Normalized_sum_mean_mag_3Days_train.csv
│  │     ├─ Normalized_sum_mean_mag_3Days_val.csv
│  │     ├─ Normalized_sum_mean_mag_5Days_test.csv
│  │     ├─ Normalized_sum_mean_mag_5Days_train.csv
│  │     ├─ Normalized_sum_mean_mag_5Days_val.csv
│  │     ├─ Normalized_sum_mean_mag_7Days_test.csv
│  │     ├─ Normalized_sum_mean_mag_7Days_train.csv
│  │     └─ Normalized_sum_mean_mag_7Days_val.csv
│  ├─ LucasKanade
│  │  ├─ FinalDataset.csv
│  │  └─ subsets
│  │     ├─ Normalized_sum_mean_mag_1Days_test.csv
│  │     ├─ Normalized_sum_mean_mag_1Days_train.csv
│  │     ├─ Normalized_sum_mean_mag_1Days_val.csv
│  │     ├─ Normalized_sum_mean_mag_3Days_test.csv
│  │     ├─ Normalized_sum_mean_mag_3Days_train.csv
│  │     ├─ Normalized_sum_mean_mag_3Days_val.csv
│  │     ├─ Normalized_sum_mean_mag_5Days_test.csv
│  │     ├─ Normalized_sum_mean_mag_5Days_train.csv
│  │     ├─ Normalized_sum_mean_mag_5Days_val.csv
│  │     ├─ Normalized_sum_mean_mag_7Days_test.csv
│  │     ├─ Normalized_sum_mean_mag_7Days_train.csv
│  │     └─ Normalized_sum_mean_mag_7Days_val.csv
│  └─ pz con doppia dish.xlsx
└─ main.py

```