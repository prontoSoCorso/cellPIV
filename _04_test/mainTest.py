import sys 
import os
import time
# Aggiungo il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf_train
from config import user_paths
from _04_test._00_test_metrics_roc_umap_barPlots_cm import test_all
from _04_test._01_mcNemarTest import compare_with_McNemar
from _04_test._02_bootstrap_evaluation import boostrap_evaluation
from _04_test._03_stratified_evaluation import stratified_evaluation


def main(do_test_all=False, 
         do_McNemar=False, 
         do_bootstrap=False, 
         do_stratified_evaluation=False,
         models = ['ROCKET', 'LSTMFCN', 'ConvTran'],
         days_for_analysis = [],
         days_mcNemar = [],
         base_models_path = conf_train.output_model_base_dir,
         base_test_csv_path = user_paths.dataset,
         method_optical_flow = conf_train.method_optical_flow
         ):
    
    # Test different models for different days
    if do_test_all:
        base_plot_path = os.path.join(current_dir, "plots_and_metrics_test", method_optical_flow)
        test_all(
            base_path = base_plot_path,
            days=days_for_analysis,
            models = models,
            base_models_path=base_models_path
            )
        
    # McNemar test (select 2 days to compare)
    if do_McNemar:
        model_type = models
        base_output_dir = os.path.join(current_dir, "mcnemar_results", method_optical_flow)
        compare_with_McNemar(
            models_type=model_type,
            days=days_mcNemar,
            data_dir=base_test_csv_path,
            model_dir=base_models_path,
            output_dir=base_output_dir
            )

    # Test several models on different days with a bootstrap on test (more robust results)
    if do_bootstrap:
        boostrap_evaluation(
            base_path=os.path.join(current_dir, "bootstrap_test_metrics", method_optical_flow),
            days=days_for_analysis,
            models_list=models,
            base_models_path=base_models_path,
            base_test_csv_path=base_test_csv_path
            )

    # Stratified analysis
    if do_stratified_evaluation:
        merge_types = ["anomalous", "not_vital"]    # "anomalous" OR "not_vital" OR "no_merging"
        output_dir_stratified_analysis = os.path.join(current_dir, "stratified_test_results", method_optical_flow)
        stratified_evaluation(
            merge_types=merge_types,
            days=days_for_analysis,
            base_path=output_dir_stratified_analysis,
            base_model_path=base_models_path,
            db_file=path_original_excel,
            model_types=models
            )


if __name__ == "__main__":
    start_time = time.time()
    main(do_test_all=True,
         do_McNemar=False,
         do_bootstrap=False,
         do_stratified_evaluation=True,
         models = ['ROCKET', 'LSTMFCN', 'ConvTran'],
         days_for_analysis = [1,3],
         days_mcNemar = [1,3],
         base_models_path = conf_train.output_model_base_dir,
         base_test_csv_path = user_paths.dataset,
         method_optical_flow = conf_train.method_optical_flow)
    print("Execution time: ", str(time.time()-start_time), "seconds")