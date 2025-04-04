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

from _04_test._00_test_metrics_roc_umap_barPlots_cm import test_all
from _04_test._01_mcNemarTest import compare_with_McNemar
from _04_test._02_bootstrap_evaluation import boostrap_evaluation
from _04_test._03_stratified_evaluation import stratified_evaluation


def main():
    test_all(
        base_path = os.path.join(current_dir, "plots_and_metrics_test"), 
        days=[1,3,5,7], 
        models = ['ROCKET', 'LSTMFCN', 'ConvTran'], 
        base_models_path=current_dir, 
        base_test_csv_path=parent_dir
        )
    
    model_type = ["ROCKET","LSTMFCN","ConvTran"]
    days = [1,3]
    data_dir = parent_dir
    model_dir = current_dir
    base_output_dir = os.path.join(current_dir, "mcnemar_results")
    compare_with_McNemar(
        models_type=model_type,
        days=days,
        data_dir=data_dir,
        model_dir=model_dir,
        output_dir=base_output_dir
        )

    boostrap_evaluation(
        base_path=os.path.join(current_dir, "bootstrap_test_metrics"),
        days=[1,3,5,7],
        models_list=['ROCKET', 'LSTMFCN', 'ConvTran'],
        base_models_path=current_dir,
        base_test_csv_path=parent_dir
        )
    
    merge_types = ["anomalous", "not_vital"]    # "anomalous" OR "not_vital" OR "no_merging"
    days_to_consider = [1,3,5,7]        # 1,3,5,7
    stratified_evaluation(
        merge_types=merge_types,
        days=days_to_consider, 
        base_path=os.path.join(current_dir, "stratified_test_results"), 
        base_model_path=current_dir,
        base_test_csv_path=parent_dir,
        db_file=os.path.join(parent_dir, "DB morpheus UniPV.xlsx"),
        model_types=["ROCKET", "LSTMFCN", "ConvTran"]
        )



if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time: ", str(time.time()-start_time), "seconds")
