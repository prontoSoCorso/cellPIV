import sys
import os

# Rileva il percorso della cartella genitore, che sarà la stessa in cui ho il file da convertire
current_dir = os.path.dirname(os.path.abspath(__file__))

# Individua la cartella 'cellPIV' come riferimento
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while os.path.basename(parent_dir) != "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from _02b_normalization import mainSplitAndNorm
from _03_train import mainTraining
from _04_test import mainTest

def main():
    days = [1,3]
    models_to_train = ["ROCKET", "ConvTran", "LSTMFCN"]

    # COMPLETE PIPELINE AFTER OPTICAL FLOW EXTRACTION

    # Split and Normalization
    inf_quantile = 0.1
    sup_quantile = 0.9
    initial_frames_to_cut = 0
    mainSplitAndNorm.main(days_to_consider=days,
                          save_normalization_example_single_pt=False, 
                          mean_data_visualization=False,
                          specific_patient_to_analyse=False, 
                          mean_data_visualization_stratified=False,
                          inf_quantile=inf_quantile, 
                          sup_quantile=sup_quantile,
                          initial_frames_to_cut=initial_frames_to_cut)

    mainTraining.main(models_to_train = models_to_train,
                      days=days)
    
    mainTest.main(do_test_all=True, 
                  do_McNemar=False, 
                  do_bootstrap=False, 
                  do_stratified_evaluation=True,
                  models=models_to_train,
                  days_for_analysis=days,
                  days_mcNemar=[1,3])


if __name__=="__main__":
    main()



