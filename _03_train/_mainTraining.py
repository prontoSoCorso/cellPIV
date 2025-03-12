import sys
import os
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf
from _a_ROCKET import main as rocket_main
from _b_LSTMFCN import main as lstm_main
from _c_ConvTran import main as convtran_main


if __name__ == "__main__":
    start_time = time.time()
    models_to_train = ["ROCKET", "ConvTran", "LSTMFCN"]   # possible choices: ["ROCKET","LSTMFCN","ConvTran"]
    day = 3
    output_dir_name = "tmp_test_results_after_training"

    for model in models_to_train:
        if model.lower == "rocket":
            rocket_main(train_path="", val_path="", test_path="", default_path=True, 
                        kernels=[50,100,300,500,700,1000,1250,1500,2500,5000,10000], 
                        seed=conf.seed, 
                        output_dir_plots=os.path.join(current_dir, output_dir_name), 
                        output_model_dir=os.path.join(parent_dir, "_04_test"), 
                        days_to_consider=day, type_model_classification="RF",
                        most_important_metric="balanced_accuracy")

        elif model.lower()=="lstmfcn":
            lstm_main(days_to_consider=day,
                      output_dir_plots = os.path.join(current_dir, output_dir_name), 
                      batch_size=conf.batch_size_FCN, 
                      lstm_size=conf.lstm_size_FCN,
                      filter_sizes=conf.filter_sizes_FCN,
                      kernel_sizes=conf.kernel_sizes_FCN,
                      dropout=conf.dropout_FCN,
                      num_layers=conf.num_layers_FCN,
                      learning_rate=conf.learning_rate_FCN,
                      num_epochs=conf.num_epochs_FCN)
            print(f"Tempo esecuzione LSTM-FCN: {(time.time() - start_time):.2f}s")


        elif model.lower()=="convtran":
            convtran_main(days_to_consider=day, 
                          save_conf_matrix=True,
                          output_dir_plots = os.path.join(current_dir, output_dir_name))
    
    

    print(f"Execution time for {models_to_train}: {time.time() - start_time:.2f} seconds\n")


    
    
    