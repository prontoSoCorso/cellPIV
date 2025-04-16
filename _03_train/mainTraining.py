import sys
import os
import time

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

def main(models_to_train = ["ROCKET", "ConvTran", "LSTMFCN"],
         days=[1,3]):
    start_time = time.time()

    for day in days:
        logging_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                        "logging_files",
                                                        f"day{str(day)}")

        for model in models_to_train:
            if model.lower() == "rocket":
                rocket_main(days_to_consider=day, 
                            train_path="", val_path="", test_path="", default_path=True, 
                            kernels=conf.kernels_set, 
                            seed=conf.seed,
                            save_plots=conf.save_plots, 
                            output_dir_plots=conf.output_dir_plots, 
                            output_model_base_dir=conf.output_model_base_dir,
                            type_model_classification=conf.type_model_classification,
                            most_important_metric=conf.most_important_metric,
                            log_dir=logging_files_dir,
                            log_filename=f'train_ROCKET_based_on_{conf.method_optical_flow}'
                            )

            elif model.lower()=="lstmfcn":
                lstm_main(days_to_consider=day,
                        train_path="", val_path="", test_path="", default_path=True, 
                        save_plots=conf.save_plots,
                        output_dir_plots = conf.output_dir_plots, 
                        output_model_base_dir=conf.output_model_base_dir,
                        batch_size=conf.batch_size_FCN, 
                        lstm_size=conf.lstm_size_FCN,
                        filter_sizes=conf.filter_sizes_FCN,
                        kernel_sizes=conf.kernel_sizes_FCN,
                        dropout=conf.dropout_FCN,
                        num_layers=conf.num_layers_FCN,
                        learning_rate=conf.learning_rate_FCN,
                        num_epochs=conf.num_epochs_FCN,
                        most_important_metric = conf.most_important_metric,
                        log_dir=logging_files_dir,
                        log_filename=f'train_LSTMFCN_based_on_{conf.method_optical_flow}'
                        )
                print(f"Tempo esecuzione LSTM-FCN: {(time.time() - start_time):.2f}s")


            elif model.lower()=="convtran":
                convtran_main(days_to_consider=day, 
                            train_path="", val_path="", test_path="", default_path=True, 
                            save_plots=conf.save_plots,
                            output_dir_plots = conf.output_dir_plots,
                            output_model_base_dir=conf.output_model_base_dir,
                            most_important_metric = conf.most_important_metric,
                            log_dir=logging_files_dir,
                            log_filename=f'train_ConvTran_based_on_{conf.method_optical_flow}')
        
    

    print(f"Execution time for {models_to_train}: {time.time() - start_time:.2f} seconds\n")


    
if __name__ == "__main__":
    main()