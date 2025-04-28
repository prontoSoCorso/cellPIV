import sys
import os
import time
import optuna
from pathlib import Path

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train_with_optimization as conf
from _a_ROCKET import main as rocket_main
from _b_LSTMFCN import main as lstm_main
from _c_ConvTran import main as convtran_main

def create_study(model_name, day):
    study_dir = Path(__file__).resolve().parent / "optuna_studies"
    study_dir.mkdir(exist_ok=True)
    db_path = study_dir / f"{model_name}_day{day}.db"
    storage_name = f"sqlite:///{db_path}"
    
    return optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_day{day}",
        storage=storage_name,
        load_if_exists=True
    )

def main(models_to_train=["ROCKET"],
         days=[1],
         optimize=conf.optimize_with_optuna,
         n_trials=conf.optuna_n_trials):
    
    start_time = time.time()

    for day in days:
        logging_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      "logging_files",
                                      f"day{str(day)}")

        for model in models_to_train:
            if optimize:
                if model.lower() == "rocket":
                    study = create_study("ROCKET", day)
                    study.optimize(lambda trial: rocket_main(
                        days_to_consider=day,
                        trial=trial,
                        run_test_evaluation=False
                    ), n_trials=n_trials)
                
                # Add similar blocks for other models when their scripts are updated
                
                print(f"Best parameters for {model} (day {day}):")
                print(study.best_params)
            else:
                if model.lower() == "rocket":
                    rocket_main(days_to_consider=day, 
                              train_path="", val_path="", test_path="", 
                              default_path=True,
                              kernels=conf.rocket_kernels_options if optimize else conf.kernels_set,
                              type_model_classification=conf.type_model_classification,
                              log_dir=logging_files_dir,
                              log_filename=f'train_ROCKET_based_on_{conf.method_optical_flow}')
                
                # Add similar blocks for other models

    print(f"Execution time for {models_to_train}: {time.time() - start_time:.2f} seconds\n")



if __name__ == "__main__":
    conf.seed_everything(conf.seed)
    main()