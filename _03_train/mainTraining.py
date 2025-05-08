import sys
import os
import time
import optuna
from pathlib import Path
from optuna.pruners import MedianPruner

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

def create_study(model_name, day, method_optical_flow):
    study_dir = Path(__file__).resolve().parent / "optuna_studies" / method_optical_flow
    study_dir.mkdir(parents=True, exist_ok=True)
    db_path = study_dir / f"{model_name}_day{day}.db"
    storage_name = f"sqlite:///{db_path}"   # TO SEE THE DASHBOARD: optuna-dashboard sqlite:///_03_train/optuna_studies/Farneback/ROCKET_day1.db
    
    pruner = MedianPruner(
        n_startup_trials=10,  # Wait n trials before pruning
        n_warmup_steps=70,    # Wait m epochs before evaluating
        interval_steps=10     # Check pruning every k epochs
    )

    return optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_day{day}",
        storage=storage_name,
        load_if_exists=True,
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=conf.seed)
    )

def main(models_to_train=["ROCKET", "LSTMFCN", "ConvTran"],
         days=[3],
         optimize=conf.optimize_with_optuna,
         method_optical_flow=conf.method_optical_flow):
    
    start_time = time.time()

    for day in days:
        logging_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      "logging_files",
                                      method_optical_flow,
                                      f"day{str(day)}")

        for model in models_to_train:
            if optimize:
                log_final_filename=f'FINAL_train_{model.upper()}_based_on_{method_optical_flow}'
                
                if model.lower() == "rocket":
                    # Run optimization
                    study = create_study(model_name="ROCKET", day=day, method_optical_flow=method_optical_flow)
                    study.optimize(lambda trial: rocket_main(
                        days_to_consider=day,
                        trial=trial,
                        run_test_evaluation=False  # Disable test eval during optimization
                    ), n_trials=conf.optuna_n_trials_ROCKET)

                    # After optimization, retrain with best params and run test evaluation
                    best_params = study.best_params
                    print(f"\n=== Retraining ROCKET with best parameters for day {day} ===")
                    rocket_main(
                        days_to_consider=day,
                        run_test_evaluation=True,  # Enable final test evaluation
                        log_dir=logging_files_dir,
                        log_filename=log_final_filename,
                        
                        kernels=[best_params["kernels"]],  # Pass as list to use single best kernel
                        type_model_classification=best_params["classifier"]
                        )

                if model.lower() == "lstmfcn":
                    # Run optimization
                    study = create_study(model_name="LSTMFCN", day=day, method_optical_flow=method_optical_flow)
                    study.optimize(lambda trial: lstm_main(
                        days_to_consider=day,
                        trial=trial,
                        run_test_evaluation=False  # Disable test eval during optimization
                    ), n_trials=conf.optuna_n_trials_LSTMFCN)

                    # After optimization, retrain with best params and run test evaluation
                    best_params = study.best_params
                    print(f"\n=== Retraining LSTM-FCN with best parameters for day {day} ===")
                    lstm_main(
                        days_to_consider=day,
                        trial=None,
                        run_test_evaluation=True,   # Enable final test evaluation
                        log_dir=logging_files_dir,
                        log_filename=log_final_filename,
                        **best_params
                        )
                    
                if model.lower() == "convtran":
                    # Run optimization
                    study = create_study("ConvTran", day, method_optical_flow)
                    study.optimize(lambda trial: convtran_main(
                        days_to_consider=day,
                        trial=trial,
                        run_test_evaluation=False
                    ), n_trials=conf.optuna_n_trials_ConvTran)

                    # Retrain with best params
                    best_params = study.best_params
                    print(f"\n=== Retraining ConvTran with best parameters for day {day} ===")
                    convtran_main(
                        days_to_consider=day,
                        trial=None,
                        run_test_evaluation=True,
                        log_dir=logging_files_dir,
                        log_filename=log_final_filename,
                        **best_params
                        )
                    
                print(f"Best parameters for {model} (day {day}):")
                print(study.best_params)

            else:
                # Original training logic
                if model.lower() == "rocket":
                    rocket_main(days_to_consider=day, 
                              train_path="", val_path="", test_path="", 
                              default_path=True,
                              kernels=conf.kernel_number_ROCKET,
                              type_model_classification=conf.type_model_classification,
                              log_dir=logging_files_dir,
                              log_filename=f'train_ROCKET_based_on_{method_optical_flow}')
                
                # LSTM-FCN
                if model.lower() == "lstmfcn":
                    lstm_main(
                        days_to_consider=day,
                        train_path="", val_path="", test_path="",
                        default_path=True,
                        run_test_evaluation=True,
                        log_dir=logging_files_dir,
                        log_filename=f'train_LSTMFCN_based_on_{method_optical_flow}',
                        # default hyperparameters from config.py
                        batch_size=conf.batch_size_FCN,
                        dropout=conf.dropout_FCN,
                        filter_sizes=conf.filter_sizes_FCN,
                        kernel_sizes=conf.kernel_sizes_FCN,
                        lstm_size=conf.lstm_size_FCN,
                        num_layers=conf.num_layers_FCN,
                        attention=conf.attention_FCN,
                        learning_rate=conf.learning_rate_FCN,
                        final_epochs=conf.final_epochs_FCN
                    )

                # ConvTran
                if model.lower() == "convtran":
                    convtran_main(
                        days_to_consider=day,
                        train_path="", val_path="", test_path="",
                        default_path=True,
                        run_test_evaluation=True,
                        log_dir=logging_files_dir,
                        log_filename=f'train_ConvTran_based_on_{method_optical_flow}',
                        # default hyperparameters from config.py
                        emb_size=conf.emb_size,
                        num_heads=conf.num_heads,
                        dropout=conf.dropout,
                        batch_size=conf.batch_size,
                        lr=conf.lr,
                        epochs=conf.epochs
                    )

    print(f"Total execution time for {models_to_train}: {time.time() - start_time:.2f} seconds\n")


if __name__ == "__main__":
    conf.seed_everything(conf.seed)
    main()