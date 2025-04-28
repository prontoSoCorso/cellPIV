import sys
import os
import pandas as pd
import logging
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import optuna
import joblib

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

import _utils_._utils as utils
from config import Config_03_train_with_optimization as conf

class InvalidHeadModelError(Exception):
    """Eccezione sollevata quando il modello scelto per la classificazione (post ROCKET) non è valido."""
    pass

def classification_model(head_type="RF"):
    if head_type.upper()=="RF":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=conf.seed,
            max_depth=10,              # Limit depth
            min_samples_split=10,       # Require 10 samples to split
            max_features='sqrt',        # Use sqrt(n_features) per split
            n_jobs=-1,
            class_weight='balanced'    # Handle class imbalance
        )

    elif head_type.upper()=="LR":
        model = LogisticRegression(
            max_iter=5000,
            random_state=conf.seed,
            solver='saga',  # Better for large datasets
            penalty='elasticnet',  # Optional: Mix of L1/L2 regularization, 
            l1_ratio=0.5,  # Only if using elasticnet
            class_weight='balanced'
            )
        
    elif head_type.upper()=="XGB":
        model = XGBClassifier(device="cuda" if torch.cuda.is_available() else "auto",
                              random_state=conf.seed)

    else:
        raise InvalidHeadModelError(f"Il modello selezionato, {head_type}, non è implementato")


    return model

def find_best_threshold(model, X_val, y_val, thresholds=np.linspace(0.0, 1.0, 101)):
    """ Trova la migliore soglia basata sulla balanced accuracy sul validation set. """
    y_prob = model.predict_proba(X_val)[:, 1]  # Probabilità della classe positiva
    best_threshold = 0.5
    best_balanced_accuracy = 0.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        acc = balanced_accuracy_score(y_val, y_pred)
        
        if acc > best_balanced_accuracy:
            best_balanced_accuracy = acc
            best_threshold = threshold

    return best_threshold

# Funzione per valutare il modello con metriche estese
def evaluate_model(model, X, y_true, threshold=0.5):
    """ Model.predict_proba(X)[:, 1] per ottenere la probabilità della classe positiva --> poi classifica i campioni con la soglia (standard=0.5) """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = utils.calculate_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    return metrics


def train_evaluate_rocket(X_train, y_train, X_val, y_val, 
                         num_kernels, classifier_type, 
                         temporal_columns, trial=False):
    rocket = Rocket(num_kernels=num_kernels,
                    random_state=conf.seed, 
                    n_jobs=-1)

    X_train_features = rocket.fit_transform(X_train)
    X_val_features = rocket.transform(X_val)
    
    model = classification_model(classifier_type)
    model.fit(X_train_features, y_train)

    best_threshold = find_best_threshold(model, X_val_features, y_val)
    val_metrics = evaluate_model(model, X_val_features, y_val, threshold=best_threshold)

    if not trial:
        logging.info(f'===== ROCKET - {conf.days_to_consider} DAYS =====')
        logging.info(f'===== RESULTS WITH {num_kernels} KERNELS =====')
        for metric, value in val_metrics.items():
            if metric not in ("conf_matrix", "fpr", "tpr"):
                logging.info(f"Validation {metric.capitalize()}: {value:.4f}")

    return val_metrics[conf.most_important_metric]


def train_final_model(df_train, df_val, df_test, best_kernel, classifier_type,
                     temporal_columns, output_model_base_dir, days_to_consider,
                     save_plots, output_dir_plots, log_dir, log_filename):
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    X_all, y_all = df_all[temporal_columns].values, df_all['BLASTO NY'].values
    X_all = X_all[:, np.newaxis, :]
    
    final_rocket = Rocket(num_kernels=best_kernel,
                         random_state=conf.seed_everything(conf.seed),
                         n_jobs=-1)
    
    X_all_features = final_rocket.fit_transform(X_all)
    final_model = classification_model(classifier_type)
    final_model.fit(X_all_features, y_all)

    # Test evaluation
    if df_test is not None:
        X_test, y_test = df_test[temporal_columns].values, df_test['BLASTO NY'].values
        X_test = X_test[:, np.newaxis, :]
        X_test_features = final_rocket.transform(X_test)
        best_threshold = find_best_threshold(final_model, X_test_features, y_test)
        test_metrics = evaluate_model(final_model, X_test_features, y_test, threshold=best_threshold)

        logging.info("\n===== FINAL TEST RESULTS =====")
        for metric, value in test_metrics.items():
            if metric not in ("conf_matrix", "fpr", "tpr"):
                logging.info(f"{metric.capitalize()}: {value:.4f}")

        if save_plots:
            complete_output_dir = os.path.join(output_dir_plots, f"day{days_to_consider}")
            os.makedirs(complete_output_dir, exist_ok=True)
            conf_matrix_filename=os.path.join(complete_output_dir,f'confusion_matrix_ROCKET_{days_to_consider}Days.png')
            utils.save_confusion_matrix(conf_matrix=test_metrics['conf_matrix'], 
                                      filename=conf_matrix_filename, 
                                      model_name="ROCKET",
                                      num_kernels=best_kernel)
            utils.plot_roc_curve(fpr=test_metrics['fpr'], tpr=test_metrics['tpr'], 
                               roc_auc=test_metrics['roc_auc'], 
                               filename=conf_matrix_filename.replace('confusion_matrix', 'roc'))

    # Save model
    best_model_path = os.path.join(output_model_base_dir, f"best_rocket_model_{days_to_consider}Days.joblib")
    joblib.dump({
        "rocket": final_rocket,
        "classifier": final_model,
        "final_threshold": best_threshold,
        "num_kernels": best_kernel
    }, best_model_path)
    logging.info(f'Model saved at: {best_model_path}')

    return final_model, final_rocket



def main(train_path="", val_path="", test_path="", default_path=True, 
         kernels=conf.kernels_set, 
         seed=conf.seed,
         save_plots=conf.save_plots,
         output_dir_plots=conf.output_dir_plots,
         output_model_base_dir=conf.output_model_base_dir, 
         days_to_consider=conf.days_to_consider, 
         type_model_classification=conf.type_model_classification,
         most_important_metric=conf.most_important_metric,
         log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging_files"),
         log_filename=f'train_ROCKET_based_on_{conf.method_optical_flow}',
         trial=None,  # Optuna trial object
         run_test_evaluation=conf.run_test_evaluation):
    
    if trial is None:
        utils.config_logging(log_dir=log_dir, log_filename=log_filename)

    os.makedirs(output_model_base_dir, exist_ok=True)
    final_threshold = 0.5

    if default_path:
        # Ottieni i percorsi dal config
        train_path, val_path, test_path = conf.get_paths(days_to_consider)

    # Carico i dati normalizzati
    df_train = utils.load_data(train_path)
    df_val = utils.load_data(val_path)
    df_test = utils.load_data(test_path) if run_test_evaluation else None

    # Seleziono le colonne che contengono i valori temporali e creo X e y (labels) 
    temporal_columns = [col for col in df_train.columns if col.startswith("value_")]

    X_train, y_train = df_train[temporal_columns].values, df_train['BLASTO NY'].values
    X_val, y_val = df_val[temporal_columns].values, df_val['BLASTO NY'].values
    
    X_train = X_train[:, np.newaxis, :]
    X_val = X_val[:, np.newaxis, :]

    if trial:
        kernel = trial.suggest_categorical("kernels", conf.rocket_kernels_options)
        classifier_type = trial.suggest_categorical("classifier", conf.rocket_classifier_options)
        return train_evaluate_rocket(
            X_train, y_train, X_val, y_val,
            kernel, classifier_type,
            temporal_columns, trial=True
        )
    
    else:
        best_metric_dict = {}
        best_metric = 0
        best_kernel = None

        for kernel in kernels:
            val_metric = train_evaluate_rocket(
                X_train, y_train, X_val, y_val,
                kernel, type_model_classification,
                temporal_columns
            )
            best_metric_dict[kernel] = val_metric

            # Aggiorna il modello migliore se l'accuratezza sul test è la migliore trovata finora
            if val_metric > best_metric:
                best_metric, best_kernel = val_metric, kernel
        
        if run_test_evaluation:
            final_model, final_rocket = train_final_model(
                df_train, df_val, df_test,
                best_kernel, type_model_classification,
                temporal_columns, output_model_base_dir,
                days_to_consider, save_plots,
                output_dir_plots, log_dir, log_filename
            )

        return best_metric_dict


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    if conf.optimize_with_optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: main(trial=trial, run_test_evaluation=False), 
                      n_trials=conf.optuna_n_trials)
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        main(days_to_consider=7)
    
    logging.info(f"Total execution time ROCKET: {(time.time() - start_time):.2f}s")