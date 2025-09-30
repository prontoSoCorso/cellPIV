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

from config import Config_03_train_with_optimization as conf
import _utils_._utils as utils

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
        model = XGBClassifier(
            device="cuda" if torch.cuda.is_available() else "auto",
            random_state=conf.seed
        )

    else:
        raise InvalidHeadModelError(f"Il modello selezionato, {head_type}, non è implementato")


    return model

def find_best_threshold(model, X_val, y_val, thresholds=np.linspace(0.0, 0.5, 101)):
    """ Trova la migliore soglia basata sulla balanced accuracy sul validation set. """
    y_prob = model.predict_proba(X_val)[:, 1]  # Probabilità della classe positiva
    best_threshold = 0.5
    best_balanced_accuracy = -np.inf

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
                         temporal_columns, trial=False, 
                         day=None):
    rocket = Rocket(num_kernels=num_kernels,
                    random_state=conf.seed, 
                    normalise=False,
                    n_jobs=-1)

    X_train_features = rocket.fit_transform(X_train)
    X_val_features = rocket.transform(X_val)
    
    model = classification_model(classifier_type)
    model.fit(X_train_features, y_train)

    best_threshold = find_best_threshold(model, X_val_features, y_val)
    val_metrics = evaluate_model(model, X_val_features, y_val, threshold=best_threshold)

    if not trial:
        logging.info(f'===== ROCKET - {day} DAYS =====')
        logging.info(f'===== RESULTS WITH {num_kernels} KERNELS, BEST THRESHOLD {best_threshold} =====')
        for metric, value in val_metrics.items():
            if metric not in ("conf_matrix", "fpr", "tpr"):
                logging.info(f"Validation {metric.capitalize()}: {value:.4f}")

    return val_metrics[conf.most_important_metric]



def train_final_model(data_dict, best_kernel, classifier_type,
                      output_model_base_dir, save_plots, output_dir_plots,
                      log_dir, log_filename, day_label=None):
    # Concatena train+val per il modello finale
    X_train, y_train, X_val, y_val, X_test, y_test = utils._check_data_dict(
        data_dict, require_test=(X_test_in := ("X_test" in data_dict and "y_test" in data_dict))
        )
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)

    final_rocket = Rocket(num_kernels=best_kernel,
                          random_state=conf.seed,
                          normalise=False,
                          n_jobs=-1)
    
    X_all_features = final_rocket.fit_transform(X_all)
    final_model = classification_model(classifier_type)
    final_model.fit(X_all_features, y_all)
    best_threshold = find_best_threshold(final_model, X_all_features, y_all)

    # Test evaluation
    test_metrics = None
    if X_test_in and X_test is not None:
        X_test_features = final_rocket.transform(X_test)
        test_metrics = evaluate_model(final_model, X_test_features, y_test, threshold=best_threshold)
        utils.save_results(test_metrics, output_dir_plots, "ROCKET", day_label)


    # Save model
    os.makedirs(output_model_base_dir, exist_ok=True)
    tag = f"{day_label}Days" if day_label is not None else "All"
    best_model_path = os.path.join(output_model_base_dir, f"best_rocket_model_{tag}.joblib")
    joblib.dump({
        "rocket": final_rocket,
        "classifier": final_model,
        "final_threshold": best_threshold,
        "num_kernels": best_kernel
    }, best_model_path)
    logging.info(f'Model saved at: {best_model_path}')

    return final_model, final_rocket, test_metrics


def main(data,
         kernels=conf.kernel_number_ROCKET,
         save_plots=conf.save_plots,
         output_dir_plots=conf.output_dir_plots,
         output_model_base_dir=conf.output_model_base_dir, 
         day_label=conf.days_label, 
         type_model_classification=conf.type_model_classification,
         most_important_metric=conf.most_important_metric,
         log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging_files"),
         log_filename=f'train_ROCKET_based_on_{conf.method_optical_flow}',
         trial=None,  # Optuna trial object
         run_test_evaluation=conf.run_test_evaluation):
    
    if trial is None:
        utils.config_logging(log_dir=log_dir, log_filename=log_filename)

    # Unpack & validate
    X_train, y_train, X_val, y_val, X_test, y_test = utils._check_data_dict(
        data, require_test=bool(run_test_evaluation)
    )

    # Creo la cartella per salvare i modelli se non esiste
    os.makedirs(output_model_base_dir, exist_ok=True)

    # Optuna: scegli iperparametri
    if trial:
        kernel = trial.suggest_categorical("kernels", conf.rocket_kernels_options)
        classifier_type = trial.suggest_categorical("classifier", conf.rocket_classifier_options)
        return train_evaluate_rocket(
            X_train, y_train, X_val, y_val,
            kernel, classifier_type,
            trial=True, day=day_label
        )
    
    # Grid sugli n_kernel
    classifier_type = type_model_classification
    best_metric = -np.inf
    best_kernel = None

    for kernel in kernels:
        score = train_evaluate_rocket(
            X_train, y_train, X_val, y_val,
            kernel, classifier_type,
            temporal_columns=None,
            trial=False, day=day_label
        )
        if score > best_metric:
            best_metric = score
            best_kernel = kernel

    # Train final model & test evaluation
    if run_test_evaluation:
        train_final_model(
            data_dict=data,
            best_kernel=best_kernel,
            classifier_type=classifier_type,
            output_model_base_dir=output_model_base_dir,
            save_plots=save_plots,
            output_dir_plots=output_dir_plots,
            log_dir=log_dir,
            log_filename=log_filename,
            day_label=day_label
        )
    
    return {"best_kernel": best_kernel, most_important_metric: best_metric}


if __name__ == "__main__":
    print("Need to run mainTraining.py instead. This module needs data input as dictionary.")
