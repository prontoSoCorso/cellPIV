import sys
import os
import pandas as pd
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch

from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, confusion_matrix, f1_score
import timeit
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import joblib

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf


class InvalidHeadModelError(Exception):
    """Eccezione sollevata quando il modello scelto per la classificazione (post ROCKET) non è valido."""
    SystemExit()


# Funzione per caricare i dati normalizzati da CSV
def load_data(csv_file_path):
    return pd.read_csv(csv_file_path)

def classification_model(head_type="RF"):
    if head_type.upper()=="RF":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,              # Limit depth
            min_samples_split=10,       # Require 10 samples to split
            max_features='sqrt',        # Use sqrt(n_features) per split
            n_jobs=-1,
            class_weight='balanced'    # Handle class imbalance
        )
        

    elif head_type.upper()=="LR":
        model = LogisticRegression(
            max_iter=5000, 
            solver='saga',  # Better for large datasets
            penalty='elasticnet',  # Optional: Mix of L1/L2 regularization, 
            l1_ratio=0.5,  # Only if using elasticnet
            class_weight='balanced'
            )
        
    elif head_type.upper()=="XGB":
        model = XGBClassifier(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    else:
        raise InvalidHeadModelError(f"Il modello selezionato, {head_type}, non è implementato")


    return model

def find_best_threshold(model, X_val, y_val, thresholds=np.linspace(0.1, 0.9, 9)):
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
def evaluate_model(model, X, y, threshold=0.5):
    """ Model.predict_proba(X)[:, 1] per ottenere la probabilità della classe positiva --> poi classifica i campioni con la soglia (standard=0.5) """
    y_prob = model.predict_proba(X)[:, 1]  # Probabilità della classe positiva
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "kappa": cohen_kappa_score(y, y_pred),
        "brier": brier_score_loss(y, y_prob, pos_label=1),
        "f1": f1_score(y, y_pred, zero_division="warn"),
        "conf_matrix": confusion_matrix(y, y_pred)
    }


# Funzione per salvare la matrice di confusione come immagine
def save_confusion_matrix(conf_matrix, filename, num_kernels):
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {num_kernels} Kernels")
    plt.savefig(filename)
    plt.close()

def main(train_path="", val_path="", test_path="", default_path=True, 
         kernels=conf.kernels_set, seed=conf.seed, output_dir_conf_mat=parent_dir, 
         output_model_dir=os.path.join(parent_dir, "_04_test"), 
         days_to_consider=conf.days_to_consider, type_model_classification="randomforest",
         most_important_metric="balanced_accuracy"):
    
    os.makedirs(output_dir_conf_mat, exist_ok=True)
    best_metric_dict = {}
    best_metric = 0
    best_kernel = None
    final_threshold = 0.5

    if default_path:
        # Ottieni i percorsi dal config
        train_path, val_path, test_path = conf.get_paths(days_to_consider)

    # Carico i dati normalizzati
    df_train = load_data(train_path)
    df_val = load_data(val_path)
    df_test = load_data(test_path)

    # Seleziono le colonne che contengono i valori temporali e creo X e y (labels) 
    temporal_columns = [col for col in df_train.columns if col.startswith("value_")]

    X_train, y_train = df_train[temporal_columns].values, df_train['BLASTO NY'].values
    X_val, y_val = df_val[temporal_columns].values, df_val['BLASTO NY'].values
    X_test, y_test = df_test[temporal_columns].values, df_test['BLASTO NY'].values

    X_train = X_train[:, np.newaxis, :]
    X_val = X_val[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    for kernel in kernels:
        # Definisco il modello (Rocket e non RocketClassifier così posso estrarre probabilità)
        rocket = Rocket(num_kernels=kernel,
                        random_state=conf.seed_everything(seed), 
                        n_jobs=-1)

        # Transform training/validation data
        X_train_features = rocket.fit_transform(X_train)
        X_val_features = rocket.transform(X_val)
        
        # Train a probabilistic classifier on the features
        model = classification_model(type_model_classification)
        model.fit(X_train_features, y_train)

        # Find best threshold for this kernel
        best_threshold = find_best_threshold(model, X_val_features, y_val)

        # Evaluate with best threshold
        train_metrics = evaluate_model(model, X_train_features, y_train, threshold=best_threshold)
        val_metrics = evaluate_model(model, X_val_features, y_val, threshold=best_threshold)

        # Salva l'accuratezza del test nel dizionario
        best_metric_dict[kernel] = val_metrics[most_important_metric]

        # Stampa dei risultati per il train set
        print(f'=====RESULTS WITH {kernel} KERNELS=====')
        for metric, value in train_metrics.items():
            if metric != "conf_matrix":
                print(f"Train {metric.capitalize()}: {value:.4f}")

        for metric, value in val_metrics.items():
            if metric != "conf_matrix":
                print(f"Validation {metric.capitalize()}: {value:.4f}")

        # Aggiorna il modello migliore se l'accuratezza sul test è la migliore trovata finora
        if val_metrics[most_important_metric] > best_metric:
            best_metric, best_kernel, final_threshold = val_metrics[most_important_metric], kernel, best_threshold

    # Stampa il dizionario delle accuratezze
    print("Accuratezza su test set per ogni kernel:", best_metric_dict)

    # Stampa il modello con la migliore accuratezza
    print(f"Best Model: {best_kernel} kernels, {most_important_metric}: {best_metric:.4f}, Threshold: {final_threshold:.4f}")

    # Rialleno il modello su tutti i dati con il numero di kernel migliore che ho trovato e lo salvo
    # Combina i due DataFrame in un unico DataFrame
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    X_all, y_all = df_all[temporal_columns].values, df_all['BLASTO NY'].values
    X_all = X_all[:, np.newaxis, :]  # Reshape to (num_series, num_features, num_time_steps)
    
    final_rocket = Rocket(num_kernels=best_kernel,
                          random_state=conf.seed_everything(seed),
                          n_jobs=-1)
    
    X_all_features = final_rocket.fit_transform(X_all)
    final_model = classification_model(type_model_classification)
    final_model.fit(X_all_features, y_all)

    # Valutazione finale sul test set (with the best threshold)
    X_test_features = final_rocket.transform(X_test)
    test_metrics = evaluate_model(final_model, X_test_features, y_test, threshold=final_threshold)

    print("\n===== FINAL TEST RESULTS =====")
    for metric, value in test_metrics.items():
        if metric != "conf_matrix":
            print(f"{metric.capitalize()}: {value:.4f}")


    cm_path = os.path.join(output_dir_conf_mat, f"confusion_matrix_ROCKET_{days_to_consider}Days.png")
    save_confusion_matrix(conf_matrix=test_metrics["conf_matrix"], filename=cm_path, num_kernels=best_kernel)

    # Save both rocket transformer and classifier
    best_model_path = os.path.join(output_model_dir, f"best_rocket_model_{days_to_consider}Days.joblib")
    joblib.dump({
        "rocket": final_rocket,
        "classifier": final_model,
        "final_threshold": final_threshold,
        "num_kernels": best_kernel
    }, best_model_path)
    print(f'Model saved at: {best_model_path}')

    '''
    # Load threshold when loading model
    loaded = joblib.load(best_model_path)
    final_rocket = loaded["rocket"]
    final_model = loaded["classifier"]
    best_threshold = loaded["best_threshold"]
    '''
    

if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(lambda: main(kernels=[50,100,300,500,700,1000,1250,1500,2500,5000,10000], days_to_consider=1, type_model_classification="RF"), number=1)
    print(f"Tempo impiegato per l'esecuzione di Rocket con vari kernel:", execution_time, "secondi")