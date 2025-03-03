import sys
import os
import pandas as pd
from sktime.classification.kernel_based import RocketClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, confusion_matrix, f1_score
import timeit
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf

# Funzione per caricare i dati normalizzati da CSV
def load_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# Funzione per addestrare il modello
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def find_best_threshold(model, X_val, y_val, thresholds=np.linspace(0.1, 0.9, 9)):
    """ Trova la migliore soglia basata sull'accuracy sul validation set. """
    y_prob = model.predict_proba(X_val)[:, 1]  # Probabilità della classe positiva
    best_threshold = 0.5
    best_accuracy = 0.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        acc = accuracy_score(y_val, y_pred)
        
        if acc > best_accuracy:
            best_accuracy = acc
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


def evaluate_model_with_thresholds(model, X, y, thresholds=np.linspace(0.1, 0.9, 9)):
    """ Valuta il modello per diverse soglie di classificazione. """
    y_prob = model.predict_proba(X)[:, 1]  # Probabilità della classe positiva
    results = {}

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)  # Cambia la soglia
        results[threshold] = {
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred),
        }

    return results


# Funzione per salvare la matrice di confusione come immagine
def save_confusion_matrix(conf_matrix, filename, num_kernels):
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {num_kernels} Kernels")
    plt.savefig(filename)
    plt.close()

def main(train_path="", val_path="", test_path="", default_path=True, kernels=conf.kernels_set, seed=conf.seed, output_dir_conf_mat=parent_dir, output_model_dir=os.path.join(parent_dir, "_04_test"), days_to_consider=conf.days_to_consider):
    os.makedirs(output_dir_conf_mat, exist_ok=True)
    accuracy_dict = {}
    best_accuracy = 0
    best_kernel = None

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

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    y_val = y_val.astype(np.int32)

    for kernel in kernels:
        # Definisco il modello RocketClassifier
        model = RocketClassifier(num_kernels=kernel, 
                                 rocket_transform="rocket", 
                                 random_state=conf.seed_everything(seed), 
                                 n_jobs=-1,
                                 estimator=LogisticRegression(multi_class='multinomial', max_iter=1000))

        # Addestramento del modello
        model = train_model(model, X_train, y_train)

        # Valutazione del modello su train e test set
        train_metrics = evaluate_model(model, X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)
        
        # Salva l'accuratezza del test nel dizionario
        accuracy_dict[kernel] = val_metrics["accuracy"]

        # Stampa dei risultati per il train set
        print(f'=====RESULTS WITH {kernel} KERNELS=====')
        for metric, value in train_metrics.items():
            if metric != "conf_matrix":
                print(f"Train {metric.capitalize()}: {value:.4f}")

        for metric, value in val_metrics.items():
            if metric != "conf_matrix":
                print(f"Validation {metric.capitalize()}: {value:.4f}")

        # Aggiorna il modello migliore se l'accuratezza sul test è la migliore trovata finora
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy, best_kernel = val_metrics["accuracy"], kernel

    # Stampa il dizionario delle accuratezze
    print("Accuratezza su test set per ogni kernel:", accuracy_dict)

    # Stampa il modello con la migliore accuratezza
    print(f"Best Model: {best_kernel} kernels, Accuracy: {best_accuracy:.4f}")

    # Rialleno il modello su tutti i dati con il numero di kernel migliore che ho trovato e lo salvo
    # Combina i due DataFrame in un unico DataFrame
    df_all = pd.concat([df_train], ignore_index=True)
    X_all, y_all = df_all[temporal_columns].values, df_all['BLASTO NY'].values

    final_model = RocketClassifier(num_kernels=best_kernel, random_state=conf.seed_everything(seed), n_jobs=-1)
    final_model = train_model(final_model, X_all, y_all)

    # Valutazione finale sul test set
    test_metrics = evaluate_model(final_model, X_test, y_test)
    print("\n===== FINAL TEST RESULTS =====")
    for metric, value in test_metrics.items():
        if metric != "conf_matrix":
            print(f"{metric.capitalize()}: {value:.4f}")


    cm_path = os.path.join(output_dir_conf_mat, f"confusion_matrix_ROCKET_{days_to_consider}Days.png")
    save_confusion_matrix(conf_matrix=test_metrics["conf_matrix"], filename=cm_path, num_kernels=best_kernel)

    best_model_path = os.path.join(output_model_dir, f"best_rocket_model_{days_to_consider}Days_{best_kernel}.pth")
    torch.save(final_model, best_model_path)
    print(f'Model saved at: {best_model_path}')
    

if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(lambda: main(), number=1)
    print(f"Tempo impiegato per l'esecuzione di Rocket con vari kernel:", execution_time, "secondi")