import sys
import os
import pandas as pd
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, confusion_matrix, f1_score
import joblib  # Per il salvataggio del modello
import timeit
import seaborn as sns
import matplotlib.pyplot as plt

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_04_test as conf
from config import paths_for_models

# Funzione per caricare i dati normalizzati da CSV
def load_test_data(test_path):
    return pd.read_csv(test_path)

# Funzione per valutare il modello con metriche estese
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  # Probabilità della classe positiva
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    brier = brier_score_loss(y, y_prob, pos_label=1)
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return accuracy, balanced_accuracy, kappa, brier, f1, cm

# Funzione per valutare il modello con metriche per lstmfcn
def evaluate_model_lstmfcn(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  # Probabilità della classe positiva
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    brier = brier_score_loss(y, y_prob, pos_label=1)
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return accuracy, balanced_accuracy, kappa, brier, f1, cm

# Funzione per salvare la matrice di confusione come immagine
def save_confusion_matrix(cm, filename):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()


# ======= Funzioni per ogni modello =======
def main_ROCKET():
    df = load_test_data(conf.test_path)
    X = df.iloc[:, 3:].values  # Le colonne da 3 in poi contengono la serie temporale
    y = df['BLASTO NY'].values  # Colonna target

    best_model_path = os.path.join(parent_dir, conf.test_dir, f"rocket_classifier_model_{conf.kernel}_kernels.pkl")
    best_model = joblib.load(best_model_path)
    test_metrics = evaluate_model(best_model, X, y)

    print(f'=====ROCKET RESULTS WITH {conf.kernel} KERNELS=====')
    print(f'Test Accuracy with {conf.kernel} kernels: {test_metrics[0]}')
    print(f'Test Balanced Accuracy with {conf.kernel} kernels: {test_metrics[1]}')
    print(f"Test Cohen's Kappa with {conf.kernel} kernels: {test_metrics[2]}")
    print(f'Test Brier Score Loss with {conf.kernel} kernels: {test_metrics[3]}')
    print(f'Test F1 Score with {conf.kernel} kernels: {test_metrics[4]}')

    save_confusion_matrix(test_metrics[5], f"confusion_matrix_rocket_{conf.kernel}_kernels.png")


def main_LSTM():
    df = load_test_data(conf.test_path)
    X = df.iloc[:, 3:].values
    y = df['BLASTO NY'].values

    best_model_path = os.path.join(parent_dir, paths_for_models.test_dir, "lstm_classifier_model1.pth")
    best_model = torch.load(best_model_path)
    test_metrics = evaluate_model(best_model, X, y)

    print(f'=====LSTM RESULTS=====')
    print(f'Test Accuracy: {test_metrics[0]}')
    print(f'Test Balanced Accuracy: {test_metrics[1]}')
    print(f"Test Cohen's Kappa: {test_metrics[2]}")
    print(f'Test Brier Score Loss: {test_metrics[3]}')
    print(f'Test F1 Score: {test_metrics[4]}')

    save_confusion_matrix(test_metrics[5], "confusion_matrix_lstm.png")


def main_HIVECOTE2():
    df = load_test_data(conf.test_path)
    X = df.iloc[:, 3:].values
    y = df['BLASTO NY'].values

    best_model_path = os.path.join(parent_dir, conf.test_dir, "hivecote2_model_best.pkl")
    best_model = joblib.load(best_model_path)
    test_metrics = evaluate_model(best_model, X, y)

    print(f'=====HIVECOTE2 RESULTS=====')
    print(f'Test Accuracy: {test_metrics[0]}')
    print(f'Test Balanced Accuracy: {test_metrics[1]}')
    print(f"Test Cohen's Kappa: {test_metrics[2]}")
    print(f'Test Brier Score Loss: {test_metrics[3]}')
    print(f'Test F1 Score: {test_metrics[4]}')

    save_confusion_matrix(test_metrics[5], "confusion_matrix_hivecote2.png")


def main_ConvTran():
    df = load_test_data(conf.test_path)
    X = df.iloc[:, 3:].values
    y = df['BLASTO NY'].values

    best_model_path = os.path.join(parent_dir, conf.test_dir, "convTran_classifier_model.pkl")
    best_model = torch.load(best_model_path)
    test_metrics = evaluate_model(best_model, X, y)

    print(f'=====ConvTran RESULTS=====')
    print(f'Test Accuracy: {test_metrics[0]}')
    print(f'Test Balanced Accuracy: {test_metrics[1]}')
    print(f"Test Cohen's Kappa: {test_metrics[2]}")
    print(f'Test Brier Score Loss: {test_metrics[3]}')
    print(f'Test F1 Score: {test_metrics[4]}')

    save_confusion_matrix(test_metrics[5], "confusion_matrix_hivecote2.png")

# Funzione principale con switch
def main():
    #model_name = input("Scegli il modello da valutare (ROCKET, LSTM, LSTMFCN, HIVECOTE2, ConvTran): ").upper()
    model_name = "ROCKET"

    if model_name == "ROCKET":
        main_ROCKET()
    elif model_name == "LSTM":
        main_LSTM()
    elif model_name == "HIVECOTE2":
        main_HIVECOTE2()
    elif model_name == "ConvTran":
        main_ConvTran()
    else:
        print("Modello non valido. Scegli tra: ROCKET, LSTM, LSTMFCN, HIVECOTE2, ConvTran.")
        main()


if __name__ == "__main__":
    execution_time = timeit.timeit(lambda: main(), number=1)
    print(f"Tempo impiegato per l'esecuzione del modello scelto: {execution_time} secondi")
