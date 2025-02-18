import sys
import os
import pandas as pd
import torch
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import DataLoader
import time
import numpy as np

# Aggiungo il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf
from _03_train.ConvTranUtils import CustomDataset
from _99_ConvTranModel.model import model_factory
from _99_ConvTranModel.utils import load_model
import _04_test.myFunctions as myFunctions

device = conf.device

# Funzione per caricare i dati
def load_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# Funzione per eseguire il test di McNemar
def apply_mcnemar(y_true, y_pred_model1, y_pred_model2, model_1_name, model_2_name):
    # Corretti e errati per entrambi i modelli
    correct_model1 = (y_pred_model1 == y_true)
    correct_model2 = (y_pred_model2 == y_true)

    # Costruzione della matrice di contingenza
    A = np.sum(~correct_model1 & ~correct_model2) # Entrambi sbagliano
    B = np.sum(~correct_model1 & correct_model2)  # M1 sbaglia, M2 corretto 
    C = np.sum(correct_model1 & ~correct_model2)  # M1 corretto, M2 sbaglia
    D = np.sum(correct_model1 & correct_model2)   # Entrambi corretti
    contingency_table = [[A, B], [C, D]]  # A e D teoricamente non servono per McNemar

    result = mcnemar(contingency_table, exact=True)

    print(f"\nStatistiche McNemar: {result.statistic}, p-value: {result.pvalue}")

    # Salva la matrice come immagine con il risultato del test
    contingency_path = os.path.join(current_dir, f"contingency_matrix_{model_1_name}_{model_2_name}.png")
    myFunctions.save_contingency_matrix_with_mcnemar(contingency_table, contingency_path, model_1_name, model_2_name, result.pvalue)
    print(f"Matrice di contingenza salvata in: {contingency_path}")

    # Interpretazione
    alpha = 0.05
    if result.pvalue < alpha:
        print("La differenza tra i modelli è statisticamente significativa.")
    else:
        print("La differenza tra i modelli non è statisticamente significativa.")

    return result.pvalue

# Funzione principale
def main():
    # PER IL MODELLO 1
    # Specifica il numero di giorni desiderati
    days_to_consider = 1
    # Ottieni i percorsi dal config
    _, _, test_path = conf.get_paths(days_to_consider)
    # Carica il dataset di test
    print("Caricamento primi dati di test...")
    data_test = pd.read_csv(test_path)
    X_test = data_test.iloc[:, 3:].values.reshape(data_test.shape[0], 1, -1)  # Reshape per ConvTran
    y_test = data_test['BLASTO NY'].values  # Etichette
    test_dataset = CustomDataset(X_test, y_test)
    test_loader_1 = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)
    # Aggiungi numero di etichette uniche
    conf.num_labels = len(set(test_loader_1.dataset.labels))
    conf.Data_shape = (test_loader_1.dataset[0][0].shape[0], test_loader_1.dataset[0][0].shape[1])
    # Carica il modello ConvTran pre-addestrato
    print("Caricamento del primo modello ConvTran...")
    model1 = model_factory(conf)
    model_1_name = f"best_convTran_model_{days_to_consider}Days.pkl"
    model_1_path = os.path.join(current_dir, model_1_name)
    model1 = load_model(model1, model_1_path)

    # PER IL MODELLO 2
    # Specifica il numero di giorni desiderati
    days_to_consider = 3
    # Ottieni i percorsi dal config
    _, _, test_path = conf.get_paths(days_to_consider)
    # Carica il dataset di test
    print("Caricamento secondi dati di test...")
    data_test = pd.read_csv(test_path)
    X_test = data_test.iloc[:, 3:].values.reshape(data_test.shape[0], 1, -1)  # Reshape per ConvTran
    y_test = data_test['BLASTO NY'].values  # Etichette
    test_dataset = CustomDataset(X_test, y_test)
    test_loader_2 = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)
    # Aggiungi numero di etichette uniche
    conf.num_labels = len(set(test_loader_2.dataset.labels))
    conf.Data_shape = (test_loader_2.dataset[0][0].shape[0], test_loader_2.dataset[0][0].shape[1])
    # Carica il modello ConvTran pre-addestrato
    print("Caricamento del secondo modello ConvTran...")
    model2 = model_factory(conf)
    model_2_name = f"best_convTran_model_{days_to_consider}Days.pkl"
    model_2_path = os.path.join(current_dir, model_2_name)
    model2 = load_model(model2, model_2_path)

    model1.eval()
    model2.eval()

    y_true1, y_true2, y_pred_model1, y_pred_model2 = [], [], [], []

    with torch.no_grad():
        for X, y in test_loader_1:
            outputs1 = model1(X)
            preds1 = torch.argmax(outputs1, dim=1)

            y_true1.extend(y.cpu().numpy())
            y_pred_model1.extend(preds1.cpu().numpy())

    with torch.no_grad():
        for X, y in test_loader_2:
            outputs2 = model2(X)
            preds2 = torch.argmax(outputs2, dim=1)
or
    model_name_without_extension_2 = os.path.splitext(model_2_name)[0]
    apply_mcnemar(np.array(y_true1), np.array(y_pred_model1), np.array(y_pred_model2), model_name_without_extension_1, model_name_without_extension_2)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time:", time.time() - start_time, "seconds")
