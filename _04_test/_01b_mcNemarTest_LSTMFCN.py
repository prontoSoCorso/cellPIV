import sys
import os
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
from  _03_train._b_LSTMFCN_PyTorch import LSTMFCN
import _04_test.myFunctions as myFunctions

device = conf.device

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

    # Devo solo recuperare le y del modello, che sono le stesse indipendentemente dal numero di giorni considerato
    days_to_consider = 1
    # Ottieni i percorsi dal config
    train_path, val_path, test_path = conf.get_paths(days_to_consider)
    # Carico i dati
    df_test = myFunctions.load_data(test_path)
    test_data = myFunctions.prepare_LSTMFCN_data(df_test)
    test_loader = DataLoader(test_data, batch_size=conf.batch_size_FCN, shuffle=False)

    # Carico i modelli da confrontare
    model_1_name = "best_lstm_fcn_model_1Days.pth"
    model_2_name = "best_lstm_fcn_model_3Days.pth"
    model_1_path = os.path.join(current_dir, model_1_name)
    model_2_path = os.path.join(current_dir, model_2_name)

    model1 = LSTMFCN(
        lstm_size=conf.lstm_size_FCN,
        filter_sizes=conf.filter_sizes_FCN,
        kernel_sizes=conf.kernel_sizes_FCN,
        dropout=conf.dropout_FCN,
        num_layers=conf.num_layers_FCN
    ).to(device)
    state_dict1 = torch.load(model_1_path, map_location=device)
    model1.load_state_dict(state_dict1)
    model1.eval()
    
    model2 = LSTMFCN(
        lstm_size=conf.lstm_size_FCN,
        filter_sizes=conf.filter_sizes_FCN,
        kernel_sizes=conf.kernel_sizes_FCN,
        dropout=conf.dropout_FCN,
        num_layers=conf.num_layers_FCN
    ).to(device)
    state_dict2 = torch.load(model_2_path, map_location=device)
    model2.load_state_dict(state_dict2)
    model2.eval()

    y_true, y_pred_model1, y_pred_model2 = [], [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds1 = torch.argmax(model1(X), dim=1)
            preds2 = torch.argmax(model2(X), dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred_model1.extend(preds1.cpu().numpy())
            y_pred_model2.extend(preds2.cpu().numpy())

    # Eseguo il test di McNemar
    model_name_without_extension_1 = os.path.splitext(model_1_name)[0]
    model_name_without_extension_2 = os.path.splitext(model_2_name)[0]
    apply_mcnemar(np.array(y_true), np.array(y_pred_model1), np.array(y_pred_model2), model_name_without_extension_1, model_name_without_extension_2)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time:", time.time() - start_time, "seconds")