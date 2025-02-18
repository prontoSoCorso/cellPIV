import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import time

# Aggiungo il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

# Importa funzioni e modello ConvTran dal progetto
from config import Config_03_train as conf
from _99_ConvTranModel.model import model_factory
from _99_ConvTranModel.utils import load_model
from _03_train.ConvTranUtils import CustomDataset
import _04_test.myFunctions as myFunctions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funzione per testare il modello ConvTran
def test_model(model, test_loader):
    model.eval()
    y_pred, y_prob, y_true = [], [], []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)

            pred = torch.argmax(output, dim=1)
            prob = torch.softmax(output, dim=1)[:, 1]
            # si considera come probabilità finale quella associata alla classe 1 --> faccio [:, 1]
            
            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(pred.cpu().numpy().flatten())
            y_prob.extend(prob.cpu().numpy().flatten())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)

# Caricamento modello e dati per più giorni
def main():
    days = [1, 3, 5, 7]
    risultati_summary = []
    risultati_bootstrap = {}

    for days_val in days:
        model_name = f"best_convTran_model_{days_val}Days.pkl"
        _, _, test_path = conf.get_paths(days_val)
        model_path = os.path.join(current_dir, model_name)

        if not os.path.exists(model_path):
            print(f"Modello non trovato: {model_path}, salto al prossimo.")
            continue
        if not os.path.exists(test_path):
            print(f"File test non trovato: {test_path}, salto al prossimo.")
            continue

        # Carica i dati di test
        df_test = pd.read_csv(test_path)
        X_test = df_test.iloc[:, 3:].values.reshape(df_test.shape[0], 1, -1)
        y_test = df_test['BLASTO NY'].values
        test_dataset = CustomDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)

        conf.num_labels = len(set(test_loader.dataset.labels))
        conf.Data_shape = (test_loader.dataset[0][0].shape[0], test_loader.dataset[0][0].shape[1])

        # Carica il modello ConvTran
        model = model_factory(conf).to(device)
        model = load_model(model, model_path)

        # Ottenere predizioni e probabilità
        y_true, y_pred, y_prob = test_model(model, test_loader)

        # Calcolare metriche bootstrap
        print(f"Test di normalità per {days_val} giorni:")
        mean, std, lower, upper, bootstrap_samples = myFunctions.bootstrap_metrics(y_true, y_pred, y_prob)

        # Definizione delle metriche
        metric_names = ["Accuracy", "Balanced Accuracy", "Cohen's Kappa", "Brier Score", "F1 Score"]

        print(f"\nRisultati per {days_val} giorni:")
        for i, metric in enumerate(metric_names):
            print(f"{metric}: Media={mean[i]:.4f}, Std={std[i]:.4f}, IC=[{lower[i]:.4f}, {upper[i]:.4f}]")

        print("\n==============================\n")

        # Salvataggio del riepilogo delle metriche
        for i, metric in enumerate(metric_names):
            risultati_summary.append({
                "Days": days_val,
                "Metric": metric,
                "Mean": mean[i],
                "Std Deviation": std[i],
                "Lower CI": lower[i],
                "Upper CI": upper[i]
            })

            # Salvataggio delle metriche bootstrap in formato orizzontale
            key = (days_val, metric)
            risultati_bootstrap[key] = bootstrap_samples[:, i]  # Salva tutte le iterazioni

    # Salvataggio riepilogo metriche
    df_summary = pd.DataFrame(risultati_summary)
    summary_file = os.path.join(current_dir, "results_summary_bootstrap_metrics_ConvTran.csv")
    df_summary.to_csv(summary_file, index=False)
    print(f"\n📁 Riepilogo delle metriche salvato in: {summary_file}")

    # Creazione DataFrame bootstrap (wide format)
    df_bootstrap = pd.DataFrame.from_dict(risultati_bootstrap, orient="index")
    df_bootstrap.index = pd.MultiIndex.from_tuples(df_bootstrap.index, names=["Days", "Metric"])
    df_bootstrap.columns = [f"Bootstrap_{i+1}" for i in range(df_bootstrap.shape[1])]

    # Salvataggio metriche bootstrap
    bootstrap_file = os.path.join(current_dir, "results_bootstrap_metrics_ConvTran.csv")
    df_bootstrap.to_csv(bootstrap_file)
    print(f"\n📁 Metriche di tutte le estrazioni bootstrap salvate in: {bootstrap_file}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time:", time.time() - start_time, "seconds")
