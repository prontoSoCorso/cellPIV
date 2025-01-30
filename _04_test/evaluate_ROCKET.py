import sys
import os
import torch
import numpy as np
import pandas as pd

# Aggiungo il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

import _04_test.myFunctions as myFunctions

# Funzione per testare il modello e ottenere predizioni e probabilità
def test_model(model, X):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  # Probabilità della classe positiva
    return y_pred, y_prob

# Caricamento modello e dati per più giorni
def main():
    days = [1,3,5,7]
    risultati = []

    for days_val in days:
        model_name = f"best_rocket_model_{days_val}Days.pth"
        test_csv_path = f"Normalized_sum_mean_mag_{days_val}Days_test.csv"
        model_path = os.path.join(current_dir, model_name)

        if not os.path.exists(model_path):
            print(f"Modello non trovato: {model_path}, salto al prossimo.")
            continue
        if not os.path.exists(test_csv_path):
            print(f"File test non trovato: {test_csv_path}, salto al prossimo.")
            continue

        # Carica il modello
        model = torch.load(model_path, weights_only=False)

        # Carica i dati di test
        df_test = pd.read_csv(test_csv_path)
        X_test = df_test.iloc[:, 3:].values  # Caratteristiche dalla colonna 3 in poi
        y_test = df_test['BLASTO NY'].values  # Colonna target

        # Verifica presenza di NaN o inf in X_test
        if np.isnan(X_test).any():
            print(f"⚠️ Warning: Il dataset di test {test_csv_path} contiene NaN.")
        if np.isinf(X_test).any():
            print(f"⚠️ Warning: Il dataset di test {test_csv_path} contiene valori infiniti.")
        
        # Sostituzione dei valori problematici (opzionale)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

        # Ottenere predizioni e probabilità
        y_pred, y_prob = test_model(model, X_test)


        # Ottenere predizioni e probabilità
        y_pred, y_prob = test_model(model, X_test)

        # Calcolare metriche bootstrap
        mean, std, lower, upper = myFunctions.bootstrap_metrics(y_test, y_pred, y_prob)

        # Stampa risultati
        metric_names = ["Accuracy", "Balanced Accuracy", "Cohen's Kappa", "Brier Score", "F1 Score"]
        print(f"\nRisultati per {days_val} giorni:")
        for i, metric in enumerate(metric_names):
            print(f"{metric}: Media={mean[i]:.4f}, Std={std[i]:.4f}, IC=[{lower[i]:.4f}, {upper[i]:.4f}]")

        print("\n==============================\n")

        # Salvataggio dei risultati
        for i, metric in enumerate(metric_names):
            risultati.append({
                "Days": days_val,
                "Metric": metric,
                "Mean": mean[i],
                "Std Deviation": std[i],
                "Lower CI": lower[i],
                "Upper CI": upper[i]
            })

    # Creazione DataFrame e salvataggio su file
    df_risultati = pd.DataFrame(risultati)
    output_file = os.path.join(current_dir, "results_bootstrap_metrics_ROCKET.csv")
    df_risultati.to_csv(output_file, index=False)
    print(f"\nRisultati salvati in: {output_file}")

if __name__ == "__main__":
    main()
