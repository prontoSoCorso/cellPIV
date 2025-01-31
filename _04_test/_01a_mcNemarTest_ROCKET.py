import os
import torch
import time
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import sys

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

import _04_test.myFunctions as myFunctions

# Funzione per caricare i dati
def load_data(csv_file_path):
    return pd.read_csv(csv_file_path)


# Funzione per valutare il modello (ritorna predizioni e probabilità)
def test_model(model, X):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  # Probabilità della classe positiva
    return y_pred, y_prob


# Funzione per applicare il test di McNemar
def apply_mcnemar(y_true1, y_true2, y_pred_model_1, y_pred_model_2, model_1_name, model_2_name, output_dir):
    # Costruzione della matrice 2x2 per McNemar
    both_correct = sum((y_pred_model_1 == y_true1) & (y_pred_model_2 == y_true2))
    both_wrong = sum((y_pred_model_1 != y_true1) & (y_pred_model_2 != y_true2))
    model_1_correct_model_2_wrong = sum((y_pred_model_1 == y_true1) & (y_pred_model_2 != y_true2))
    model_2_correct_model_1_wrong = sum((y_pred_model_2 == y_true2) & (y_pred_model_1 != y_true1))
    
    contingency_table = [
        [both_wrong, model_1_correct_model_2_wrong],
        [model_2_correct_model_1_wrong, both_correct]
    ]

    print("\nMatrice di contingenza (McNemar):")
    print(contingency_table)

    # Test di McNemar
    result = mcnemar(contingency_table, exact=True)
    print(f"\nStatistiche McNemar: {result.statistic}, p-value: {result.pvalue}")

    # Salva la matrice come immagine con il risultato del test
    contingency_path = os.path.join(output_dir, f"contingency_matrix_{model_1_name}_{model_2_name}.png")
    myFunctions.save_contingency_matrix_with_mcnemar(contingency_table, contingency_path, model_1_name, model_2_name, result.pvalue)
    print(f"Matrice di contingenza salvata in: {contingency_path}")

    # Interpretazione
    alpha = 0.05
    if result.pvalue < alpha:
        print("La differenza tra i modelli è statisticamente significativa.")
    else:
        print("La differenza tra i modelli non è statisticamente significativa.")


# Script principale
def main():
    # Specifica i percorsi per i modelli salvati
    model_1_name = "best_rocket_model_1Days.pth"
    model_2_name = "best_rocket_model_3Days.pth"
    model_1_path = os.path.join(current_dir, model_1_name)
    model_2_path = os.path.join(current_dir, model_2_name)

    # Percorso ai dati di test (due file ma appaiati perché uno è a 1 giorno, l'altro a 3 giorni)
    test_csv_path1 = os.path.join(parent_dir, "Normalized_sum_mean_mag_1Days_test.csv")
    test_csv_path2 = os.path.join(parent_dir, "Normalized_sum_mean_mag_3Days_test.csv")

    # Directory di output per salvare l'immagine
    output_dir = current_dir

    # Carica i modelli
    model_1 = torch.load(model_1_path, weights_only=False)
    model_2 = torch.load(model_2_path, weights_only=False)

    # Carica i dati di test
    df_test1 = load_data(test_csv_path1)
    df_test2 = load_data(test_csv_path2)

    X_test1 = df_test1.iloc[:, 3:].values  # Le colonne da 3 in poi contengono la serie temporale
    y_test1 = df_test1['BLASTO NY'].values  # Colonna target

    X_test2 = df_test2.iloc[:, 3:].values  # Le colonne da 3 in poi contengono la serie temporale
    y_test2 = df_test2['BLASTO NY'].values  # Colonna target

    # Test sui due modelli
    y_pred_1, _ = test_model(model_1, X_test1)
    y_pred_2, _ = test_model(model_2, X_test2)

    # Applicazione del test di McNemar
    model_name_without_extension_1 = os.path.splitext(model_1_name)[0]
    model_name_without_extension_2 = os.path.splitext(model_2_name)[0]
    apply_mcnemar(y_test1, y_test2, y_pred_1, y_pred_2, model_name_without_extension_1, model_name_without_extension_2, output_dir)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time:", time - start_time, "seconds")

