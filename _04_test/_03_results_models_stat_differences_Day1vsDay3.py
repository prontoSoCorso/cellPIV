import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import time

# Aggiungo il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)


def main():
    # Caricamento dei file CSV con tutte le iterazioni bootstrap
    files = {
        "ConvTran": os.path.join(current_dir, "results_bootstrap_metrics_ConvTran.csv"),
        "ROCKET": os.path.join(current_dir, "results_bootstrap_metrics_ROCKET.csv"),
        "LSTMFCN": os.path.join(current_dir, "results_bootstrap_metrics_LSTMFCN.csv"),
    }

    # Legge tutti i file in un dizionario di DataFrame
    df_models = {model: pd.read_csv(file, index_col=[0, 1]) for model, file in files.items()}

    # Definizione delle metriche e giorni da confrontare
    metrics = ["Accuracy", "F1 Score"]
    days = [1, 3]

    # Funzione per calcolare l'effect size r per Mann-Whitney
    def mann_whitney_u_test_with_effect_size(group1, group2):
        u_stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
        
        # Calcolo della statistica Z
        n1, n2 = len(group1), len(group2)
        mean_rank = (n1 + n2 + 1) / 2.0
        std_rank = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12.0)
        z_stat = (u_stat - (n1 * n2 / 2.0)) / std_rank
        
        # Effect size r
        effect_size_r = abs(z_stat) / np.sqrt(n1 + n2)
        
        # Interpretazione dell'effect size
        if effect_size_r < 0.1:
            effect_label = "Trascurabile"
        elif effect_size_r < 0.3:
            effect_label = "Piccolo"
        elif effect_size_r < 0.5:
            effect_label = "Medio"
        else:
            effect_label = "Grande"
        
        return u_stat, p_value, effect_size_r, effect_label

    # Risultati per i test statistici
    results = []

    for model, df in df_models.items():
        for metric in metrics:
            print(f"\n### {model}: {metric} (Day 1 vs Day 3) ###")

            # Estraggo i dati per day 1 e day 3
            if (1, metric) in df.index and (3, metric) in df.index:
                data_day1 = df.loc[(1, metric)].values.flatten()
                data_day3 = df.loc[(3, metric)].values.flatten()

                # Test Mann-Whitney U
                u_stat, p_value_u, effect_size, effect_label = mann_whitney_u_test_with_effect_size(data_day1, data_day3)
                results.append({
                    "Model": model, "Metric": metric, "Comparison": "Day 1 vs Day 3",
                    "Test": "Mann-Whitney U", "Statistic": u_stat, "p-value": p_value_u,
                    "Effect Size r": effect_size, "Effect Label": effect_label
                })
                print(f"{model} (Day 1 vs Day 3): U={u_stat:.4f}, p={p_value_u:.4e}, r={effect_size:.4f} ({effect_label})")

    # Salva i risultati in un CSV
    df_results = pd.DataFrame(results)
    output_file = os.path.join(current_dir, "results_model_comparison_Day1vsDay3.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\n📁 Risultati dei test salvati in: {output_file}")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Execution time:", time.time() - start_time, "seconds")
