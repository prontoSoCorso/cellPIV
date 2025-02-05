import sys
import os
import pandas as pd
import numpy as np
import itertools
from scipy.stats import mannwhitneyu

# Aggiungo il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

# Caricamento dei file CSV con tutte le iterazioni bootstrap
files = {
    "ConvTran": os.path.join(current_dir, "results_bootstrap_metrics_ConvTran.csv"),
    "ROCKET": os.path.join(current_dir, "results_bootstrap_metrics_ROCKET.csv"),
    "LSTMFCN": os.path.join(current_dir, "results_bootstrap_metrics_LSTMFCN.csv"),
}

# Legge tutti i file in un dizionario di DataFrame
df_models = {model: pd.read_csv(file, index_col=[0, 1]) for model, file in files.items()}

# Lista dei giorni e delle metriche da confrontare
days = [1, 3, 5, 7]
metrics = ["Accuracy", "F1 Score"]
#metrics = ["Accuracy", "Balanced Accuracy", "Cohen's Kappa", "Brier Score", "F1 Score"]

# Funzione per calcolare l'effect size r per Mann-Whitney
def mann_whitney_u_test_with_effect_size(group1, group2):
    u_stat, p_value = mannwhitneyu(group1, group2, alternative="greater")
    
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

for day in days:
    for metric in metrics:
        print(f"\n### Giorno {day} - {metric} ###")
        
        # Estraggo tutti i valori bootstrap per ciascun modello
        data = {}
        for model, df in df_models.items():
            subset = df.loc[(day, metric)]
            if not subset.empty:
                data[model] = subset.values.flatten()

        models = list(data.keys())

        # **Test di Mann-Whitney U con effect size**
        for m1, m2 in itertools.combinations(models, 2):
            u_stat, p_value_u, effect_size, effect_label = mann_whitney_u_test_with_effect_size(data[m1], data[m2])
            results.append({
                "Days": day, "Metric": metric, "Comparison": f"{m1} vs {m2}",
                "Test": "Mann-Whitney U", "Statistic": u_stat, "p-value": p_value_u,
                "Effect Size r": effect_size, "Effect Label": effect_label
            })
            print(f"{m1} vs {m2} (Mann-Whitney U): U={u_stat:.4f}, p={p_value_u:.4e}, r={effect_size:.4f} ({effect_label})")


# Salva i risultati dei test statistici in un CSV
df_results = pd.DataFrame(results)
output_file = os.path.join(current_dir, "results_model_comparison.csv")
df_results.to_csv(output_file, index=False)
print(f"\n📁 Risultati dei test salvati in: {output_file}")
