import sys
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import itertools
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp

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
metrics = ["Accuracy"]
#metrics = ["Accuracy", "Balanced Accuracy", "Cohen's Kappa", "Brier Score", "F1 Score"]

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

        '''
        # **Test t di Student per coppie di modelli**
        for m1, m2 in itertools.combinations(models, 2):
            t_stat, p_value_t = stats.ttest_rel(data[m1], data[m2])
            results.append({
                "Days": day, "Metric": metric, "Comparison": f"{m1} vs {m2}",
                "Test": "T-test", "Statistic": t_stat, "p-value": p_value_t
            })
            print(f"{m1} vs {m2} (T-test): T={t_stat:.4f}, p={p_value_t:.4e}")
        '''

        # **Test di Wilcoxon per coppie di modelli (alternativa non parametrica)**
        for m1, m2 in itertools.combinations(models, 2):
            w_stat, p_value_w = wilcoxon(data[m1], data[m2])
            results.append({
                "Days": day, "Metric": metric, "Comparison": f"{m1} vs {m2}",
                "Test": "Wilcoxon", "Statistic": w_stat, "p-value": p_value_w
            })
            print(f"{m1} vs {m2} (Wilcoxon): W={w_stat:.4f}, p={p_value_w:.4e}")

        '''
        # **ANOVA a misure ripetute**
        if len(models) == 3:
            stacked_data = pd.DataFrame({
                "Value": np.concatenate([data[m] for m in models]),
                "Model": np.repeat(models, len(data[models[0]])),
                "ID": np.tile(np.arange(len(data[models[0]])), len(models))
            })

            anova = AnovaRM(stacked_data, "Value", "ID", within=["Model"]).fit()
            p_value_anova = anova.anova_table["Pr > F"].values[0]
            print("\nANOVA Risultati:\n", anova.summary())

            results.append({
                "Days": day, "Metric": metric, "Test": "ANOVA", "Statistic": anova.anova_table["F Value"].values[0], 
                "p-value": p_value_anova
            })

            # **Test post-hoc di Tukey**
            if p_value_anova < 0.05:
                tukey = pairwise_tukeyhsd(stacked_data["Value"], stacked_data["Model"])
                print("\nTest post-hoc di Tukey:\n", tukey)
        '''
        '''
        # **Test di Friedman (alternativa non parametrica all'ANOVA)**
        if len(models) == 3:
            friedman_stat, p_value_friedman = friedmanchisquare(*[data[m] for m in models])
            results.append({
                "Days": day, "Metric": metric, "Test": "Friedman", "Statistic": friedman_stat, 
                "p-value": p_value_friedman
            })
            print(f"\nTest di Friedman: Chi2={friedman_stat:.4f}, p={p_value_friedman:.4e}")

            # **Test post-hoc di Conover (se Friedman è significativo)**
            if p_value_friedman < 0.05:
                df_friedman = pd.DataFrame(data)
                conover_results = sp.posthoc_conover(df_friedman, p_adjust="holm")
                print("\nTest post-hoc di Conover:\n", conover_results)

        '''

# Salva i risultati dei test statistici in un CSV
df_results = pd.DataFrame(results)
output_file = os.path.join(current_dir, "results_model_comparison.csv")
df_results.to_csv(output_file, index=False)
print(f"\n📁 Risultati dei test salvati in: {output_file}")
