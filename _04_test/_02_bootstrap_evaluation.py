import os
import sys
import warnings
import torch
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from scipy.stats import shapiro, probplot
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection

warnings.filterwarnings("ignore", category=UserWarning)

# Path configuration
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

# Import project modules
from config import Config_03_train as conf
from _03_train._c_ConvTranUtils import CustomDataset
import _04_test._myFunctions as _myFunctions

##########################################
# Helper functions for bootstrap and stats
##########################################
def load_and_prepare_test_data(model_type, days_val, base_test_csv_path):
    df_test = _myFunctions.load_test_data(days_val=days_val, base_test_csv_path=base_test_csv_path)
    if df_test is None:
        return None
    X, y = _myFunctions.prepare_data(model_type=model_type, df=df_test)
    
    if model_type == "LSTMFCN":
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=conf.batch_size_FCN, shuffle=False)
        return loader  # loader will be used for evaluation
    else:
        return (X, y)

# Funzione di bootstrap per ottenere tutte le metriche
def bootstrap_metrics(y_true, y_pred, y_prob, n_bootstraps=30, alpha=0.95, 
                      show_normality=False, undersampling_proportion=0.5, seed=2024):
    np.random.seed(seed=seed)
    bootstrapped_metrics = []
    metric_order = ["accuracy", "balanced_accuracy", "kappa", "brier", "f1"]

    for _ in range(n_bootstraps):
        indices = np.random.randint(0, len(y_true), int(len(y_true)*undersampling_proportion))
        if len(np.unique(y_true[indices])) < 2 or len(np.unique(y_pred[indices])) < 2:
            continue
        metrics = _myFunctions.calculate_metrics(y_true[indices], y_pred[indices], y_prob[indices])
        # Store metrics as ordered list of values
        bootstrapped_metrics.append([metrics[key] for key in metric_order])

    bootstrapped_metrics = np.array(bootstrapped_metrics)

    # Test di normalità per ogni metrica
    for i, metric in enumerate(["Accuracy", "Balanced Accuracy", "Kappa", "Brier", "F1"]):
        stat, p_value = shapiro(bootstrapped_metrics[:, i])
        print(f"Test di Shapiro-Wilk per {metric}: stat={stat:.4f}, p-value={p_value:.4f}")

        if show_normality and (p_value < 0.05):
            plt.figure(figsize=(8, 5))
            sns.histplot(bootstrapped_metrics[:, i], kde=True, bins=50)
            plt.title(f"Distribuzione bootstrap - {metric}")
            plt.show()

            plt.figure(figsize=(6, 5))
            probplot(bootstrapped_metrics[:, i], dist="norm", plot=plt)
            plt.title(f"QQ-plot - {metric}")
            plt.show()

    # Calcolo delle statistiche riassuntive
    mean = np.mean(bootstrapped_metrics, axis=0)
    std = np.std(bootstrapped_metrics, axis=0)
    lower = np.percentile(bootstrapped_metrics, (1 - alpha) / 2 * 100, axis=0)
    upper = np.percentile(bootstrapped_metrics, (1 + alpha) / 2 * 100, axis=0)
    
    return mean, std, lower, upper, bootstrapped_metrics

def test_model_wrapper(model_type, model_info, test_data, device):
    """
    Returns y_true, y_pred, y_prob based on model type.
      - For ROCKET, test_data is (X, y) and uses _myFunctions.test_model_ROCKET.
      - For LSTMFCN and ConvTran, test_data is either a DataLoader (for LSTMFCN) or tuple (X, y) (for ConvTran).
    """
    if model_type == "ROCKET":
        X, y = test_data
        y_pred, y_prob = _myFunctions.test_model_ROCKET(model_info=model_info, X=X)
        return y, y_pred, y_prob
    
    elif model_type == "LSTMFCN":
        # test_data is a DataLoader
        # Use our custom test_model defined here
        model = model_info["model"]
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in test_data:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                pred = torch.argmax(output, dim=1)
                prob = torch.softmax(output, dim=1)[:, 1]
                y_true.extend(y_batch.cpu().numpy().flatten())
                y_pred.extend(pred.cpu().numpy().flatten())
                y_prob.extend(prob.cpu().numpy().flatten())
        return np.array(y_true), np.array(y_pred), np.array(y_prob)
    
    elif model_type == "ConvTran":
        X, y = test_data
        # Prepare data using CustomDataset
        dataset = CustomDataset(X.reshape(X.shape[0], 1, -1), y)
        loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False)
        model = model_info["model"]
        model.eval()
        all_pred, all_prob = [], []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                prob = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                all_pred.extend((prob >= model_info["threshold"]).astype(int))
                all_prob.extend(prob)
        return y, np.array(all_pred), np.array(all_prob)
    
    else:
        print(f"Unknown model type: {model_type}")
        return None, None, None

def compute_effect_size(bootstrap1, bootstrap2):
    """
    Compute Cohen's d effect size and perform a t-test.
    bootstrap1, bootstrap2: arrays of bootstrap metric values.
    Returns: d (Cohen's d) and p-value from t-test.
    """
    # Compute Cohen's d
    mean1, mean2 = np.mean(bootstrap1), np.mean(bootstrap2)
    std1, std2 = np.std(bootstrap1), np.std(bootstrap2)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
    # t-test
    t_stat, p_val = stats.ttest_rel(bootstrap1, bootstrap2, alternative="two-sided")
    return d, p_val

def interpret_cohen_d(d: float) -> str:
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif 0.2 <= abs_d < 0.5:
        return "Small"
    elif 0.5 <= abs_d < 0.8:
        return "Medium"
    else:
        return "Large"

def interpret_p_value(p: float) -> str:
    """Interpret statistical significance"""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "NS"

##########################################
# Main bootstrap evaluation code
##########################################
def boostrap_evaluation(base_path=os.path.join(current_dir, "final_test_bootstrap_metrics"),
         days=[1, 3, 5, 7],
         models_list=['ROCKET', 'LSTMFCN', 'ConvTran'],
         base_models_path=current_dir,
         base_test_csv_path=parent_dir):
    
    os.makedirs(base_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_results = []
    bootstrap_results = {}  # key: (day, model, metric) -> bootstrap samples
    
    # Loop over days and models
    for day_val in days:
        for m_type in models_list:
            print(f"Processing {m_type} for {day_val} Days...")
            # Load test data
            test_data = load_and_prepare_test_data(m_type, day_val, base_test_csv_path)
            if test_data is None:
                continue

            model_info = _myFunctions.load_model_by_type(m_type, day_val, base_models_path, device, data=test_data)
            if model_info is None:
                continue

            # Get predictions/probabilities
            y_true, y_pred, y_prob = test_model_wrapper(model_type=m_type, model_info=model_info, test_data=test_data, device=device)
            # Compute bootstrap metrics (using _myFunctions.bootstrap_metrics)
            print(f"Bootstrap for {m_type} {day_val} Days:")
            mean, std, lower, upper, bootstrap_samples = bootstrap_metrics(y_true, y_pred, y_prob)
            
            metric_names = ["Accuracy", "Balanced Accuracy", "Cohen's Kappa", "Brier Score", "F1 Score"]
            for i, metric in enumerate(metric_names):
                summary_results.append({
                    "Days": day_val,
                    "Model": m_type,
                    "Metric": metric,
                    "Mean": round(mean[i], 4),
                    "Std Deviation": round(std[i], 4),
                    "Lower CI": round(lower[i], 4),
                    "Upper CI": round(upper[i], 4)
                })
                bootstrap_results[(day_val, m_type, metric)] = np.round(bootstrap_samples[:, i], 4)
            print("\n---------------------------\n")
    
    # Save summary results
    df_summary = pd.DataFrame(summary_results).round(4)
    summary_file = os.path.join(base_path, "results_summary_bootstrap_metrics.csv")
    df_summary.to_csv(summary_file, index=False)
    print(f"📁 Summary bootstrap metrics saved to: {summary_file}")

    # Save bootstrap results in wide format
    df_bootstrap = pd.DataFrame.from_dict(bootstrap_results, orient="index").round(4)
    df_bootstrap.index = pd.MultiIndex.from_tuples(df_bootstrap.index, names=["Days", "Model", "Metric"])
    df_bootstrap.columns = [f"Bootstrap_{i+1}" for i in range(df_bootstrap.shape[1])]
    bootstrap_file = os.path.join(base_path, "results_bootstrap_metrics.csv")
    df_bootstrap.to_csv(bootstrap_file)
    print(f"📁 All bootstrap iterations saved to: {bootstrap_file}")
    
    ##########################################
    # Pairwise comparisons & effect size tests
    ##########################################
    comparisons = []
    # Compare models on the same day (e.g., ROCKET vs LSTMFCN on day 3)
    for day_val in days:
        for metric in metric_names:
            for i in range(len(models_list)):
                for j in range(i+1, len(models_list)):
                    key1 = (day_val, models_list[i], metric)
                    key2 = (day_val, models_list[j], metric)
                    if key1 in bootstrap_results and key2 in bootstrap_results:
                        bs1 = bootstrap_results[key1]
                        bs2 = bootstrap_results[key2]
                        d, p_val = compute_effect_size(bs1, bs2)
                        comparisons.append({
                            "Comparison": f"{models_list[i]} vs {models_list[j]}",
                            "Days": day_val,
                            "Metric": metric,
                            "Cohen_d": round(d, 4),
                            "Effect Size": interpret_cohen_d(d),
                            "p_value": round(p_val, 4),
                            "Significance": interpret_p_value(p_val)
                            })
    # Compare same model across days (e.g., LSTMFCN day1 vs day3)
    for m_type in models_list:
        for metric in metric_names:
            for i in range(len(days)):
                for j in range(i+1, len(days)):
                    key1 = (days[i], m_type, metric)
                    key2 = (days[j], m_type, metric)
                    if key1 in bootstrap_results and key2 in bootstrap_results:
                        bs1 = bootstrap_results[key1]
                        bs2 = bootstrap_results[key2]
                        d, p_val = compute_effect_size(bs1, bs2)
                        comparisons.append({
                            "Comparison": f"{m_type} {days[i]}Days vs {days[j]}Days",
                            "Days": f"{days[i]} vs {days[j]}",
                            "Metric": metric,
                            "Cohen_d": round(d, 4),
                            "Effect Size": interpret_cohen_d(d),
                            "p_value": round(p_val, 4),
                            "Significance": interpret_p_value(p_val)
                            })
    df_comp = pd.DataFrame(comparisons)

    # Extract all p-values for correction
    p_values = df_comp['p_value'].values
    # Perform FDR correction
    _, pvals_corrected = fdrcorrection(p_values, alpha=0.05)

    # Add corrected values to DataFrame
    df_comp['p_value_fdr'] = pvals_corrected.round(4)
    df_comp['Significance_fdr'] = df_comp['p_value_fdr'].apply(interpret_p_value)

    # Reorder columns for clarity
    df_comp = df_comp[[
        'Comparison', 'Days', 'Metric', 'Cohen_d', 'Effect Size',
        'p_value', 'Significance', 'p_value_fdr', 'Significance_fdr'
    ]]

    comp_file = os.path.join(base_path, "pairwise_comparisons_bootstrap_metrics.csv")
    df_comp.to_csv(comp_file, index=False)
    print(f"📁 Pairwise comparisons saved to: {comp_file}")


if __name__ == "__main__":
    start_time = time.time()
    boostrap_evaluation(base_path=os.path.join(current_dir, "final_test_bootstrap_metrics"),
         days=[1,3],
         models_list=['ROCKET', 'LSTMFCN', 'ConvTran'],
         base_models_path=current_dir,
         base_test_csv_path=parent_dir)
    print("Total execution time:", time.time() - start_time, "seconds")
