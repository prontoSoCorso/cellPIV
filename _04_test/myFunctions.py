import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, probplot
import numpy as np

# Funzione per salvare la matrice di contingenza con il risultato di McNemar
def save_contingency_matrix_with_mcnemar(matrix, filename, model_1_name, model_2_name, p_value):
    plt.figure(figsize=(6, 6))  # Riduci la dimensione della matrice
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Model 2 Wrong", "Model 2 Correct"],
                yticklabels=["Model 1 Wrong", "Model 1 Correct"],
                annot_kws={"fontsize": 12})  # Aumenta dimensione dei numeri annotati

    # Aumenta la dimensione dei font per gli assi
    plt.xlabel(f"{model_1_name} Predictions", fontsize=15, labelpad=20)
    plt.ylabel(f"{model_2_name} Predictions", fontsize=15, labelpad=20)

    # Aumenta dimensione xticks e yticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title("\nContingency Matrix with McNemar Test\n", fontsize=16)  # Aumenta titolo

    # Mostra il p-value con un carattere più grande
    plt.figtext(0.5, -0.1, f"McNemar Test p-value: {p_value:.2e}",
                ha="center", fontsize=14, wrap=True, bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5})
    
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# Funzione per caricare i dati
def load_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# Funzione per preparare i dati
def prepare_LSTMFCN_data(df):
    X = torch.tensor(df.iloc[:, 3:].values, dtype=torch.float32).unsqueeze(-1)  # Aggiungo dimensione per il canale
    y = torch.tensor(df['BLASTO NY'].values, dtype=torch.long)
    return TensorDataset(X, y)

# Funzione per testare il modello e ottenere predizioni e probabilità
def test_model_ROCKET(model, X):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  # Probabilità della classe positiva
    return y_pred, y_prob


# Funzione per calcolare le metriche
def calculate_metrics(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob, pos_label=1)
    f1 = f1_score(y_true, y_pred, zero_division="warn")
    return np.array([accuracy, balanced_accuracy, kappa, brier, f1])


def calculate_metrics_with_spec(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob, pos_label=1)

    # Estraggo TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calcolo Specificità (TNR), gestendo la divisione per zero
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  
    
    # Calcolo F1-score manualmente per gestire la divisione per zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return np.array([accuracy, balanced_accuracy, kappa, brier, f1, specificity])


# Funzione di bootstrap per ottenere tutte le metriche
def bootstrap_metrics(y_true, y_pred, y_prob, n_bootstraps=50, alpha=0.95, show_normality=False):
    bootstrapped_metrics = []

    for _ in range(n_bootstraps):
        undersampling_proportion = 0.8
        indices = np.random.randint(0, len(y_true), int(len(y_true)*undersampling_proportion))
        if len(np.unique(y_true[indices])) < 2 or len(np.unique(y_pred[indices])) < 2:
            continue
        metrics = calculate_metrics(y_true[indices], y_pred[indices], y_prob[indices])
        bootstrapped_metrics.append(metrics)

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

