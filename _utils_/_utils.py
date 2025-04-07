import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                             cohen_kappa_score, brier_score_loss, 
                             confusion_matrix, f1_score,
                             roc_curve, auc, 
                             precision_score, recall_score, 
                             matthews_corrcoef)

# Funzione per caricare i dati normalizzati da CSV
def load_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# Configure logging
def config_logging(log_dir, log_filename):
    os.makedirs(log_dir, exist_ok=True)
    complete_filename = os.path.join(log_dir, log_filename)
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(complete_filename)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Add handlers
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


# Funzioni per salvare la matrice di confusione come immagine
def save_confusion_matrix(conf_matrix, filename, model_name, num_kernels=0):
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if num_kernels>0:
        plt.title(f"Confusion Matrix - {model_name} - {num_kernels} Kernels")
    else: 
        plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(filename)
    plt.close()


# Plot ROC Curve
def plot_roc_curve(fpr, tpr, roc_auc, filename):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()



# Funzione per calcolare le metriche
def calculate_metrics(y_true, y_pred, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc,
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "conf_matrix": confusion_matrix(y_true, y_pred),
        "fpr": fpr,
        "tpr": tpr
    }

    decimals = 4
    for key, value in metrics.items():
        if isinstance(value, float):
            metrics[key] = round(value, decimals)
        elif isinstance(value, np.ndarray): # Gestisce array NumPy
            metrics[key] = np.round(value, decimals) # Arrotonda gli elementi dell'array

    return metrics
