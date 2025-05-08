import seaborn as sns
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import sys
import joblib
from typing import Tuple


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
from _03_train._b_LSTMFCN import LSTMFCN
from _99_ConvTranModel.model import model_factory


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


# To plot different confusion matrices in a single image
def plot_summary_confusion_matrices(model_name, cm_data, day, output_dir):
    """
    Plot a single PNG with all confusion matrices for the subgroups of a model.
    cm_data: list of tuples (group, cm)
    """
    n = len(cm_data)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]
    for i, (group, cm) in enumerate(cm_data):
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(f"{group}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    # Remove unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"{model_name} Confusion Matrices ({day} Days)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = os.path.join(output_dir, f"{model_name}_conf_matrix_{day}Days.png")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


# To plot different roc curves in a single image
def plot_summary_roc_curves(model_name, roc_data, day, output_dir):
    """
    Plot a single ROC plot per model per day.
    roc_data: list of tuples (group, fpr, tpr, roc_auc)
    """
    plt.figure(figsize=(8, 6))
    for group, fpr, tpr, roc_auc in roc_data:
        plt.plot(fpr, tpr, lw=2, label=f"{group} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"{model_name} ROC Curve ({day} Days)", fontsize=14)
    plt.legend(loc="lower right")
    output_file = os.path.join(output_dir, f"{model_name}_ROC_{day}Days.png")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


# Richiamo questa così da essere coerente nei vari script
def instantiate_LSTMFCN(checkpoint, device):
    params = checkpoint.get('params', {})

    model = LSTMFCN(
        lstm_size=params.get('lstm_size'),
        filter_sizes=tuple(map(int, params.get('filter_sizes', '').split(','))),
        kernel_sizes=tuple(map(int, params.get('kernel_sizes', '').split(','))),
        dropout=params.get('dropout'),
        num_layers=params.get('num_layers')
        ).to(device)
    return model


# Funzione per importare dati di test
def load_test_data(days_val):
    train_path, val_path, test_path = conf.get_paths(days_val)
    if not os.path.exists(test_path):
        print(f"Test file not found: {test_path}")
        return None, None
    df_test = pd.read_csv(test_path)
    return df_test


# Funzione per preparare i dati
def prepare_data(model_type: str, df=None, path_csv=None) -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare test data with model-specific preprocessing"""
    if path_csv:
        df = pd.read_csv(path_csv)
    if df is None:
        raise ValueError("Either 'df' or 'path_csv' must be provided.")
        
    temporal_cols = [c for c in df.columns if c.startswith('value_')]
    X = df[temporal_cols].values
    y = df['BLASTO NY'].values

    if model_type == "LSTMFCN":
        X = X.unsqueeze(-1) if torch.is_tensor(X) else np.expand_dims(X, -1)
    return X, y


# Per definire valori di convTran per fare il model_factory(config)
def value_for_config_convTran(X):
    num_labels = 2

    if X.ndim == 2:
        data_shape = (1, X.shape[1])
    elif X.ndim == 3:
        data_shape = (X.shape[0], X.shape[2])
    else:
        raise ValueError("X deve avere 2 o 3 dimensioni.")

    return {
        "num_labels": num_labels, 
        "data_shape": data_shape
        }


# Funzione per testare il modello e ottenere predizioni e probabilità
def test_model_ROCKET(model_info, X):
    # Trasformazione features
    X_3d = X[:, np.newaxis, :]  # Aggiunge dimensione canale
    X_features = model_info['transformer'].transform(X_3d)

    # Predizioni con soglia ottimale
    y_prob = model_info['model'].predict_proba(X_features)[:, 1]
    y_pred = (y_prob >= model_info['threshold']).astype(int)

    return y_pred, y_prob


# Different type of loading specific for each model
def load_model_by_type(model_type: str, days: int, base_models_path: str, device: torch.device, data=None):
    """
    Loads the specified model type for the given day.
      - ROCKET: uses joblib.load
      - LSTMFCN: uses torch.load and instantiates the LSTMFCN model
      - ConvTran: creates model via model_factory(conf) and then loads weights
    """
    model_file = f"best_{model_type.lower()}_model_{days}Days.{'pth' if model_type == 'LSTMFCN' else 'pkl' if model_type == 'ConvTran' else 'joblib'}"
    model_path = os.path.join(base_models_path, model_file)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_type.lower()} not found: {model_path}")

    if model_type == "ROCKET":
        rocket = joblib.load(model_path)
        # For ROCKET, we return a dict with classifier, transformer and threshold
        return {
            "model": rocket['classifier'],
            "transformer": rocket['rocket'],
            "threshold": rocket['final_threshold']
            }
    
    elif model_type == "LSTMFCN":
        lstm_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = instantiate_LSTMFCN(checkpoint=lstm_checkpoint, device=device)
        model.load_state_dict(lstm_checkpoint['model_state_dict'])
        return {
            "model": model,
            "threshold": lstm_checkpoint.get('best_threshold', 0.5),
            "batch_size": lstm_checkpoint['params']['batch_size']
            }
    
    elif model_type == "ConvTran":
        conv_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Restore parameters to config
        saved_config = conv_checkpoint['config']
        for key, value in saved_config.items():
            setattr(conf, key, value)

        # Rebuild model with original config and save best threshold
        model = model_factory(conf).to(device)
        model.load_state_dict(conv_checkpoint['model_state_dict'])
        return {
            "model": model,
            "threshold": conv_checkpoint.get('best_threshold', 0.5)
            }
    
    else:
        print(f"Unknown model type: {model_type}")
        return None

