import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                             cohen_kappa_score, brier_score_loss, 
                             confusion_matrix, f1_score,
                             roc_curve, auc, 
                             precision_score, recall_score, 
                             matthews_corrcoef)

from _03_train._b_LSTMFCN import TimeSeriesClassifier

# Funzione per caricare i dati normalizzati da CSV
def load_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# Funzione per prendere solo le colonne che indicano le ore dei tempi di acquisizione
def detect_time_columns(df):
    # riconosce colonne che finiscono con 'h' come '0.00h', '131.75h', ecc.
    time_cols = [c for c in df.columns if re.match(r'^\d+(\.\d+)?h$', str(c))]
    # fallback: se non trovi nulla, prendi tutte le colonne dopo 'maternal age' se presente
    if len(time_cols) == 0 and 'maternal age' in df.columns:
        start_idx = list(df.columns).index('maternal age') + 1
        time_cols = list(df.columns[start_idx:])
    return time_cols


def build_data_dict(df_train=None, df_val=None, df_test=None):
    data_dict = {}
    # Rileva le colonne temporali
    time_cols = detect_time_columns(df_train) if df_train is not None else (
                detect_time_columns(df_val) if df_val is not None else (
                    detect_time_columns(df_test) if df_test is not None else []))
    if len(time_cols) == 0:
        raise ValueError("Non sono state trovate colonne temporali nei DataFrame forniti.")

    # Costruisci X con shape (N, C, T)
    X_train, y_train, X_val, y_val, X_test, y_test = None, None, None, None, None, None
    if df_train is not None:
        X_train = df_train[time_cols].values[:, np.newaxis, :]
        y_train = df_train['BLASTO NY'].values.astype(int)
        data_dict["X_train"] = X_train
        data_dict["y_train"] = y_train
    if df_val is not None:
        X_val = df_val[time_cols].values[:, np.newaxis, :]
        y_val = df_val['BLASTO NY'].values.astype(int)
        data_dict["X_val"] = X_val
        data_dict["y_val"] = y_val
    if df_test is not None:
        X_test = df_test[time_cols].values[:, np.newaxis, :]
        y_test = df_test['BLASTO NY'].values.astype(int)
        data_dict["X_test"] = X_test
        data_dict["y_test"] = y_test

    return data_dict


# Funzione per validare il dizionario dei dati
def _check_data_dict(data: dict, require_test: bool, only_check_test=False):
    required = []
    if not only_check_test:
        required = ["X_train", "y_train", "X_val", "y_val"]
    
    if require_test:
        required += ["X_test", "y_test"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Mancano le chiavi nel data dict: {missing}")

    # Converti in numpy se necessario e valida forme base
    def _to_np(x):
        return x.values if hasattr(x, "values") else x
    
    X_train = _to_np(data["X_train"]) if "X_train" in data else None
    y_train = _to_np(data["y_train"]) if "y_train" in data else None
    X_val = _to_np(data["X_val"]) if "X_val" in data else None
    y_val = _to_np(data["y_val"]) if "y_val" in data else None

    X_test = _to_np(data["X_test"]) if "X_test" in data else None
    y_test = _to_np(data["y_test"]) if "y_test" in data else None

    # Check dimensionalità (N, 1, T)
    if not only_check_test:
        for name, X in [("X_train", X_train), ("X_val", X_val)]:
            if X.ndim != 3 or X.shape[1] != 1:
                raise ValueError(f"{name} must have shape (N, 1, T). Found: {X.shape}")

        for name, y in [("y_train", y_train), ("y_val", y_val)]:
            if y.ndim != 1:
                raise ValueError(f"{name} must be 1D. Found: {y.shape}")

    if X_test is not None:
        if X_test.ndim != 3 or X_test.shape[1] != 1:
            raise ValueError(f"X_test must have shape (N, 1, T). Found: {X_test.shape}")
    if y_test is not None and y_test.ndim != 1:
        raise ValueError(f"y_test must be 1D. Found: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def _check_test_data(data: dict):
    require_test = "X_test" in data and "y_test" in data
    if require_test:
        if data["X_test"] is None or data["y_test"] is None:
            raise ValueError("If 'X_test' or 'y_test' are present, they must not be None.")
    else:
        raise ValueError("Keys 'X_test' and 'y_test' must be present in data dict to perform testing.")
        
    # Check dimensionalità (N, 1, T)
    if require_test:
        X_test = data["X_test"]
        y_test = data["y_test"]
        if X_test.ndim != 3 or X_test.shape[1] != 1:
            raise ValueError(f"X_test must have shape (N, 1, T). Found: {X_test.shape}")
        if y_test.ndim != 1:
            raise ValueError(f"y_test must be 1D. Found: {y_test.shape}")
    
    return require_test


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
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, 
                xticklabels=["Class 0 (no_blasto)", "Class 1 (blasto)"], 
                yticklabels=["Class 0 (no_blasto)", "Class 1 (blasto)"],
                annot_kws={"size": 16}  # Increase font size for numbers
                )
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


def save_results(metrics, output_dir, model_name, day, save_plots=True):
    logging.info("\n===== FINAL TEST RESULTS =====")
    for metric, value in metrics.items():
        if metric not in ['conf_matrix', 'fpr', 'tpr']:
            logging.info(f"{metric.capitalize()}: {value:.4f}")
    
    if save_plots:
        complete_output_dir = os.path.join(output_dir, f"day{day}")
        os.makedirs(complete_output_dir, exist_ok=True)
        conf_matrix_filename=os.path.join(complete_output_dir,f'confusion_matrix_{model_name}_{day}Days.png')
        save_confusion_matrix(conf_matrix=metrics['conf_matrix'], 
                                    filename=conf_matrix_filename, 
                                    model_name=model_name)
        plot_roc_curve(fpr=metrics['fpr'], tpr=metrics['tpr'],
                             roc_auc=metrics['roc_auc'], 
                             filename=conf_matrix_filename.replace('confusion_matrix', 'roc'))
        


import torch
import inspect
import torch.nn as nn

def _strip_module_prefix(state_dict):
    """Rimuove 'module.' se presente (salvataggi da DataParallel)."""
    new_state = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("module."):
            new_key = k[len("module."):]
        new_state[new_key] = v
    return new_state

def load_lstmfcn_from_checkpoint(checkpoint_path: str, model_class, device=None, strict=True):
    """
    Carica un checkpoint salvato con la struttura suggerita e ricostruisce automaticamente
    il modello chiamando model_class(**params_filtered).
    
    Args:
      - checkpoint_path: percorso .pth
      - model_class: la classe del modello (es. TimeSeriesClassifier)
      - device: torch.device o stringa; se None rileva automaticamente
      - strict: passato a load_state_dict
    
    Returns:
      - model (in eval mode, sul device),
      - best_threshold (float, fallback 0.5 se non presente),
      - saved_params (dict, vuoto se non presenti)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_loc = device if isinstance(device, torch.device) else torch.device(device)

    loaded = torch.load(checkpoint_path, map_location=map_loc, weights_only=False)

    # Caso: checkpoint è direttamente un modello nn.Module serializzato
    if isinstance(loaded, nn.Module):
        model = loaded.to(device)
        model.eval()
        return model, 0.5, {}

    # Se è dict, cerchiamo i campi che ci servono
    if isinstance(loaded, dict):
        # possibili nomi: 'model_state_dict' / 'state_dict' / 'model' ecc.
        state_dict = loaded.get("model_state_dict") or loaded.get("state_dict") or None
        saved_params = loaded.get("params", {}) or {}
        best_threshold = float(loaded.get("best_threshold", loaded.get("final_threshold", 0.5)))
    else:
        raise RuntimeError(f"Formato checkpoint {type(loaded)} non supportato. Aspettavo dict o nn.Module.")

    # Se non c'è state_dict ma il dict contiene un modello salvato a oggetto (raramente)
    if state_dict is None:
        # Prova a vedere se il dict contiene un oggetto modello sotto altra chiave
        for k, v in loaded.items():
            if isinstance(v, nn.Module):
                v = v.to(device)
                v.eval()
                return v, best_threshold, saved_params
        raise RuntimeError("Checkpoint caricato non contiene 'model_state_dict' né 'state_dict'.")

    # Rimuovi eventuale 'module.' da DataParallel
    state_dict = _strip_module_prefix(state_dict)

    # Costruzione parametri da passare al costruttore del modello.
    # Prendiamo la signature del costruttore e filtri i saved_params per quelli validi.
    try:
        sig = inspect.signature(model_class.__init__)
        valid_args = set(sig.parameters.keys()) - {"self", "args", "kwargs"}
    except Exception:
        valid_args = set()

    # Convertiamo nomi param in formattazione accettata:
    model_kwargs = {}
    for k, v in saved_params.items():
        if k in valid_args:
            model_kwargs[k] = v

    # Alcuni fallback utili se mancano param essenziali:
    if "input_channels" in valid_args and "input_channels" not in model_kwargs:
        model_kwargs.setdefault("input_channels", 1)
    if "num_classes" in valid_args and "num_classes" not in model_kwargs:
        model_kwargs.setdefault("num_classes", 2)

    # Costruzione del modello
    try:
        model = model_class(**model_kwargs)
    except Exception as e:
        # Se non riusciamo a costruire con i params salvati, proviamo a costruire usando solo defaults
        # (Questo può succedere se la classe è stata modificata). In questo caso warn e ricostruisce model vuoto.
        print(f"[load_checkpoint] Warning: non ho potuto costruire il modello con i parametri salvati: {e}")
        print("[load_checkpoint] Ricostruisco il modello con i parametri di default (se il costruttore lo permette).")
        model = model_class()  # può fallire se il costruttore richiede arg obbligatori

    model = model.to(device)

    # Carica pesi (potrebbe fallire se architetture diverse)
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        # Tenta un caricamento non-strict per recuperare quello possibile
        print(f"[load_checkpoint] load_state_dict strict failed: {e}")
        print("[load_checkpoint] Riprovo con strict=False per caricare i pesi compatibili.")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, float(best_threshold), saved_params
