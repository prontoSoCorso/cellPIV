import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score

# Percorsi
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from _utils_._utils import (save_results, config_logging,
                            calculate_metrics, _check_data_dict)
from config import Config_03_train_with_optimization as conf
from _03_train._b_LSTMFCN import TimeSeriesClassifier

device = conf.device

DEFAULTS = {
    "enc_hidden_dim": 64,
    "dropout_enc": 0.2,
    "num_features": 128,
    "n_heads": 8,
    "dropout_attn": 0.2,
    "lstm_hidden_dim": 128,
    "lstm_layers": 2,
    "bidirectional": True,
    "dropout_lstm": 0.2,
    "cnn_filter_sizes": (128, 256, 128),
    "cnn_kernel_sizes": (7, 5, 3),
    "dropout_cnn": 0.3,
    "se_ratio": 0.25,
    "dropout_classifier": 0.3,
    "use_positional": True,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "num_epochs": 150,
    "patience": 50,
    "scheduler_factor": 0.75,
    }

def prepare_data(X, y):
    # X: (N, 1, T) -> torch [N, 1, T] ; il modello accetta anche [N, 1, T]
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X, y)

def find_best_threshold(model, val_loader, thresholds=np.linspace(0.0, 0.5, 101)):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            out = model(X)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            y_prob.extend(prob)
            y_true.extend(y.numpy())
    best_t, best_acc = 0.5, 0.0
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    for t in thresholds:
        acc = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t

def evaluate_model(model, dataloader, threshold=0.5):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            y_true.extend(y.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
    return calculate_metrics(y_true, y_pred, y_prob)

def val_ce_loss(model, dataloader, criterion):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total += loss.item() * X.size(0)
            n += X.size(0)
    return total / max(1, n)

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, patience, factor, trial=None):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=max(1, patience//5), factor=factor, verbose=True)
    best_score = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

        vloss = val_ce_loss(model, val_loader, criterion)
        scheduler.step(vloss)
        # metric per Optuna / tracking
        metrics = evaluate_model(model, val_loader, threshold=0.5)
        score = metrics[conf.most_important_metric]

        if trial:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader.dataset):.4f}, Val CE Loss: {vloss:.4f}, Val {conf.most_important_metric}: {score:.4f}")
        
        if score > best_score + 1e-5:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_score

def objective(trial, train_data, val_data):
    params = {
        "enc_hidden_dim": trial.suggest_categorical("enc_hidden_dim", [32, 64, 96, 128]),
        "num_features": trial.suggest_categorical("num_features", [64, 96, 128, 192]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "dropout_attn": trial.suggest_float("dropout_attn", 0.05, 0.4),
        "lstm_hidden_dim": trial.suggest_categorical("lstm_hidden_dim", [32, 64, 96, 128]),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        "dropout_lstm": trial.suggest_float("dropout_lstm", 0.0, 0.4),
        "cnn_filter_sizes": trial.suggest_categorical("cnn_filter_sizes",
            [(64, 128, 256), (64, 128, 128), (96, 128, 192)]),
        "cnn_kernel_sizes": trial.suggest_categorical("cnn_kernel_sizes",
            [(3, 3, 3), (3, 5, 5), (5, 5, 7)]),
        "dropout_cnn": trial.suggest_float("dropout_cnn", 0.0, 0.4),
        "dropout_classifier": trial.suggest_float("dropout_classifier", 0.0, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "use_positional": trial.suggest_categorical("use_positional", [True, False]),
        "scheduler_factor": trial.suggest_float("scheduler_factor", 0.5, 0.9)
    }

    # DataLoaders with batch_size
    train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=params["batch_size"])

    # Model, optimizer, criterion
    model = TimeSeriesClassifier(
        input_channels=1,
        enc_hidden_dim=params["enc_hidden_dim"],
        num_features=params["num_features"],
        n_heads=params["n_heads"],
        dropout_attn=params["dropout_attn"],
        lstm_hidden_dim=params["lstm_hidden_dim"],
        lstm_layers=params["lstm_layers"],
        bidirectional=params["bidirectional"],
        dropout_lstm=params["dropout_lstm"],
        cnn_filter_sizes=params["cnn_filter_sizes"],
        cnn_kernel_sizes=params["cnn_kernel_sizes"],
        dropout_cnn=params["dropout_cnn"],
        dropout_classifier=params["dropout_classifier"],
        use_positional=params["use_positional"],
        num_classes=2,
    ).to(device)

    print("Model architecture:")
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    metric = train_model(model, train_loader, val_loader, optimizer, criterion,
                         num_epochs=conf.optuna_num_epochs, patience=conf.early_stopping_patience, factor=params["scheduler_factor"],
                         trial=trial)
    return metric


def main(
        data=None,
        save_plots=conf.save_plots,
        output_dir_plots=conf.output_dir_plots,
        output_model_base_dir=conf.output_model_base_dir,
        trial=None,
        run_test_evaluation=conf.run_test_evaluation,
        **kwargs,
        ):
    if data is None:
        raise ValueError("Il parametro 'data' è obbligatorio e deve essere un dizionario con i dataset.")

    log_filename = kwargs.get("log_filename")
    if (trial is None) and (log_filename != "" and log_filename is not None):
        config_logging(log_dir=kwargs.get("log_dir"), log_filename=kwargs.get("log_filename"))

    os.makedirs(output_model_base_dir, exist_ok=True)

    # Validate & unpack
    X_train, y_train, X_val, y_val, X_test, y_test = _check_data_dict(data, require_test=bool(run_test_evaluation))

    train_data = prepare_data(X_train, y_train)
    val_data = prepare_data(X_val, y_val)
    test_data = prepare_data(X_test, y_test) if (run_test_evaluation and X_test is not None) else None

    # Optuna path
    if trial is not None:
        return objective(trial, train_data, val_data)

    # Full training with provided or default params
    p = {**DEFAULTS, **{k: v for k, v in kwargs.items() if v is not None}}
    train_loader = DataLoader(train_data, batch_size=p["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=p["batch_size"])

    model = TimeSeriesClassifier(
        input_channels=1,
        enc_hidden_dim=p["enc_hidden_dim"],
        dropout_enc=p["dropout_enc"],
        num_features=p["num_features"],
        n_heads=p["n_heads"],
        dropout_attn=p["dropout_attn"],
        lstm_hidden_dim=p["lstm_hidden_dim"],
        lstm_layers=p["lstm_layers"],
        bidirectional=p["bidirectional"],
        dropout_lstm=p["dropout_lstm"],
        cnn_filter_sizes=p["cnn_filter_sizes"],
        cnn_kernel_sizes=p["cnn_kernel_sizes"],
        dropout_cnn=p["dropout_cnn"],
        se_ratio=p["se_ratio"],
        dropout_classifier=p["dropout_classifier"],
        use_positional=p["use_positional"],
        num_classes=2,
    ).to(device)

    print("Model architecture:")
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = optim.Adam(model.parameters(), lr=p["learning_rate"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, val_loader, optimizer, criterion,
                num_epochs=p["num_epochs"], patience=p["patience"], factor=p["scheduler_factor"], trial=None)

    # Threshold & save
    final_threshold = find_best_threshold(model, val_loader)
    state = {
        "model_state_dict": model.state_dict(),
        "best_threshold": final_threshold,
        "params": p,
    }
    day_label = kwargs.get("day_label", "Unknown")
    if isinstance(day_label, int):
        day_label = f"{day_label}"
    elif isinstance(day_label, str):
        day_label = day_label.replace(" ", "_")
    best_model_path = os.path.join(output_model_base_dir, f"best_lstmfcn_model_{day_label}Days.pth")
    torch.save(state, best_model_path)
    logging.info(f"Model saved at: {best_model_path}")

    # Test
    test_metrics = None
    if run_test_evaluation and test_data is not None:
        test_loader = DataLoader(test_data, batch_size=p["batch_size"])
        test_metrics = evaluate_model(model, test_loader, threshold=final_threshold)
        save_results(test_metrics, output_dir_plots, "LSTMFCN", day_label, save_plots=save_plots)

    return test_metrics if run_test_evaluation else None

if __name__ == "__main__":
    pass