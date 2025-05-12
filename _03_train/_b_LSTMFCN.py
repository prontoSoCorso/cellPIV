import sys
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score
import multiprocessing as mp

mp.set_start_method("forkserver", force=True)

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from _utils_._utils import save_confusion_matrix, config_logging, plot_roc_curve, load_data, calculate_metrics
from config import Config_03_train_with_optimization as conf

device = conf.device

class LSTMFCN(nn.Module):
    def __init__(self, lstm_size, filter_sizes, kernel_sizes, dropout, num_layers):
        super(LSTMFCN, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_size, num_layers=num_layers, batch_first=True)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, filter_sizes[0], kernel_sizes[0]),
            nn.BatchNorm1d(filter_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(filter_sizes[0], filter_sizes[1], kernel_sizes[1]),
            nn.BatchNorm1d(filter_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(filter_sizes[1], filter_sizes[2], kernel_sizes[2]),
            nn.BatchNorm1d(filter_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(lstm_size + filter_sizes[-1], 2)

    def forward(self, x):
        # x may be either [N, L, C] (quello che facevo prima) or [N, C, L] (what Grad-CAM passes in)
        if x.shape[2] == self.lstm.input_size:
            # it’s already [N, L, C]
            lstm_in = x
            conv_in = x.permute(0, 2, 1)    # → [N, C, L]
        else:
            # it must be [N, C, L]
            lstm_in = x.permute(0, 2, 1)    # → [N, L, C]
            conv_in = x                     # already [N, C, L]
 
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = lstm_out[:, -1, :]
        x = conv_in

        conv_out = self.conv1(x)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = self.global_pooling(conv_out)
        conv_out = torch.flatten(conv_out, 1)
        combined = torch.cat((lstm_out, conv_out), dim=-1)
        return self.fc(combined)

def prepare_data(df):
    temporal_columns = [col for col in df.columns if col.startswith("value_")]
    X = torch.tensor(df[temporal_columns].values, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(df['BLASTO NY'].values, dtype=torch.long)
    return TensorDataset(X, y)

def find_best_threshold(model, val_loader, thresholds=np.linspace(0.0, 0.5, 101)):
    """ Trova la migliore soglia basata sulla balanced accuracy sul validation set. """
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            y_prob.extend(probs)
            y_true.extend(y.cpu().numpy())
    
    best_threshold = 0.5
    best_balanced_accuracy = 0.0
    for threshold in thresholds:
        y_pred = (np.array(y_prob) >= threshold).astype(int)
        acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        
        if acc > best_balanced_accuracy:
            best_balanced_accuracy = acc
            best_threshold = threshold

    return best_threshold

def evaluate_model(model, dataloader, threshold=0.5):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
    return calculate_metrics(y_true, y_pred, y_prob)

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, patience, trial=None):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=math.floor(patience/4), factor=0.5)
    best_val_loss = float('inf')
    best_metric = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

        val_metrics = evaluate_model(model, val_loader)
        val_loss = val_metrics['brier']
        scheduler.step(val_loss)

        if trial:
            trial.report(val_metrics[conf.most_important_metric], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if val_loss < (best_val_loss - 0.001):
            best_val_loss = val_loss
            best_metric = val_metrics[conf.most_important_metric]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_metric

def objective(trial, days_to_consider, train_data, val_data):
    params = {
        'lstm_size': trial.suggest_categorical('lstm_size', conf.lstm_size_options),
        'filter_sizes': tuple(map(int, trial.suggest_categorical('filter_sizes', conf.filter_sizes_options).split(','))),
        'kernel_sizes': tuple(map(int, trial.suggest_categorical('kernel_sizes', conf.kernel_sizes_options).split(','))),
        'dropout': trial.suggest_float('dropout', *conf.dropout_range),
        'num_layers': trial.suggest_int('num_layers', *conf.num_layers_range),
        'batch_size': trial.suggest_categorical('batch_size', conf.batch_size_options),
        'learning_rate': trial.suggest_float('learning_rate', *conf.learning_rate_range, log=True)
    }

    # 2) build DataLoaders with the suggested batch_size
    train_loader = DataLoader(train_data,
                              batch_size=params['batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              num_workers=16,
                              persistent_workers=True,  # keep workers alive across epochs
                              multiprocessing_context='forkserver')
    val_loader   = DataLoader(val_data,
                              batch_size=params['batch_size'])

    model = LSTMFCN(
        lstm_size=params['lstm_size'],
        filter_sizes=params['filter_sizes'],
        kernel_sizes=params['kernel_sizes'],
        dropout=params['dropout'],
        num_layers=params['num_layers']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    metric = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=conf.optuna_num_epochs,
        patience=conf.early_stopping_patience,
        trial=trial
    )
    return metric




def main(days_to_consider=1,
         train_path="", val_path="", test_path="", default_path=True, 
         save_plots=conf.save_plots,
         output_dir_plots=conf.output_dir_plots, 
         output_model_base_dir=conf.output_model_base_dir,
         trial=None,
         run_test_evaluation=conf.run_test_evaluation, **kwargs):
    
    log_filename = kwargs.get('log_filename')
    if (trial is None) and (log_filename!="" and log_filename is not None):
        config_logging(log_dir=kwargs.get('log_dir'), log_filename=kwargs.get('log_filename'))
    
    if default_path:
        train_path, val_path, test_path = conf.get_paths(days_to_consider)

    os.makedirs(output_model_base_dir, exist_ok=True)
    if default_path:
        # Ottieni i percorsi dal config
        train_path, val_path, test_path = conf.get_paths(days_to_consider)
    
    # Caricamento e preparazione dati
    df_train = load_data(train_path)
    df_val = load_data(val_path)
    df_test = load_data(test_path) if run_test_evaluation else None

    if trial:  # Use smaller subset for optimization
        df_train = df_train.sample(frac=0.5, random_state=conf.seed)
        df_val = df_val.sample(frac=0.5, random_state=conf.seed)
    
    train_data = prepare_data(df_train)
    val_data = prepare_data(df_val)
    test_data = prepare_data(df_test) if df_test is not None else None
    
    if trial is not None:
        return objective(trial, days_to_consider, 
                         train_data, val_data)

    # Full training logic
    model = LSTMFCN(
        lstm_size=kwargs.get('lstm_size'),
        filter_sizes=tuple(map(int, kwargs.get('filter_sizes').split(','))),
        kernel_sizes=tuple(map(int, kwargs.get('kernel_sizes').split(','))),
        dropout=kwargs.get('dropout'),
        num_layers=kwargs.get('num_layers')
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=kwargs.get('learning_rate'))
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_data, batch_size=kwargs.get('batch_size'), shuffle=True)
    val_loader = DataLoader(val_data, batch_size=kwargs.get('batch_size'))

    train_model(model, train_loader, val_loader, optimizer, criterion,
               conf.num_epochs_FCN, conf.early_stopping_patience)

    # Final evaluation
    final_threshold = find_best_threshold(model, val_loader)
    model_state = {
        'model_state_dict': model.state_dict(),
        'best_threshold': final_threshold,
        'params': kwargs
    }
    best_model_path = os.path.join(output_model_base_dir, f"best_lstmfcn_model_{days_to_consider}Days.pth")
    torch.save(model_state, best_model_path)

    if run_test_evaluation and test_data:
        test_metrics = evaluate_model(model, DataLoader(test_data, batch_size=kwargs.get('batch_size')), final_threshold)
        logging.info("\n===== FINAL TEST RESULTS =====")
        for metric, value in test_metrics.items():
            if metric not in ("conf_matrix", "fpr", "tpr"):
                logging.info(f"{metric.capitalize()}: {value:.4f}")

        if save_plots:
            complete_output_dir = os.path.join(output_dir_plots, f"day{days_to_consider}")
            os.makedirs(complete_output_dir, exist_ok=True)
            conf_matrix_filename = os.path.join(complete_output_dir, f'confusion_matrix_LSTMFCN_{days_to_consider}Days.png')
            save_confusion_matrix(test_metrics['conf_matrix'], conf_matrix_filename, "LSTMFCN")
            plot_roc_curve(test_metrics['fpr'], test_metrics['tpr'], test_metrics['roc_auc'], 
                          conf_matrix_filename.replace('confusion_matrix', 'roc'))

    return test_metrics if run_test_evaluation else None

if __name__ == "__main__":
    import time
    start_time = time.time()
    day=3
    
    main(
        days_to_consider=day,
        train_path="", val_path="", test_path="",
        default_path=True,
        run_test_evaluation=False,
        log_dir="",
        log_filename="",
        # default hyperparameters from config.py
        batch_size=conf.batch_size_FCN,
        dropout=conf.dropout_FCN,
        filter_sizes=conf.filter_sizes_FCN,
        kernel_sizes=conf.kernel_sizes_FCN,
        lstm_size=conf.lstm_size_FCN,
        num_layers=conf.num_layers_FCN,
        attention=conf.attention_FCN,
        learning_rate=conf.learning_rate_FCN,
        final_epochs=conf.final_epochs_FCN
    )
    
    logging.info(f"Total execution time: {time.time() - start_time:.2f}s")