import sys
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score

from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler
import numpy as np

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from _utils_._utils import save_confusion_matrix, config_logging, plot_roc_curve, load_data, calculate_metrics
from config import Config_03_train as conf
if conf.device:
    device = conf.device

def prepare_data(df):
    temporal_columns = [col for col in df.columns if col.startswith("value_")]
    X = torch.tensor(df[temporal_columns].values, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(df['BLASTO NY'].values, dtype=torch.long)
    return TensorDataset(X, y)

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
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = torch.transpose(x, 1, 2)
        conv_out = self.conv1(x)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = self.global_pooling(conv_out)
        conv_out = torch.flatten(conv_out, 1)
        combined = torch.cat((lstm_out, conv_out), dim=-1)
        return self.fc(combined)


def find_best_threshold(model, val_loader, thresholds=np.linspace(0.0, 1.0, 101)):
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


def evaluate_model_torch(model, dataloader, threshold=0.5):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)  # Usa la soglia, se no faccio argmax(outputs, dim=1)
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    metrics = calculate_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    return metrics


def main(days_to_consider=1,
         train_path="", val_path="", test_path="", default_path=True, 
         save_plots=conf.save_plots,
         output_dir_plots = conf.output_dir_plots, 
         output_model_base_dir=conf.output_model_base_dir,
         batch_size=conf.batch_size_FCN, 
         lstm_size=conf.lstm_size_FCN,
         filter_sizes=conf.filter_sizes_FCN,
         kernel_sizes=conf.kernel_sizes_FCN,
         dropout=conf.dropout_FCN,
         num_layers=conf.num_layers_FCN,
         learning_rate=conf.learning_rate_FCN,
         num_epochs=conf.num_epochs_FCN,
         most_important_metric = conf.most_important_metric,

         log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "logging_files"),
         log_filename=f'train_LSTMFCN_based_on_{conf.method_optical_flow}'):
    
    config_logging(log_dir=log_dir, log_filename=log_filename)

    os.makedirs(output_model_base_dir, exist_ok=True)
    if default_path:
        # Ottieni i percorsi dal config
        train_path, val_path, test_path = conf.get_paths(days_to_consider)
    
    # Caricamento e preparazione dati
    df_train = load_data(train_path)
    df_val = load_data(val_path)
    df_test = load_data(test_path)
    
    train_data = prepare_data(df_train)
    val_data = prepare_data(df_val)
    test_data = prepare_data(df_test)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Inizializzazione modello
    def instantiate_model():
        model = LSTMFCN(
            lstm_size=lstm_size,
            filter_sizes=filter_sizes,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            num_layers=num_layers
            ).to(device)
        return model

    model = instantiate_model()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.8)
    
    early_stopping_patience = 31  # Epoche di attesa prima di fermarsi
    early_stopping_delta = 0.001  # Miglioramento minimo richiesto
    epochs_no_improve = 0
    best_val_loss = float('inf')

    best_val_acc = 0
    best_model_path = os.path.join(output_model_base_dir, f"best_lstmfcn_model_{days_to_consider}Days.pth")

    # Addestramento
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Training phase
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*X.size(0)
        
        # Validation phase
        val_metrics = evaluate_model_torch(model, val_loader)
        val_loss = val_metrics['brier']  # uso Brier score per l'early stopping
        scheduler.step(val_loss)

        # Log epoch metrics
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
        #logging.info(f"Train Loss: {train_loss/len(train_loader.dataset):.4f}")
        logging.info((
            f"Val Brier Score: {val_loss:.4f} | "
            f"Val {most_important_metric.capitalize()}: {val_metrics[most_important_metric]:.4f} | "
            f"Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}" 
            )
        )
        logging.info(f"Val {most_important_metric.capitalize()}: {val_metrics[most_important_metric]:.4f}")
        logging.info(f"")
    
        # Early stopping and save best model
        if val_loss < (best_val_loss - early_stopping_delta):
            best_val_acc = val_metrics[most_important_metric]
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Model saved with validation {most_important_metric.capitalize()}: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Caricamento miglior modello
    model.load_state_dict(torch.load(best_model_path))
    
    # Trova la soglia migliore sul validation set con il migliore modello trovato
    final_threshold = find_best_threshold(model=model, val_loader=val_loader)
    logging.info(f"\nBest Threshold Found: {final_threshold:.4f}")

    # Model update with the best threshold
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_threshold': final_threshold,
        'lstm_size': lstm_size,
        'filter_sizes': filter_sizes,
        'kernel_sizes': kernel_sizes,
        'dropout': dropout,
        'num_layers': num_layers,
        'batch_size': batch_size
    }, best_model_path)
    logging.info(f"Final model saved to: {best_model_path}")

    # Valutazione finale
    test_metrics = evaluate_model_torch(model, test_loader, threshold=final_threshold)
    
    logging.info("\n===== FINAL TEST RESULTS =====")
    for metric, value in test_metrics.items():
        if (metric!="conf_matrix") and (metric!="fpr") and (metric!="tpr"):
            logging.info(f"{metric.capitalize()}: {value:.4f}")
    
    # Save plots
    if save_plots:
        complete_output_dir = os.path.join(output_dir_plots, f"day{days_to_consider}")
        os.makedirs(complete_output_dir, exist_ok=True)
        conf_matrix_filename=os.path.join(complete_output_dir,f'confusion_matrix_LSTMFCN_{days_to_consider}Days.png')
        save_confusion_matrix(conf_matrix=test_metrics['conf_matrix'], 
                                filename=conf_matrix_filename, 
                                model_name="LSTMFCN")
        plot_roc_curve(fpr=test_metrics['fpr'], tpr=test_metrics['tpr'], 
                        roc_auc=test_metrics['roc_auc'], 
                        filename=conf_matrix_filename.replace('confusion_matrix', 'roc'))

    return test_metrics

if __name__ == "__main__":
    import time
    start_time = time.time()
    main(days_to_consider=7)
    print(f"Total execution time LSTMFCN: {(time.time() - start_time):.2f}s")