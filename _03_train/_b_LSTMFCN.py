import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import timeit
import torch.optim.lr_scheduler

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf
if conf.device:
    device = conf.device

# Funzione per caricare i dati normalizzati da CSV
def load_data(csv_file_path):
    return pd.read_csv(csv_file_path)

def prepare_data(df):
    temporal_columns = [col for col in df.columns if col.startswith("value_")]
    X = torch.tensor(df[temporal_columns].values, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(df['BLASTO NY'].values, dtype=torch.long)
    return TensorDataset(X, y)

# Funzione per salvare la matrice di confusione come immagine
def save_confusion_matrix(conf_matrix, filename):
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - LSTMFCN")
    plt.savefig(filename)
    plt.close()

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


def evaluate_model_torch(model, dataloader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    return {
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

def main(days_to_consider=1,
         output_dir_plots = parent_dir, 
         batch_size=conf.batch_size_FCN, 
         lstm_size=conf.lstm_size_FCN,
         filter_sizes=conf.filter_sizes_FCN,
         kernel_sizes=conf.kernel_sizes_FCN,
         dropout=conf.dropout_FCN,
         num_layers=conf.num_layers_FCN,
         learning_rate=conf.learning_rate_FCN,
         num_epochs=conf.num_epochs_FCN):
    
    os.makedirs(output_dir_plots, exist_ok=True)
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
    most_important_metric = "balanced_accuracy"
    best_model_path = os.path.join(parent_dir, "_04_test", f"best_lstmfcn_{days_to_consider}Days.pth")

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
        
        # Early stopping and save best model
        if val_loss < (best_val_loss - early_stopping_delta):
            best_val_acc = val_metrics[most_important_metric]
            best_val_loss = val_loss
            epochs_no_improve = 0
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch: {epoch}, Model saved with validation {most_important_metric.capitalize()}: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Epoch: {epoch}, Validation {most_important_metric.capitalize()}: {best_val_acc:.4f}, Brier_score: {val_loss}, Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")
            
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping activated at epoch number {epoch}")
                break

    # Caricamento miglior modello
    model.load_state_dict(torch.load(best_model_path))
    
    # Valutazione finale
    test_metrics = evaluate_model_torch(model, test_loader)
    
    print("\n===== FINAL TEST RESULTS =====")
    for metric, value in test_metrics.items():
        if (metric!="conf_matrix") and (metric!="fpr") and (metric!="tpr"):
            print(f"{metric.capitalize()}: {value:.4f}")
    plot_roc_curve(test_metrics['fpr'], test_metrics['tpr'], 
                   test_metrics['roc_auc'], 
                   os.path.join(output_dir_plots, f"roc_curve_LSTMFCN_{days_to_consider}Days.png"))

    # Salvataggio matrice di confusione
    cm_path = os.path.join(output_dir_plots, f"confusion_matrix_LSTMFCN_{days_to_consider}Days.png")
    save_confusion_matrix(conf_matrix=test_metrics["conf_matrix"], filename=cm_path)
    
    return test_metrics

if __name__ == "__main__":
    execution_time = timeit.timeit(lambda: main(days_to_consider=7), number=1)
    print(f"Tempo esecuzione LSTM-FCN: {execution_time:.2f}s")