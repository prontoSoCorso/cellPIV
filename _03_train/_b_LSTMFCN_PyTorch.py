import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import timeit
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Aggiungo il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf
device = conf.device

# Funzione per caricare i dati
def load_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# Funzione per salvare la matrice di confusione come immagine
def save_confusion_matrix(cm, filename):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

# Funzione per valutare il modello
def evaluate_model(model, dataloader, criterion):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    # Calcolo delle metriche
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob, pos_label=1)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return total_loss / len(dataloader), accuracy, balanced_accuracy, kappa, brier, f1, cm


# Definizione del modello LSTM-FCN in PyTorch
class LSTMFCN(nn.Module):
    def __init__(self, lstm_size, filter_sizes, kernel_sizes, dropout, num_layers):
        super(LSTMFCN, self).__init__()
        
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_size, num_layers=num_layers, batch_first=True)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=filter_sizes[0], kernel_size=kernel_sizes[0]),
            nn.BatchNorm1d(filter_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=filter_sizes[0], out_channels=filter_sizes[1], kernel_size=kernel_sizes[1]),
            nn.BatchNorm1d(filter_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=filter_sizes[1], out_channels=filter_sizes[2], kernel_size=kernel_sizes[2]),
            nn.BatchNorm1d(filter_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(lstm_size + filter_sizes[-1], 2)  # Classificazione binaria

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Prendo l'ultimo stato

        x = torch.transpose(x, 1, 2)  # Trasposizione per convoluzioni
        conv_out = self.conv1(x)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = self.global_pooling(conv_out)
        conv_out = torch.flatten(conv_out, start_dim=1)  # Appiattisco il risultato

        combined = torch.cat((lstm_out, conv_out), dim=-1)
        out = self.fc(combined)
        return out


# Funzione principale
def main():
   # Specifica il numero di giorni da considerare
    days_to_consider = 1

    # Ottieni i percorsi dal config
    train_path, val_path, test_path = conf.get_paths(days_to_consider)

    # Carico i dati normalizzati
    df_train = load_data(train_path)
    df_val = load_data(val_path)
    df_test = load_data(test_path)

    # Preparo i dati
    def prepare_data(df):
        X = torch.tensor(df.iloc[:, 3:].values, dtype=torch.float32).unsqueeze(-1)  # Aggiungo la dimensione per il canale
        y = torch.tensor(df['BLASTO NY'].values, dtype=torch.long)
        return TensorDataset(X, y)

    train_data = prepare_data(df_train)
    val_data = prepare_data(df_val)
    test_data = prepare_data(df_test)

    train_loader = DataLoader(train_data, batch_size=conf.batch_size_FCN, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=conf.batch_size_FCN, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=conf.batch_size_FCN, shuffle=False)

    # Definizione del modello
    model = LSTMFCN(
        lstm_size=conf.lstm_size_FCN,
        filter_sizes=conf.filter_sizes_FCN,
        kernel_sizes=conf.kernel_sizes_FCN,
        dropout=conf.dropout_FCN,
        num_layers=conf.num_layers_FCN
    ).to(device)

    # Impostazioni per l'ottimizzazione
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate_FCN)

    # Scheduler ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.8)

    # Addestramento e validazione
    best_val_accuracy = 0
    num_epochs_final_train = 10
    best_model_path = os.path.join(parent_dir, conf.test_dir, f"best_lstm_fcn_model_{days_to_consider}Days.pth")

    for epoch in range(conf.num_epochs_FCN):
        model.train()
        y_true_train, y_pred_train = [], []
        total_train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            y_true_train.extend(y.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())
            total_train_loss += loss.item()

        # Validazione ogni 5 epoche
        if (epoch + 1) % 5 == 0:
            # Calcolo delle metriche di training
            train_accuracy = accuracy_score(y_true_train, y_pred_train)
            val_loss, val_accuracy, _, _, _, _, _ = evaluate_model(model, val_loader, criterion)
            print(f"Epoch {epoch + 1}: Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            # Salva il modello con la migliore accuratezza di validazione
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                num_epochs_final_train = epoch
                torch.save(model.state_dict(), best_model_path)
                print(f"Model saved with Validation Accuracy: {best_val_accuracy:.4f}")

            # Aggiorna lo scheduler in base alla loss di validazione
            scheduler.step(val_loss)
            print(f"Learning rate attuale: {scheduler.optimizer.param_groups[0]['lr']}")


    # Ricarica il modello migliore
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # Riallenamento su train + validation
    combined_data = torch.utils.data.ConcatDataset([train_data, val_data])
    combined_loader = DataLoader(combined_data, batch_size=conf.batch_size_FCN, shuffle=True)

    for epoch in range(num_epochs_final_train):
        model.train()
        for X, y in combined_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    # Salva il modello dopo il riallenamento
    torch.save(model.state_dict(), best_model_path)
    print(f"Final model saved at: {best_model_path}")

    # Test finale
    test_loss, test_accuracy, test_balanced_accuracy, test_kappa, test_brier, test_f1, test_cm = evaluate_model(model, test_loader, criterion)

    print("\n=====FINAL TEST RESULTS=====")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Balanced Accuracy: {test_balanced_accuracy:.4f}")
    print(f"Test Cohen's Kappa: {test_kappa:.4f}")
    print(f"Test Brier Score Loss: {test_brier:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    cm_path = os.path.join(parent_dir, "confusion_matrix_lstmfcn_" + str(days_to_consider) + "Days.png")
    save_confusion_matrix(test_cm, cm_path)
    print(f"Confusion Matrix saved at: {cm_path}")

if __name__ == "__main__":
    # Misuro il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(main, number=1)
    print("Tempo impiegato per l'esecuzione dell'ottimizzazione LSTMFCN con pytorch:", execution_time, "secondi")
