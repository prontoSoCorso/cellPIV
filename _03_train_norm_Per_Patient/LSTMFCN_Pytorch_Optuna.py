import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score, cohen_kappa_score, brier_score_loss
import optuna
import timeit
import os
import joblib
import sys

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train_lstmfcn_with_optuna as conf

# Definizione del modello LSTM-FCN in PyTorch
class LSTMFCN(nn.Module):
    def __init__(self, lstm_size, filter_sizes, kernel_sizes, dropout, num_layers):
        super(LSTMFCN, self).__init__()
        
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_size, num_layers=num_layers, batch_first=True)
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filter_sizes[0], kernel_size=kernel_sizes[0])
        self.conv2 = nn.Conv1d(in_channels=filter_sizes[0], out_channels=filter_sizes[1], kernel_size=kernel_sizes[1])
        self.conv3 = nn.Conv1d(in_channels=filter_sizes[1], out_channels=filter_sizes[2], kernel_size=kernel_sizes[2])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_size + filter_sizes[-1], 2)  # Classificazione binaria

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Prendo l'ultimo stato

        x = torch.transpose(x, 1, 2)  # Trasposizione per convoluzioni
        conv_out = self.conv1(x)
        conv_out = torch.relu(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = torch.relu(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = torch.relu(conv_out)
        conv_out = torch.mean(conv_out, dim=-1)  # Global Average Pooling

        combined = torch.cat((lstm_out, conv_out), dim=-1)
        combined = self.dropout(combined)
        out = self.fc(combined)
        return out

# Funzione per caricare i dati normalizzati da CSV
def load_normalized_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# Funzione per preparare i data loader (in questo caso utilizziamo sklearn)
def prepare_data_loaders(df, val_size=0.3):
    X = df.iloc[:, 3:].values
    y = df['BLASTO NY'].values

    # Splitting the data into train, validation, and val sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)
    
    # Converti i dati in tensori PyTorch e aggiungi input_size=1 espandendo la dimensione finale
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # [batch_size, seq_length, input_size=1]
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)      # [batch_size, seq_length, input_size=1]
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor

# Funzione per addestrare il modello
def train_model(trial, model, X_train, y_train, X_val, y_val, num_epochs, batch_size, lr):
    # Usa DataParallel per distribuire il modello su più GPU se disponibili
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(conf.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(conf.device)
            labels = labels.to(conf.device)

            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(2))
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            val_metrics = evaluate_model(model, X_val, y_val)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {val_metrics[0]:.4f}')

            # Report della prestazione del trial per Optuna
            trial.report(val_metrics[0], epoch)

            # Controlla se il trial deve essere fermato per pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
            
    return model

# Funzione per valutare il modello
def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X, dtype=torch.float32).unsqueeze(2).to(conf.device))
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()
        accuracy = accuracy_score(y, y_pred)
        balanced_accuracy = balanced_accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        return accuracy, balanced_accuracy, f1, cm

def objective(trial):
    # Definisco lo spazio di ricerca degli iperparametri
    num_epochs = trial.suggest_categorical('num_epochs', [200, 300, 400])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lstm_size = trial.suggest_int('lstm_size', 64, 128)
    filter_sizes = trial.suggest_categorical('filter_sizes', [(128, 256, 128), (256, 128, 64)])
    kernel_sizes = trial.suggest_categorical('kernel_sizes', [(8, 5, 3)])
    num_layers = trial.suggest_categorical('num_layers', [2,3,4])
    dropout = trial.suggest_float('dropout', 0.1, 0.4)

    # Carico i dati e preparo i data loader
    df = load_normalized_data(conf.data_path)
    X_train, X_val, y_train, y_val = prepare_data_loaders(df, val_size=0.3)

    # Modello LSTMFCN
    model = LSTMFCN(lstm_size, filter_sizes, kernel_sizes, dropout, num_layers)
    
    # Addestro il modello
    model = train_model(trial, model, X_train, y_train, X_val, y_val, num_epochs, batch_size, lr=0.001)
    
    # Valutazione
    val_metrics = evaluate_model(model, X_val, y_val)

    # Stampa dei risultati
    print(f'MODEL WITH PARAMETERS: num_epochs:{num_epochs}, batch_size:{batch_size}, hidden_size:{hidden_size}, num_layers:{num_layers}, dropout:{dropout}, learning_rate:{learning_rate}  \t\t=====================')
    print(f'val Accuracy: {val_metrics[0]}')
    print(f'val Balanced Accuracy: {val_metrics[1]}')
    print(f"val Cohen's Kappa: {val_metrics[2]}")
    print(f'val Brier Score Loss: {val_metrics[3]}')
    print(f'val F1 Score: {val_metrics[4]}')
    
    return val_metrics[0]


# Funzione per allenare il miglior modello sui dati completi
def retrain_best_model(best_trial):
    # Carico i dati per allenare il miglior modello
    df = load_normalized_data(conf.data_path)
    X = df.iloc[:, 3:].values
    y = df['BLASTO NY'].values
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    best_params = best_trial.params
    lstm_size = best_params['lstm_size']
    filter_sizes = best_params['filter_sizes']
    kernel_sizes = best_params['kernel_sizes']
    dropout = best_params['dropout']
    num_epochs = best_params['num_epochs']
    batch_size = best_params['batch_size']
        
    best_model = LSTMFCN(lstm_size, filter_sizes, kernel_sizes, dropout)
    if torch.cuda.device_count() > 1:
        best_model = nn.DataParallel(best_model)
    best_model = best_model.to(conf.device)
    
    train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=best_trial.params['batch_size'], shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_trial.params['learning_rate'])

    num_epochs = best_trial.params['num_epochs']

    for epoch in range(num_epochs):
        best_model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(conf.device)
            labels = labels.to(conf.device)
            
            outputs = best_model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    model_save_path = os.path.join(parent_dir, conf.test_dir, "best_lstmfcn_model.pkl")
    joblib.dump(best_model, model_save_path)
    print(f"Best model saved at: {model_save_path}")


def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    best_trial = study.best_trial
    print(f'Numero di trial: {len(study.trials)}')
    print(f'Migliore trial: {best_trial.number}, \nMigliore Accuracy: {best_trial.value}, \nBest trial params: {best_trial.params}')

    # Alleno il miglior modello su tutto il dataset
    retrain_best_model(best_trial)

if __name__ == "__main__":
    start_time = timeit.default_timer()
    main()
    print("Execution time: ", timeit.default_timer() - start_time)
