import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, f1_score, confusion_matrix
import optuna
import joblib
import timeit
import seaborn as sns
import matplotlib.pyplot as plt

# Aggiungi il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_LSTM_WithOptuna as conf

# Definizione del modello LSTM con PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Inizializza gli stati nascosti h0 e c0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Stato nascosto iniziale
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Stato della cella iniziale

        out, _ = self.lstm(x, (h0, c0))  # out ha dimensioni [batch_size, seq_length, hidden_size]
        out = self.fc(out[:, -1, :])  # Prendi l'ultimo passo temporale e passa al fully connected
        return out

def load_normalized_data(csv_file_path):
    return pd.read_csv(csv_file_path)

def prepare_data_loaders(df, val_size=0.3):
    X = df.iloc[:, 3:].values  # Dati a partire dalla quarta colonna
    y = df['BLASTO NY'].values  # Etichette target

    # Suddivisione del dataset in train e validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=conf.seed_everything(conf.seed))

    # Converti i dati in tensori PyTorch e aggiungi input_size=1 espandendo la dimensione finale
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # [batch_size, seq_length, input_size=1]
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)      # [batch_size, seq_length, input_size=1]
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor

def train_model(trial, model, X_train, y_train, X_val, y_val, num_epochs, batch_size, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(conf.device)
            labels = labels.to(conf.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
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

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        X = X.to(conf.device)
        y = y.to(conf.device)
        
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()
        y_prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        
        accuracy = accuracy_score(y.cpu().numpy(), y_pred)
        balanced_accuracy = balanced_accuracy_score(y.cpu().numpy(), y_pred)
        kappa = cohen_kappa_score(y.cpu().numpy(), y_pred)
        brier = brier_score_loss(y.cpu().numpy(), y_prob, pos_label=1)
        f1 = f1_score(y.cpu().numpy(), y_pred)
        cm = confusion_matrix(y.cpu().numpy(), y_pred)
        return accuracy, balanced_accuracy, kappa, brier, f1, cm

# Funzione per salvare la matrice di confusione come immagine
def save_confusion_matrix(cm, filename):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

def objective(trial):
    # Definisce lo spazio di ricerca degli iperparametri
    num_epochs = trial.suggest_categorical('num_epochs', conf.num_epochs)
    batch_size = trial.suggest_categorical('batch_size', conf.batch_size)
    hidden_size = trial.suggest_categorical('hidden_size', conf.hidden_size)
    num_layers = trial.suggest_categorical('num_layers', conf.num_layers)
    dropout = trial.suggest_categorical('dropout', conf.dropout)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    # Carica i dati normalizzati
    df = load_normalized_data(conf.data_path)

    # Prepara i data loader
    X_train, X_val, y_train, y_val = prepare_data_loaders(df, conf.val_size)

    # Definisce il modello LSTM
    model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, 
                      dropout=dropout, num_classes=conf.num_classes).to(conf.device)
    
    # Addestramento del modello con pruning
    model = train_model(trial, model, X_train, y_train, X_val, y_val, num_epochs, batch_size, learning_rate)

    # Valutazione finale del modello
    val_metrics = evaluate_model(model, X_val, y_val)

    # Stampa dei risultati
    print(f'val Accuracy: {val_metrics[0]}')
    print(f'val Balanced Accuracy: {val_metrics[1]}')
    print(f"val Cohen's Kappa: {val_metrics[2]}")
    print(f'val Brier Score Loss: {val_metrics[3]}')
    print(f'val F1 Score: {val_metrics[4]}')

    return val_metrics[0]

def main():
    # Definisci uno studio con il pruner MedianPruner per il pruning dei trial
    study = optuna.create_study(direction='maximize', sampler=conf.sampler, pruner=conf.pruner)
    study.optimize(objective, n_trials=50)

    best_trial = study.best_trial
    print(f'Numero di trial: {len(study.trials)}')
    print(f'Migliore trial: {best_trial.number}, \nMigliore Accuracy: {best_trial.value}, \nBest trial params: {best_trial.params}')

    #Devo riaddestrare il modello migliore. Infatti Il best_trial che si ottiene da Optuna contiene solo i parametri ottimali che 
    # hanno prodotto la miglior performance, non il modello addestrato stesso. Quindi, se dovessi salvare il best_trial, salverei
    # semplicemente i dati riguardo ai parametri, ma non il modello LSTM addestrato.
    # Ricostruisco il modello usando i migliori parametri trovati
    best_model = LSTMModel(input_size=1, hidden_size=best_trial.params['hidden_size'], 
                        num_layers=best_trial.params['num_layers'], 
                        dropout=best_trial.params['dropout'], num_classes=conf.num_classes).to(conf.device)

    # Addestro il modello di nuovo con i dati e i parametri ottimali
    df = load_normalized_data(conf.data_path)
    X_train, X_val, y_train, y_val = prepare_data_loaders(df, conf.val_size)
    best_model.fit(X_train, y_train)

    # Valutazione finale del modello
    val_metrics = evaluate_model(best_model, X_val, y_val)

    # Stampa dei risultati
    print(f'val Accuracy: {val_metrics[0]}')
    print(f'val Balanced Accuracy: {val_metrics[1]}')
    print(f"val Cohen's Kappa: {val_metrics[2]}")
    print(f'val Brier Score Loss: {val_metrics[3]}')
    print(f'val F1 Score: {val_metrics[4]}')

    # Salva la matrice di confusione
    save_confusion_matrix(val_metrics[5], f'confusion_matrix_lstm_.png')
    
    # Salvataggio del modello addestrato
    model_save_path = os.path.join(parent_dir, conf.test_dir, "lstm_model_with_optuna.pkl")
    joblib.dump(best_model, model_save_path)
    print(f"Modello addestrato salvato in: {model_save_path}")

if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(main, number=1)
    print("Tempo impiegato per l'esecuzione dell'ottimizzazione LSTM:", execution_time, "secondi")
