import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, confusion_matrix
import timeit
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Aggiungo il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_LSTM as conf
device = conf.device

# Definizione del modello LSTM con PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes, bidirectional):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)
    
    def forward(self, x):
        # Inizializza stati nascosti (h0, c0)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out ha dimensioni [batch_size, seq_length, hidden_size]
        out = self.fc(out[:, -1, :])  # Passo l'ultimo step temporale attraverso il fully connected

        return out

def load_normalized_data(csv_file_path):
    return pd.read_csv(csv_file_path)

def prepare_data_loaders(df, val_size=0.3):
    X = df.iloc[:, 3:].values  # Le colonne da 3 in poi contengono la serie temporale
    y = df['BLASTO NY'].values  # Colonna target

    # Suddivisione del dataset in train e validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=conf.seed)

    # Converti i dati in tensori PyTorch e aggiungi input_size=1 espandendo la dimensione finale
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # [batch_size, seq_length, input_size=1]
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor

def train_model(model, X_train, y_train, X_val, y_val, num_epochs, batch_size, learning_rate, patience):
    # Usa DataParallel per distribuire il modello su più GPU se disponibili
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(conf.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Early stopping setup
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            # Valutazione su validation set ogni 10 epoche
            val_metrics = evaluate_model(model, X_val, y_val)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {val_metrics[0]:.4f}')
            val_loss = criterion(model(X_val.to(conf.device)), y_val.to(conf.device)).item()

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    # Load the best model if early stopping occurred
    if best_model:
        model.load_state_dict(best_model)

    return model

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        y = y.to(device)
        
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

def main():
    # Carico i dati normalizzati
    df = load_normalized_data(conf.data_path)

    # Preparo i data loader
    X_train, X_val, y_train, y_val = prepare_data_loaders(df, conf.val_size)

    # Definisco il modello LSTM
    model = LSTMModel(input_size=1, hidden_size=conf.hidden_size, num_layers=conf.num_layers, 
                      dropout=conf.dropout, num_classes=conf.num_classes, 
                      bidirectional=conf.bidirectional).to(device)
    
    # Addestramento del modello con valutazione sul validation set e early stopping
    model = train_model(model, X_train, y_train, X_val, y_val, conf.num_epochs, conf.batch_size, 
                        conf.learning_rate, patience=10)

    # Valutazione finale del modello sul validation set
    val_metrics = evaluate_model(model, X_val, y_val)

    print(f'MODEL WITH PARAMETERS: num_epochs:{conf.num_epochs}, batch_size:{conf.batch_size}, hidden_size:{conf.hidden_size}, num_layers:{conf.num_layers}, dropout:{conf.dropout}, learning_rate:{conf.learning_rate}  \t\t=====================')
    print(f'val Accuracy: {val_metrics[0]}')
    print(f'val Balanced Accuracy: {val_metrics[1]}')
    print(f"val Cohen's Kappa: {val_metrics[2]}")
    print(f'val Brier Score Loss: {val_metrics[3]}')
    print(f'val F1 Score: {val_metrics[4]}')
    
    # Salvataggio del modello
    model_save_path = os.path.join(parent_dir, conf.test_dir, "lstm_classifier_model.pkl")
    joblib.dump(model, model_save_path)
    print(f'Modello salvato in: {model_save_path}')

if __name__ == "__main__":
    # Misuro il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(main, number=1)
    print("Tempo impiegato per l'esecuzione dell'ottimizzazione LSTM:", execution_time, "secondi")
