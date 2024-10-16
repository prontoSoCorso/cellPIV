import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, confusion_matrix
import timeit
import seaborn as sns
import matplotlib.pyplot as plt

# Aggiungo il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_LSTM as conf
from config import paths_for_models as paths_for_models
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

def prepare_data_loaders(df_train, df_val):
    # Combina i due DataFrame in un unico DataFrame
    df = pd.concat([df_train, df_val], ignore_index=True)
    
    X_train = df.iloc[:, 3:].values  # Le colonne da 3 in poi contengono la serie temporale
    y_train = df['BLASTO NY'].values  # Colonna target

    # Converti i dati in tensori PyTorch e aggiungi input_size=1 espandendo la dimensione finale
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    return X_train_tensor, y_train_tensor

# Funzione per salvare la matrice di confusione come immagine
def save_confusion_matrix(cm, filename):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

def train_model(model, X_train, y_train, num_epochs, batch_size, learning_rate):
    model = model.to(conf.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

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
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
    return model

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()
        y_prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        
        accuracy = accuracy_score(y, y_pred)
        balanced_accuracy = balanced_accuracy_score(y, y_pred)
        kappa = cohen_kappa_score(y, y_pred)
        brier = brier_score_loss(y, y_prob, pos_label=1)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        
        return accuracy, balanced_accuracy, kappa, brier, f1, cm

def main():
    # Carico i dati normalizzati
    df_train = load_normalized_data(paths_for_models.data_path_train)
    df_val = load_normalized_data(paths_for_models.data_path_val)

    # Preparo i data loader
    X_train, y_train = prepare_data_loaders(df_train, df_val)
    
    # Definisco il modello LSTM
    model = LSTMModel(input_size=1, hidden_size=conf.hidden_size, num_layers=conf.num_layers, 
                      dropout=conf.dropout, num_classes=conf.num_classes, 
                      bidirectional=conf.bidirectional)
    
    # Addestramento del modello con valutazione sul validation set e early stopping
    model = train_model(model, X_train, y_train, conf.num_epochs, conf.batch_size, 
                        conf.learning_rate)


    df_test = load_normalized_data(paths_for_models.test_path)
    X_test = df_test.iloc[:, 3:].values
    y_test = df_test['BLASTO NY'].values

    test_metrics = evaluate_model(model, X_test, y_test)

    print(f'=====LSTM RESULTS=====')
    print(f'Test Accuracy: {test_metrics[0]}')
    print(f'Test Balanced Accuracy: {test_metrics[1]}')
    print(f"Test Cohen's Kappa: {test_metrics[2]}")
    print(f'Test Brier Score Loss: {test_metrics[3]}')
    print(f'Test F1 Score: {test_metrics[4]}')

    save_confusion_matrix(test_metrics[5], "confusion_matrix_lstm.png")

    # Salvataggio del modello
    model_save_path = os.path.join(parent_dir, conf.test_dir, "lstm_classifier_model.pth")
    torch.save(model, model_save_path)  # Salvataggio del modello con torch
    print(f'Modello salvato in: {model_save_path}')

if __name__ == "__main__":
    # Misuro il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(main, number=1)
    print("Tempo impiegato per l'esecuzione dell'ottimizzazione LSTM:", execution_time, "secondi")
