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

from config import Config_03_train_lstmfcn as conf
from config import paths_for_models as paths_for_models
device = conf.device

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
    model = LSTMFCN(lstm_size=conf.hidden_size, filter_sizes=conf.filter_sizes, 
                    kernel_sizes=conf.kernel_sizes, dropout=conf.dropout, num_layers=conf.num_layers)
            
    # Addestramento del modello con valutazione sul validation set e early stopping
    model = train_model(model, X_train, y_train, conf.num_epochs, conf.batch_size, 
                        conf.learning_rate)


    df_test = load_normalized_data(paths_for_models.test_path)
    X_test = df_test.iloc[:, 3:].values
    y_test = df_test['BLASTO NY'].values

    test_metrics = evaluate_model(model, X_test, y_test)

    print(f'=====LSTMFCN with PyTorch RESULTS=====')
    print(f'Test Accuracy: {test_metrics[0]}')
    print(f'Test Balanced Accuracy: {test_metrics[1]}')
    print(f"Test Cohen's Kappa: {test_metrics[2]}")
    print(f'Test Brier Score Loss: {test_metrics[3]}')
    print(f'Test F1 Score: {test_metrics[4]}')

    save_confusion_matrix(test_metrics[5], "confusion_matrix_lstmfcn_pytorch.png")

    # Salvataggio del modello
    model_save_path = os.path.join(parent_dir, conf.test_dir, "lstmfcn_pytorch_classifier_model.pth")
    torch.save(model, model_save_path)  # Salvataggio del modello con torch
    print(f'Modello salvato in: {model_save_path}')

if __name__ == "__main__":
    # Misuro il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(main, number=1)
    print("Tempo impiegato per l'esecuzione dell'ottimizzazione LSTMFCN con pytorch:", execution_time, "secondi")
