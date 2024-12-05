import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import umap.umap_ as umap
import matplotlib.pyplot as plt

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf
from _03_train.LSTMFCN_PyTorch import LSTMFCN

# Riduci i log di TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

device = conf.device

# Funzione per caricare i dati
def load_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# Funzione per preparare i dati
def prepare_data(df):
    X = torch.tensor(df.iloc[:, 3:].values, dtype=torch.float32).unsqueeze(-1)  # Aggiungo la dimensione per il canale
    y = torch.tensor(df['BLASTO NY'].values, dtype=torch.long)
    return TensorDataset(X, y)

# Funzione per valutare il modello
def evaluate_model_and_extract_features(model, dataloader):
    model.eval()
    features, labels = [], []
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Output dal modello
            lstm_out, _ = model.lstm(X)
            lstm_out = lstm_out[:, -1, :]  # Ultimo stato nascosto dell'LSTM

            X_transposed = torch.transpose(X, 1, 2)
            conv_out = model.conv1(X_transposed)
            conv_out = model.conv2(conv_out)
            conv_out = model.conv3(conv_out)
            conv_out = model.global_pooling(conv_out)
            conv_out = torch.flatten(conv_out, start_dim=1)

            # Combino LSTM e CNN feature
            combined_features = torch.cat((lstm_out, conv_out), dim=-1)
            features.append(combined_features.cpu().numpy())
            labels.append(y.cpu().numpy())

            # Predizioni per le metriche
            outputs = model.fc(combined_features)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Calcolo delle metriche di accuratezza
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Concatena tutte le feature e le etichette
    features = np.vstack(features)
    labels = np.hstack(labels)

    return features, labels, accuracy, cm

# Funzione per visualizzare UMAP
def visualize_umap(features, labels, save_path):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embeddings = reducer.fit_transform(features)

    plt.figure(figsize=(10, 8))
    # Usare colori distinti per classi binarie
    colors = ['red', 'blue']
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap=plt.matplotlib.colors.ListedColormap(colors), alpha=0.7)

    # Aggiungere legenda
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
    plt.legend(handles, ['Class 0', 'Class 1'], title='Classes')

    plt.title("Feature Visualization with UMAP")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(save_path)
    plt.show()
    print(f"UMAP visualization saved at: {save_path}")

# Script principale
def main():
    # Percorso del modello salvato
    final_model_path = os.path.join(conf.test_dir, "best_lstm_fcn_model.pth")

    if not os.path.exists(final_model_path):
        raise FileNotFoundError(f"Model file not found at: {final_model_path}")

    # Carico i dati di test
    df_test = load_data(conf.test_path)
    test_data = prepare_data(df_test)
    test_loader = DataLoader(test_data, batch_size=conf.batch_size_FCN, shuffle=False)

    # Carico il modello
    model = LSTMFCN(
        lstm_size=conf.lstm_size_FCN,
        filter_sizes=conf.filter_sizes_FCN,
        kernel_sizes=conf.kernel_sizes_FCN,
        dropout=conf.dropout_FCN,
        num_layers=conf.num_layers_FCN
    ).to(device)
    model.load_state_dict(torch.load(final_model_path))
    print("Model loaded successfully.")

    # Valutazione del modello e estrazione delle feature
    features, labels, test_accuracy, test_cm = evaluate_model_and_extract_features(model, test_loader)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Confusion Matrix:\n{test_cm}")

    # Visualizzazione con UMAP
    umap_path = os.path.join(conf.test_dir, "feature_visualization_umap.png")
    visualize_umap(features, labels, umap_path)

if __name__ == "__main__":
    main()
