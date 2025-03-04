import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while os.path.basename(parent_dir) != "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf

# Importa la definizione del modello LSTMFCN
from _03_train._b_LSTMFCN import LSTMFCN  # Importa il modello

# Funzione principale
def main():
    # Specifica il numero di giorni desiderati
    days_to_consider = 1

    # Ottieni i percorsi dal config
    _, _, test_path = conf.get_paths(days_to_consider)
    
    # Carica il dataset di test
    df_test = pd.read_csv(test_path)
    
    # Prepara i dati di test
    X_test = torch.tensor(df_test.iloc[:, 3:].values, dtype=torch.float32).unsqueeze(-1)  # Aggiungo dimensione canale
    y_test = df_test['BLASTO NY'].values  # Target
    
    test_dataset = TensorDataset(X_test, torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size_FCN, shuffle=False)
    
    # Carica il modello LSTMFCN pre-addestrato
    model = LSTMFCN(
        lstm_size=conf.lstm_size_FCN,
        filter_sizes=conf.filter_sizes_FCN,
        kernel_sizes=conf.kernel_sizes_FCN,
        dropout=conf.dropout_FCN,
        num_layers=conf.num_layers_FCN
    ).to(conf.device)
    
    model_path = os.path.join(parent_dir, conf.test_dir, f"best_lstm_fcn_model_{days_to_consider}Days.pth")
    model.load_state_dict(torch.load(model_path, map_location=conf.device))
    model.eval()
    print(f"Modello LSTMFCN caricato da: {model_path}")
    
    # Estrai le feature dal modello
    features = []
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(conf.device)
            lstm_out, _ = model.lstm(X)
            lstm_out = lstm_out[:, -1, :]  # Ultimo stato dell'LSTM
            
            # Passaggio nei convoluzionali
            x_conv = torch.transpose(X, 1, 2)
            conv_out = model.conv1(x_conv)
            conv_out = model.conv2(conv_out)
            conv_out = model.conv3(conv_out)
            conv_out = model.global_pooling(conv_out).squeeze(-1)
            
            # Concatenazione delle feature
            combined_features = torch.cat((lstm_out, conv_out), dim=1)
            features.append(combined_features.cpu())

    features = torch.cat(features, dim=0).numpy()

    # Step 1: Standardizza le feature
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Step 2: Riduzione dimensionale con UMAP
    umap = UMAP(n_components=2, random_state=42)
    X_umap = umap.fit_transform(features_scaled)
    
    # Step 3: Creazione del DataFrame per il plot
    umap_df = pd.DataFrame(X_umap, columns=["Dim1", "Dim2"])
    umap_df["Label"] = y_test
    
    # Step 4: Visualizzazione
    plt.figure(figsize=(10, 8))
    colors = {0: "red", 1: "blue"}
    for label, color in colors.items():
        subset = umap_df[umap_df["Label"] == label]
        plt.scatter(subset["Dim1"], subset["Dim2"], c=color, label=f"Classe {label}", alpha=0.7)
    
    plt.title(f"Visualizzazione UMAP delle feature LSTMFCN, {days_to_consider}Days")
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")
    plt.legend()
    plt.grid(True)

    # Salvataggio del grafico
    current_dir = os.path.dirname(current_file_path)
    output_path = os.path.join(current_dir, f"umap_LSTMFCN_{days_to_consider}Days.png")
    plt.savefig(output_path)
    print(f"Grafico salvato in: {output_path}")

    # Mostro il grafico dopo averlo salvato, altrimenti se lo mostro prima si azzera la memoria e mi salva img bianca
    plt.show()
    

if __name__ == "__main__":
    main()
