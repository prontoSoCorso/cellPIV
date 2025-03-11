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

from _03_train._c_ConvTranUtils import CustomDataset, load_my_data
from _99_ConvTranModel.model import model_factory
from _99_ConvTranModel.utils import load_model
from config import Config_03_train as conf

# Funzione principale
def main():
    # Specifica il numero di giorni desiderati
    days_to_consider = 1

    # Ottieni i percorsi dal config
    _, _, test_path = conf.get_paths(days_to_consider)
    
    # Carica il dataset di test
    print("Caricamento dati di test...")
    data_test = pd.read_csv(test_path)
    X_test = data_test.iloc[:, 3:].values.reshape(data_test.shape[0], 1, -1)  # Reshape per ConvTran
    y_test = data_test['BLASTO NY'].values  # Etichette

    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)

    # Aggiungi numero di etichette uniche
    conf.num_labels = len(set(test_loader.dataset.labels))
    conf.Data_shape = (test_loader.dataset[0][0].shape[0], test_loader.dataset[0][0].shape[1])

    # Carica il modello ConvTran pre-addestrato
    print("Caricamento modello ConvTran...")
    model = model_factory(conf)
    model_path = os.path.join(parent_dir, conf.test_dir, f"best_convtran_model_{days_to_consider}Days.pkl")
    model = load_model(model, model_path)
    model.eval().to(conf.device)
    print(f"Modello ConvTran caricato da: {model_path}")
    
    # Estrai le feature dal modello
    print("Estrazione delle feature...")
    features = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = torch.tensor(inputs, dtype=torch.float32).to(conf.device)
            outputs = model(inputs)  # Passa attraverso il modello
            features.append(outputs.cpu())

    features = torch.cat(features, dim=0).numpy()

    # Step 1: Standardizza le feature
    print("Standardizzazione delle feature...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Step 2: Riduzione dimensionale con UMAP
    print("Applicazione di UMAP...")
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
    
    plt.title(f"Visualizzazione UMAP delle feature ConvTran, {days_to_consider}Days")
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")
    plt.legend()
    plt.grid(True)

    # Salvo plot
    current_dir = os.path.dirname(current_file_path)
    output_path = os.path.join(current_dir, "umap_ConvTran_" + str(days_to_consider) + "Days")
    plt.savefig(output_path)
    print(f"Grafico salvato in: {output_path}")

    # Mostro il grafico dopo averlo salvato
    plt.show()

if __name__ == "__main__":
    main()
