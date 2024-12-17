import os
import sys
import pandas as pd
from sktime.transformations.panel.rocket import Rocket  # Importa Rocket come trasformatore
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while os.path.basename(parent_dir) != "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf

def main():
    # Specifica il numero di giorni desiderati
    selected_days = "7Days"

    # Ottieni i percorsi dei dataset
    _, _, test_path = conf.get_paths(selected_days)

    # Carica i dati di test
    df_test = pd.read_csv(test_path)
    X_test = df_test.iloc[:, 3:].values  # Le colonne da 3 in poi contengono la serie temporale
    y_test = df_test['BLASTO NY'].values  # Colonna target

    # Prepara X_test nel formato richiesto da Rocket (3D)
    n_samples = X_test.shape[0]  # Numero di campioni
    n_time_steps = X_test.shape[1]  # Numero di time steps per campione
    X_test_reshaped = X_test.reshape(n_samples, 1, n_time_steps)

    # Step 1: Trasformazione delle feature con Rocket
    rocket = Rocket(num_kernels=10000, random_state=42)  # Usa lo stesso numero di kernel usato per l'allenamento
    rocket.fit(X_test_reshaped)  # Rocket calcola i kernel

    X_test_transformed = rocket.transform(X_test_reshaped)  # Ottieni le feature trasformate

    # Step 2: Standardizza le feature trasformate
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test_transformed)

    # Step 3: Riduzione dimensionale con UMAP
    umap = UMAP(n_components=2, random_state=42)
    X_umap = umap.fit_transform(X_test_scaled)

    # Step 4: Creazione di un DataFrame per la visualizzazione
    umap_df = pd.DataFrame(X_umap, columns=["Dim1", "Dim2"])
    umap_df["Label"] = y_test  # Ora le dimensioni corrispondono

    # Step 5: Visualizzazione
    plt.figure(figsize=(10, 8))
    colors = {0: "red", 1: "blue"}
    for label, color in colors.items():
        subset = umap_df[umap_df["Label"] == label]
        plt.scatter(subset["Dim1"], subset["Dim2"], c=color, label=f"Classe {label}", alpha=0.7)

    plt.title(f"Visualizzazione UMAP delle feature ROCKET, {selected_days}")
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")
    plt.legend()
    plt.grid(True)

    # Salvo plot
    current_dir = os.path.dirname(current_file_path)
    output_path = os.path.join(current_dir, "umap_ROCKET_"+selected_days)
    plt.savefig(output_path)
    print(f"Grafico salvato in: {output_path}")

    # Mostro il grafico dopo averlo salvato, altrimenti se lo mostro prima si azzera la memoria e mi salva img bianca
    plt.show()


if __name__ == "__main__":
    main()
