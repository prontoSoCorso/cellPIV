import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from umap.umap_ import UMAP
from sklearn.preprocessing import StandardScaler

import os
import sys

# Rileva il percorso della cartella genitore, che sarà la stessa in cui ho il file da convertire
current_dir = os.path.dirname(os.path.abspath(__file__))

# Individua la cartella 'cellPIV' come riferimento
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while os.path.basename(parent_dir) != "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_02b_normalization as conf

# Imposta il backend grafico interattivo
try:
    matplotlib.use("TkAgg")  # O "Qt5Agg" se preferisci
except Exception as e:
    print(f"Impossibile impostare il backend interattivo: {e}")

# %% Caricamento del dataset
data = pd.read_csv(conf.normalized_train_path_7Days)

# Seleziona le colonne delle feature
features = data.iloc[:, 3:672+3]  # Ignora le prime 3 colonne (patient_id, dish_well, BLASTO NY)
labels = data["BLASTO NY"]

# Standardizzazione delle feature
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Riduzione delle dimensioni con UMAP
umap = UMAP(n_components=2, random_state=42)
features_2d = umap.fit_transform(features_scaled)

# Creazione del DataFrame per la visualizzazione
umap_df = pd.DataFrame(features_2d, columns=["Dim1", "Dim2"])
umap_df["Label"] = labels

# %% Visualizzazione
plt.figure(figsize=(10, 8))

# Colori per le classi
colors = {0: "red", 1: "blue"}
for label, color in colors.items():
    subset = umap_df[umap_df["Label"] == label]
    plt.scatter(subset["Dim1"], subset["Dim2"], c=color, label=f"Classe {label}", alpha=0.7)

plt.title("Visualizzazione UMAP")
plt.xlabel("Dimensione 1")
plt.ylabel("Dimensione 2")
plt.legend()
plt.grid(True)

# Mostra il grafico o salva l'immagine
try:
    plt.show()  # Mostra il grafico interattivo
except Exception as e:
    print(f"Impossibile mostrare il grafico interattivo: {e}")
    output_path = "/path/dove/salvare/umap_visualization.png"
    plt.savefig(output_path)
    print(f"Grafico salvato in: {output_path}")