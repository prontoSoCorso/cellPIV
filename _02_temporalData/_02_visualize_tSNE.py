import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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

from config import Config_02_temporalData as conf
from config import utils

# Imposta il backend grafico interattivo
try:
    matplotlib.use("TkAgg")  # O "Qt5Agg" se preferisci
except Exception as e:
    print(f"Impossibile impostare il backend interattivo: {e}")

# %% Caricamento del dataset
data = pd.read_csv(conf.final_csv_path)

# Seleziona le colonne delle feature
selected_days = "3Days"

if selected_days == "3Days":
    features = data.iloc[:, 3:utils.num_frames_3Days]  # Ignora le prime 3 colonne (patient_id, dish_well, BLASTO NY)
else:
    features = data.iloc[:, 3:utils.num_frames_7Days]  # Ignora le prime 3 colonne (patient_id, dish_well, BLASTO NY)

labels = data["BLASTO NY"]

# %% Riduzione delle dimensioni con t-SNE
tsne = TSNE(n_components=2, random_state=42, max_iter=300)  # max_iter per sklearn 1.5+
features_2d = tsne.fit_transform(features)

# %% Creazione del DataFrame per la visualizzazione
tsne_df = pd.DataFrame(features_2d, columns=["Dim1", "Dim2"])
tsne_df["Label"] = labels

# %% Visualizzazione
plt.figure(figsize=(10, 8))

# Colori per le classi
colors = {0: "red", 1: "blue"}
for label, color in colors.items():
    subset = tsne_df[tsne_df["Label"] == label]
    plt.scatter(subset["Dim1"], subset["Dim2"], c=color, label=f"Classe {label}", alpha=0.7)

plt.title(f"Visualizzazione t-SNE, {selected_days}")
plt.xlabel("Dimensione 1")
plt.ylabel("Dimensione 2")
plt.legend()
plt.grid(True)

# Salva immagine
output_path = os.path.join(parent_dir, f"tSNE_{selected_days}.png")
plt.savefig(output_path)
print(f"Grafico salvato in: {output_path}")

# Mostra grafico
plt.show()  # Mostra il grafico interattivo