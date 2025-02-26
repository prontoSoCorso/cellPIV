import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
from sklearn.preprocessing import StandardScaler
import os

def compute_UMAP(csv_path, days_to_consider, max_frames, output_path_base):
    # Caricamento del dataset
    data = pd.read_csv(csv_path)

    # Seleziono solo le colonne che contengono "value_" e poi filtro fino al numero di giorni da considerare
    features = data.filter(like="value_")
    features = features.iloc[:, :max_frames]
    labels = data["BLASTO NY"]

    # Standardizzazione delle feature
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Riduzione delle dimensioni con UMAP
    umap = UMAP(n_components=2, n_jobs=-1)  # Usa tutti i core disponibili
    features_2d = umap.fit_transform(features_scaled)

    # Creazione del DataFrame per la visualizzazione
    umap_df = pd.DataFrame(features_2d, columns=["Dim1", "Dim2"])
    umap_df["Label"] = labels

    # Visualizzazione
    plt.figure(figsize=(10, 8))

    # Colori per le classi
    colors = {0: "red", 1: "blue"}
    for label, color in colors.items():
        subset = umap_df[umap_df["Label"] == label]
        plt.scatter(subset["Dim1"], subset["Dim2"], c=color, label=f"Classe {label}", alpha=0.7)

    plt.title(f"Visualizzazione UMAP, {days_to_consider} Days")
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")
    plt.legend()
    plt.grid(True)

    # Salva immagine
    output_path = os.path.join(output_path_base, f"umap_{days_to_consider}Days.png")
    plt.savefig(output_path)
    print(f"Grafico salvato in: {output_path}")

    # Mostra grafico
    plt.show()  # Mostra il grafico interattivo



def compute_tSNE(csv_path, days_to_consider, max_frames, output_path_base):
    # Caricamento del dataset
    data = pd.read_csv(csv_path)

    # Seleziono solo le colonne che contengono "value_" e poi filtro fino al numero di giorni da considerare
    features = data.filter(like="value_")
    features = features.iloc[:, :max_frames]
    labels = data["BLASTO NY"]

    # Riduzione delle dimensioni con t-SNE
    tsne = TSNE(n_components=2, random_state=42, max_iter=300)  # max_iter per sklearn 1.5+
    features_2d = tsne.fit_transform(features)

    # Creazione del DataFrame per la visualizzazione
    tsne_df = pd.DataFrame(features_2d, columns=["Dim1", "Dim2"])
    tsne_df["Label"] = labels

    # Visualizzazione
    plt.figure(figsize=(10, 8))

    # Colori per le classi
    colors = {0: "red", 1: "blue"}
    for label, color in colors.items():
        subset = tsne_df[tsne_df["Label"] == label]
        plt.scatter(subset["Dim1"], subset["Dim2"], c=color, label=f"Classe {label}", alpha=0.7)

    plt.title(f"Visualizzazione t-SNE, {days_to_consider} Days")
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")
    plt.legend()
    plt.grid(True)

    # Salva immagine
    output_path = os.path.join(output_path_base, f"tSNE_{days_to_consider}Days.png")
    plt.savefig(output_path)
    print(f"Grafico salvato in: {output_path}")

    # Mostra grafico
    plt.show()  # Mostra il grafico interattivo

