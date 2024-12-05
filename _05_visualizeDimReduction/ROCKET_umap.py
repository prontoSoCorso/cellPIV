import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap
import torch
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf
from sktime.classification.kernel_based import RocketClassifier

# Funzione per caricare i dati
def load_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# Funzione per valutare il modello e estrarre le feature
def evaluate_model_and_extract_features(model, X, y):
    # Previsione
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  # Probabilità della classe positiva

    # Metriche
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    # Estrazione delle feature (trasformazione con ROCKET)
    features = model.steps[0][1].transform(X)  # Estraggo le feature tramite il trasformatore ROCKET
    return features, accuracy, cm

# Funzione per visualizzare UMAP
def visualize_umap(features, labels, save_path):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embeddings = reducer.fit_transform(features)

    plt.figure(figsize=(10, 8))
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
    # Specifica il numero di giorni desiderati
    selected_days = "5Days"

    # Ottieni i percorsi dal config
    _, _, test_path = conf.get_paths(selected_days)

    # Percorsi
    model_path = os.path.join(conf.test_dir, f"best_rocket_model_{selected_days}.pkl")  # Percorso del modello salvato
    test_data_path = test_path  # Percorso ai dati di test

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Carico i dati di test
    df_test = load_data(test_data_path)
    X_test = df_test.iloc[:, 3:].values  # Serie temporale
    y_test = df_test['BLASTO NY'].values  # Etichette

    # Carico il modello ROCKET
    print("Caricamento del modello ROCKET...")
    model = torch.load(model_path)  # Cambiato da joblib a torch.save/torch.load
    if not isinstance(model, Pipeline):
        raise ValueError("Il modello caricato non è un pipeline. Verifica il file salvato.")

    print("Modello caricato con successo.")

    # Valutazione ed estrazione delle feature
    print("Valutazione del modello e estrazione delle feature...")
    features, test_accuracy, test_cm = evaluate_model_and_extract_features(model, X_test, y_test)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Confusion Matrix:\n{test_cm}")

    # Visualizzazione con UMAP
    umap_path = os.path.join(conf.test_dir, "feature_visualization_umap_rocket.png")
    visualize_umap(features, y_test, umap_path)

if __name__ == "__main__":
    main()
