import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
from sklearn.preprocessing import StandardScaler
import os
import mplcursors
import plotly.express as px

def compute_UMAP(csv_path, days_to_consider, max_frames, output_path_base):
    # Caricamento del dataset
    data = pd.read_csv(csv_path)

    # Seleziono solo le colonne che contengono "value_" e poi filtro fino al numero di giorni da considerare
    features = data.filter(like="value_")
    features = features.iloc[:, :max_frames]
    labels = data["BLASTO NY"]
    dish_well = data.get("dish_well", data.index.astype(str))

    # Standardizzazione delle feature
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Riduzione delle dimensioni con UMAP
    umap_model = UMAP(n_components=2, n_jobs=-1)  # Usa tutti i core disponibili
    features_2d = umap_model.fit_transform(features_scaled)

    # Creazione del DataFrame per la visualizzazione
    umap_df = pd.DataFrame(features_2d, columns=["Dim1", "Dim2"])
    umap_df["Label"] = labels
    umap_df["ID"] = dish_well

    # Plot setup
    plt.figure(figsize=(10, 8))
    colors = {0: "red", 1: "blue"}
    scatters = []
    
    # Create scatter plots and store references
    for label, color in colors.items():
        subset = umap_df[umap_df["Label"] == label]
        sc = plt.scatter(subset["Dim1"], subset["Dim2"], 
                        c=color, label=f"Classe {label}", alpha=0.7)
        scatters.append((sc, subset["ID"].tolist()))  # Store scatter and IDs

    # Add interactive hover functionality
    for sc, ids_scatter in scatters:
        cursor = mplcursors.cursor(sc)
        cursor.connect("add", lambda sel, ids=ids_scatter: sel.annotation.set_text(f"ID: {ids[sel.index]}"))

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




def compute_UMAP_with_plotly(csv_path, days_to_consider, max_frames, output_path_base):
    data = pd.read_csv(csv_path)

    all_time_features = data.filter(like="value_")
    features = all_time_features.iloc[:, :max_frames]
    labels = data["BLASTO NY"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    umap_model = UMAP(n_components=2, n_jobs=-1)
    features_2d = umap_model.fit_transform(features_scaled)

    umap_df = pd.DataFrame(features_2d, columns=["Dim1", "Dim2"])
    umap_df["Label"] = labels.map({0: "No Blasto", 1: "Blasto"})  # Convert to categorical label

    # Define all the other features to display in the umap (hover)
    hover_cols = [col for col in data.columns if col not in list(all_time_features.columns) + ["BLASTO NY"]]
    
    # Add these additional columns to the umap_df so they are available for Plotly's hover
    umap_df = pd.concat([umap_df, data[hover_cols].reset_index(drop=True)], axis=1)
    
    # Explicit color mapping with category ordering
    color_discrete_map = {
        "No Blasto": "red",
        "Blasto": "blue"
    }

    # Using Plotly Express to create an interactive scatter plot
    fig = px.scatter(umap_df, x="Dim1", y="Dim2", 
                     color="Label", 
                     color_discrete_map=color_discrete_map,
                     category_orders={"Label": ["No Blasto", "Blasto"]},
                     hover_data=hover_cols,
                     title=f"UMAP visualization, {days_to_consider} Days")
    
    # Salva la figura in formato HTML se necessario
    output_file = os.path.join(output_path_base, f"umap_{days_to_consider}Days.html")
    fig.write_html(output_file)
    print(f"Grafico interattivo salvato in: {output_file}")
    
    fig.show()





if __name__=="__main__":
    import webbrowser
    webbrowser.open('/home/phd2/Scrivania/CorsoRepo/cellPIV/_02_temporalData/dim_reduction_files/Farneback/umap_1Days.html')