import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
from sklearn.preprocessing import StandardScaler
import os
import mplcursors
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output

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
    """
    Computes a 2D UMAP projection of time series features and visualizes it using Plotly.

    Parameters:
    - csv_path (str): Path to the CSV file containing time series features and metadata.
    - days_to_consider (int): Number of days used to define the temporal window for analysis (used in the plot title).
    - max_frames (int): Maximum number of time steps (frames) to consider from the temporal features.
    - output_path_base (str): If non-empty, the static UMAP plot is saved as a PNG file in this path.
                              If empty, an interactive Dash app is launched with dynamic dropdown filters.

    Description:
    - Extracts and scales time series features.
    - Applies UMAP to project the high-dimensional data into 2D space.
    - Visualizes the result with separate colors for 'Blasto' and 'No Blasto' samples.
    - If output_path_base is not specified, launches an interactive Dash app allowing
      filtering by categorical metadata with dynamic dropdowns and live plot updates.
    """
    data = pd.read_csv(csv_path)

    all_time_features = data.filter(like="value_")
    features = all_time_features.iloc[:, :max_frames]
    labels = data["BLASTO NY"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    umap_model = UMAP(n_components=2, n_jobs=-1)
    features_2d = umap_model.fit_transform(features_scaled)

    # Create DataFrame with coordinates and metadata
    umap_df = pd.DataFrame(features_2d, columns=["Dim1", "Dim2"])
    umap_df["Label"] = labels.map({0: "No Blasto", 1: "Blasto"})  # Convert to categorical label

    # Automatically identify categorical features (excluding time features and BLASTO NY)
    excluded_columns = all_time_features.columns.union(["BLASTO NY"])  # Exclude dish_well
    categorical_features = [col for col in data.columns 
                           if col not in excluded_columns] # Limit to features with <50 unique values
    
    # Ensure categorical values are strings
    for feature in categorical_features:
        umap_df[feature] = data[feature].astype(str).fillna('missing')
    
    # Build a custom hovertemplate string
    cats = categorical_features # Lista di colonne da mostrare nell'hover
    hover_template = (
        "Dim1: %{x}<br>" +
        "Dim2: %{y}<br>" +
        "".join([f"{col}: %{{customdata[{i}]}}<br>" for i, col in enumerate(cats)]) +
        "<extra></extra>"
    )

    # Masks for the two groups
    blasto_mask = umap_df["Label"] == "Blasto"
    no_blasto_mask = umap_df["Label"] == "No Blasto"

    # Store original trace data for reset (All button)
    x_blasto_orig = umap_df.loc[blasto_mask, "Dim1"].tolist()
    y_blasto_orig = umap_df.loc[blasto_mask, "Dim2"].tolist()
    x_noblasto_orig = umap_df.loc[no_blasto_mask, "Dim1"].tolist()
    y_noblasto_orig = umap_df.loc[no_blasto_mask, "Dim2"].tolist()

    customdata_blasto_orig = umap_df.loc[blasto_mask, cats].to_numpy().tolist()
    customdata_noblasto_orig = umap_df.loc[no_blasto_mask, cats].to_numpy().tolist()

    # Create base plot with separate traces
    base_fig = go.Figure()
    base_fig.add_trace(go.Scatter(
        x=x_blasto_orig,
        y=y_blasto_orig,
        mode='markers',
        marker=dict(color='blue'),
        name='Blasto',
        customdata=customdata_blasto_orig,
        hovertemplate=hover_template
    ))
    base_fig.add_trace(go.Scatter(
        x=x_noblasto_orig,
        y=y_noblasto_orig,
        mode='markers',
        marker=dict(color='red'),
        name='No Blasto',
        customdata=customdata_noblasto_orig,
        hovertemplate=hover_template
    ))
    base_fig.update_layout(
        title=f"UMAP: {days_to_consider} Days",
        hovermode='closest',
        width=1200,
        height=800
    )

    # Se output_path_base non è vuoto, salva la figura come immagine PNG e la mostra
    if output_path_base:
        output_file = os.path.join(output_path_base, f"umap_{days_to_consider}Days.png")
        base_fig.write_image(output_file)  # Assicurarsi di avere "kaleido" installato per il salvataggio in PNG
        print(f"Grafico interattivo salvato in: {output_file}")
        base_fig.show()
    else:
        # Inizializza l'app Dash
        # Create dropdown menus—one for each categorical feature
        dropdowns = []
        for feature in categorical_features:
            unique_values = umap_df[feature].unique()
            # Inserisci sempre un'opzione "All" per visualizzare tutti i dati
            options = [{"label": f"All {feature}", "value": "All"}] + [
                {"label": val, "value": val} for val in unique_values
                ]
            
            dropdowns.append(html.Div([
                html.Label(feature),
                dcc.Dropdown(
                    id=f"dropdown-{feature}",
                    options=options,
                    value="All",      # Valore iniziale: nessun filtro
                    clearable=False,
                )
            ], style={"width": "20%", "display": "inline-block", "padding": "10px"}))

        
        # Definisci il layout dell'app con i dropdown e il grafico UMAP
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H1("Dash UMAP con dropdown dinamici"),
            html.Div(dropdowns, style={"display": "flex", "flexWrap": "wrap"}),
            dcc.Graph(id="umap-graph", figure=base_fig)
        ])

        # Callback che aggiorna la figura in base ai valori selezionati
        @app.callback(
            Output("umap-graph", "figure"),
            [Input(f"dropdown-{feature}", "value") for feature in categorical_features]
        )

        def update_figure(*selected_values):
            # Crea una copia del DataFrame per filtrare in base ai dropdown
            filtered_df = umap_df.copy()
            # Per ogni feature, se il valore selezionato non è "All", applica il filtro
            for feature, value in zip(categorical_features, selected_values):
                if value != "All":
                    filtered_df = filtered_df[filtered_df[feature] == value]
            
            # Separa i dati in base alla label
            filtered_blasto = filtered_df[filtered_df["Label"] == "Blasto"]
            filtered_noblasto = filtered_df[filtered_df["Label"] == "No Blasto"]

            # Ricrea la figura aggiornata
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_blasto["Dim1"].tolist(),
                y=filtered_blasto["Dim2"].tolist(),
                mode='markers',
                marker=dict(color='blue'),
                name='Blasto',
                customdata=filtered_blasto[cats].to_numpy().tolist(),
                hovertemplate=hover_template
            ))
            fig.add_trace(go.Scatter(
                x=filtered_noblasto["Dim1"].tolist(),
                y=filtered_noblasto["Dim2"].tolist(),
                mode='markers',
                marker=dict(color='red'),
                name='No Blasto',
                customdata=filtered_noblasto[cats].to_numpy().tolist(),
                hovertemplate=hover_template
            ))
            fig.update_layout(
                title=f"UMAP: {days_to_consider} Days",
                hovermode='closest',
                width=1200,
                height=800
            )
            return fig
        
        # Run dell'app
        app.run_server(debug=True)
    



if __name__ == "__main__":
    import sys
    # Individua la cartella 'cellPIV' come riferimento
    current_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_file_path)
    while os.path.basename(parent_dir) != "cellPIV":
        parent_dir = os.path.dirname(parent_dir)
    sys.path.append(parent_dir)
    from config import Config_02_temporalData as conf
    from config import utils

    day = 1
    max_frames = utils.num_frames_by_days(day)
    compute_UMAP_with_plotly(csv_path="/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/Farneback/FinalDataset.csv", 
                             days_to_consider=day, 
                             max_frames=max_frames, 
                             output_path_base="")