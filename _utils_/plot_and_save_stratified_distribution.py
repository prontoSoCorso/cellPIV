import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tabulate import tabulate
import time

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

# Funzione per caricare il dataset
def load_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    df['tSB'] = df['tSB'].replace('-', np.nan).astype(float)
    return df

# Funzione per calcolare statistiche basate su PN
def calculate_statistics_by_PN(df):
    pn_counts = df['PN'].value_counts().to_dict()
    blastocysts_by_PN = df[df['BLASTO NY'] == 1]['PN'].value_counts().to_dict()
    non_blastocysts_by_PN = df[df['BLASTO NY'] == 0]['PN'].value_counts().to_dict()
    
    return pn_counts, blastocysts_by_PN, non_blastocysts_by_PN

# Funzione per salvare il grafico migliorato
def save_improved_plot(pn_counts, blastocysts_by_PN, non_blastocysts_by_PN, output_path):
    # Creo set con tutte le categorie (evito di non visualizzare categorie con 0 blasto)
    all_pn_keys = set(pn_counts.keys()) | set(blastocysts_by_PN.keys()) | set(non_blastocysts_by_PN.keys())
    
    df_plot = pd.DataFrame({
        'PN': list(all_pn_keys),
        'Blastocysts': [blastocysts_by_PN.get(pn, 0) for pn in all_pn_keys],
        'Samples': [pn_counts.get(pn, 0) for pn in all_pn_keys],
        'Non-Blastocysts': [non_blastocysts_by_PN.get(pn, 0) for pn in all_pn_keys]
    })

    df_plot = df_plot.sort_values(by="Samples", ascending=False)  # Ordina per numero totale di campioni

    # Configura il plot
    plt.figure(figsize=(12, 7))
    bar_width = 0.3  # Larghezza delle barre
    x = np.arange(len(df_plot['PN']))  # Indici per le barre

    # Creazione delle barre affiancate
    plt.bar(x - bar_width, df_plot['Blastocysts'], width=bar_width, label="Blastocisti", color='green', alpha=0.7)
    plt.bar(x, df_plot['Samples'], width=bar_width, label="Campioni Totali", color='blue', alpha=0.7)
    plt.bar(x + bar_width, df_plot['Non-Blastocysts'], width=bar_width, label="Non-Blastocisti", color='red', alpha=0.7)

    # Aggiunta delle etichette sopra le barre
    for i in range(len(df_plot)):
        plt.text(x[i] - bar_width, df_plot['Blastocysts'].iloc[i] + 1, str(df_plot['Blastocysts'].iloc[i]), ha='center', fontsize=10)
        plt.text(x[i], df_plot['Samples'].iloc[i] + 1, str(df_plot['Samples'].iloc[i]), ha='center', fontsize=10)
        plt.text(x[i] + bar_width, df_plot['Non-Blastocysts'].iloc[i] + 1, str(df_plot['Non-Blastocysts'].iloc[i]), ha='center', fontsize=10)

    # Label e titolo
    plt.xticks(x, df_plot['PN'], rotation=45)
    plt.xlabel("PN")
    plt.ylabel("Numero di Campioni")
    plt.title("Distribuzione Campioni, Blastocisti e Non-Blastocisti per PN")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Salva il grafico
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

# Main
def main(input_csv_path, output_path):
    start_time = time.time()

    df = load_data(input_csv_path)
    pn_counts, blastocysts_by_PN, non_blastocysts_by_PN = calculate_statistics_by_PN(df)

    # Stampa tabella riepilogativa
    summary_table = []
    for pn in pn_counts.keys():
        summary_table.append([
            pn,
            blastocysts_by_PN.get(pn, 0),
            pn_counts.get(pn, 0),
            non_blastocysts_by_PN.get(pn, 0)
        ])

    print("Summary by PN:")
    print(tabulate(summary_table, headers=['PN', 'Blastocysts', 'Samples', 'Non-Blastocysts'], tablefmt='grid'))

    # Genera e salva il grafico migliorato
    save_improved_plot(pn_counts, blastocysts_by_PN, non_blastocysts_by_PN, output_path)

    print("Execution time:", time.time() - start_time, "seconds")
