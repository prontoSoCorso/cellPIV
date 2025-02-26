import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd


def visualize_normalized_data(original_data, normalized_data, output_base):
    """
    Visualizza e confronta i dati originali e normalizzati per un paziente selezionato casualmente.

    :param original_data: DataFrame con i dati originali.
    :param normalized_data: DataFrame con i dati normalizzati.
    :param current_dir: Directory in cui salvare l'immagine.
    """
    # Filtra i pazienti con almeno 5 righe ma meno di 8
    valid_patients = original_data.groupby("patient_id").size()
    valid_patients = valid_patients[(valid_patients > 3) & (valid_patients < 7)].index.tolist()
    
    if not valid_patients:
        print("Nessun paziente soddisfa i criteri.")
        return
    
    patient_id_example = random.choice(valid_patients)

    # Filtra solo i dati del paziente selezionato
    original_signals = original_data[original_data["patient_id"] == patient_id_example]
    normalized_signals = normalized_data[normalized_data["patient_id"] == patient_id_example]
    
    # Estrai le colonne dei valori
    value_columns = [col for col in original_data.columns if col.startswith("value_")]
    
    # Creazione cartella per salvare il grafico
    output_folder = os.path.join(output_base, "examples")
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, f"patient_{patient_id_example}.png")

    # Plot
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12, 8))
    fig.suptitle(f"Confronto Segnali Originali e Normalizzati - Paziente {patient_id_example}", fontsize=10, fontweight='bold')

    # Grafico segnali originali
    axes[0].set_title(f"Segnali Originali")
    for _, row in original_signals.iterrows():
        axes[0].plot(row[value_columns], alpha=0.6, label=row["dish_well"])  # Usa dish_well come etichetta
    axes[0].set_ylabel("Valore")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    
    # Grafico segnali normalizzati
    axes[1].set_title(f"Segnali Normalizzati")
    for _, row in normalized_signals.iterrows():
        axes[1].plot(row[value_columns], alpha=0.6, label=row["dish_well"])  # Usa dish_well come etichetta
    axes[1].set_xlabel("Time Point")
    axes[1].set_ylabel("Valore")
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)

    # Gestione degli xticks per evitare sovrapposizioni
    num_xticks = min(15, len(value_columns))  # Mostra max 20 tick
    xtick_positions = list(range(0, len(value_columns), len(value_columns) // num_xticks))
    xtick_labels = [value_columns[i] for i in xtick_positions]
    
    axes[1].set_xticks(xtick_positions)
    axes[1].set_xticklabels(xtick_labels, rotation=45, ha="right")
    
    plt.tight_layout()
    axes[0].grid()
    axes[1].grid()
    plt.savefig(output_file_path)
    plt.close()
    print(f"Grafico salvato in: {output_file_path}")





def create_and_save_plots(train_data, val_data, test_data, output_base, seed, temporal_data_type, days_to_consider):
    """
    Carica i dati, separa i gruppi Blasto e No Blasto, e crea i grafici per train, validation e test.
    
    :param train_path: Percorso del file CSV del train set
    :param val_path: Percorso del file CSV del validation set
    :param test_path: Percorso del file CSV del test set
    :param output_dir: Directory in cui salvare i grafici
    :param seed: Seed usato per la randomizzazione
    :param temporal_data_type: Tipo di dati temporali
    """
    os.makedirs(output_base, exist_ok=True)  # Crea la cartella se non esiste
    
    def separate_data(df_input):
        """Separa i dati in base alla colonna 'BLASTO NY'."""
        blasto = df_input[df_input['BLASTO NY'] == 1]
        no_blasto = df_input[df_input['BLASTO NY'] == 0]
        return blasto, no_blasto
    
    def create_plot(blasto, no_blasto, title, filename, output_folder):
        """Genera e salva il grafico delle medie e deviazioni standard per Blasto e No Blasto."""
        temporal_columns = [col for col in blasto.columns if col.startswith("value_")]
        x = np.arange(1, len(temporal_columns) + 1)

        blasto_mean = blasto[temporal_columns].mean()
        blasto_std = blasto[temporal_columns].std()
        
        no_blasto_mean = no_blasto[temporal_columns].mean()
        no_blasto_std = no_blasto[temporal_columns].std()
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(x, blasto_mean, label='Blasto', color='blue')
        plt.fill_between(x, blasto_mean - blasto_std, blasto_mean + blasto_std, color='blue', alpha=0.2)
        
        plt.plot(x, no_blasto_mean, label='No Blasto', color='red')
        plt.fill_between(x, no_blasto_mean - no_blasto_std, no_blasto_mean + no_blasto_std, color='red', alpha=0.2)
        
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Optical Flow Metric')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()
    
    # Caricamento e creazione dei grafici per train, validation e test
    output_folder = os.path.join(output_base, "examples")
    os.makedirs(output_folder, exist_ok=True)
        
    for dataset_name, data in zip(["Train", "Validation", "Test"], [train_data, val_data, test_data]):
        blasto, no_blasto = separate_data(df_input=data)
        filename = f"mean_{dataset_name.lower()}_data_{temporal_data_type}_{str(days_to_consider)}Days_seed{seed}.jpg"
        create_plot(blasto=blasto, no_blasto=no_blasto, title=f"Media dei valori temporali - {dataset_name} Set - {str(days_to_consider)} Days", filename=filename, output_folder=output_folder)
    
    print(f"Grafici salvati nella cartella '{output_base}' con seed {seed} e tipo dati '{temporal_data_type}'")

