import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time


def visualize_normalized_data_single_pt(original_data, normalized_data, output_base, specific_patient_id=None, shift_x=0):
    """
    Visualizza e confronta i dati originali e normalizzati per un paziente selezionato casualmente.

    :param original_data: DataFrame con i dati originali.
    :param normalized_data: DataFrame con i dati normalizzati.
    :param current_dir: Directory in cui salvare l'immagine.
    """
    if specific_patient_id is None:
        # Filtra i pazienti con almeno 5 righe ma meno di 8
        valid_patients = original_data.groupby("patient_id").size()
        valid_patients = valid_patients[(valid_patients > 3) & (valid_patients < 7)].index.tolist()
        
        if not valid_patients:
            print("Nessun paziente soddisfa i criteri.")
            return
        
        patient_id_example = random.choice(valid_patients)
        print(f"Randomly Selected ID: {patient_id_example}")
    
    else:
        if specific_patient_id not in normalized_data["patient_id"]:
            print(f"Selected ID: {specific_patient_id}, not in data. Please, select the ID of a patient in the given subset")
            return
        else:
            print(f"Selected ID: {specific_patient_id}")
            patient_id_example = specific_patient_id
        
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
        # Legend with dish_well and PN type
        label = f"{row['dish_well']} ({row.get('merged_PN', 'PN sconosciuto')})"
        axes[0].plot(row[value_columns], alpha=0.6, label=label)
    
    axes[0].set_ylabel("Valore")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    
    # Grafico segnali normalizzati
    axes[1].set_title(f"Segnali Normalizzati")
    for _, row in normalized_signals.iterrows():
        label = f"{row['dish_well']} ({row.get('merged_PN', 'PN sconosciuto')})"
        axes[1].plot(row[value_columns], alpha=0.6, label=label)  
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
    print(f"Plot saved at the path: {output_file_path}")


def create_and_save_plots_mean_temp_data(train_data, val_data, test_data, seed, temporal_data_type, days_to_consider, temporal_columns=None, output_base=None, shift_x=0):
    """
    Carica i dati, separa i gruppi Blasto e No Blasto, e crea i grafici per train, validation e test.
    
    :param train_path: Percorso del file CSV del train set
    :param val_path: Percorso del file CSV del validation set
    :param test_path: Percorso del file CSV del test set
    :param output_dir: Directory in cui salvare i grafici
    :param seed: Seed usato per la randomizzazione
    :param temporal_data_type: Tipo di dati temporali
    """
    if output_base:
        os.makedirs(output_base, exist_ok=True)
    
    def separate_data(df_input):
        """Separa i dati in base alla colonna 'BLASTO NY'."""
        blasto = df_input[df_input['BLASTO NY'] == 1]
        no_blasto = df_input[df_input['BLASTO NY'] == 0]
        return blasto, no_blasto
    
    def create_plot(blasto, no_blasto, title):
        """Genera e salva il grafico delle medie e deviazioni standard per Blasto e No Blasto."""
        x = np.arange(shift_x, shift_x + len(temporal_columns))

        blasto_mean = blasto[temporal_columns].mean()
        blasto_std = blasto[temporal_columns].std()
        
        no_blasto_mean = no_blasto[temporal_columns].mean()
        no_blasto_std = no_blasto[temporal_columns].std()

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(x, blasto_mean, label='Blasto', color='blue')
        ax.fill_between(x, blasto_mean - blasto_std, blasto_mean + blasto_std, color='blue', alpha=0.2)

        ax.plot(x, no_blasto_mean, label='No Blasto', color='red')
        ax.fill_between(x, no_blasto_mean - no_blasto_std, no_blasto_mean + no_blasto_std, color='red', alpha=0.2)

        ax.set_title(title)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Optical Flow Metric')
        ax.legend()
        ax.grid(True)

        return fig, ax
    
    # Caricamento e creazione dei grafici per train, validation e test
    if output_base is None:
        for dataset_name, data in zip(["Train", "Validation", "Test"], [train_data, val_data, test_data]):
            blasto, no_blasto = separate_data(df_input=data)
            filename = f"mean_{dataset_name.lower()}_data_{temporal_data_type}_{str(days_to_consider)}Days_seed{seed}.png"
            fig, ax = create_plot(blasto=blasto, no_blasto=no_blasto, 
                              title=f"Media dei valori temporali - {dataset_name} Set - {str(days_to_consider)} Days")
            plt.show()
    else:
        output_folder = os.path.join(output_base, "examples")
        os.makedirs(output_folder, exist_ok=True)
        for dataset_name, data in zip(["Train", "Validation", "Test"], [train_data, val_data, test_data]):
            blasto, no_blasto = separate_data(df_input=data)
            filename = f"mean_{dataset_name.lower()}_data_{temporal_data_type}_{str(days_to_consider)}Days_seed{seed}.png"
            fig, ax = create_plot(blasto=blasto, no_blasto=no_blasto, 
                              title=f"Media dei valori temporali - {dataset_name} Set - {str(days_to_consider)} Days")
            fig.savefig(os.path.join(output_base, filename), dpi=500, bbox_inches='tight')
            plt.close(fig)
    
        print(f"Plots saved in the folder '{output_base}' with seed {seed} and temporal data type '{temporal_data_type}'")



def create_and_save_stratified_plots_mean_temp_data(train_merged, val_merged, test_merged, output_base, seed, temporal_data_type, days_to_consider, shift_x=0):
    """
    Creates stratified mean temporal plots for each PN category across datasets,
    with the first subplot showing the entire dataset.
    """
    output_folder = os.path.join(output_base, "examples_stratified")
    os.makedirs(output_folder, exist_ok=True)

    def create_stratified_plot(data, title, filename):
        temporal_columns = [col for col in data.columns if col.startswith("value_")]
        x = np.arange(shift_x, shift_x + len(temporal_columns))
        
        # Get unique PN categories
        pn_groups = ["Overall"] + sorted(data['merged_PN'].unique(), key=lambda x: str(x))

        # Create subplot grid
        n_cols = 3
        n_rows = int(np.ceil( (len(pn_groups)) / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        fig.suptitle(title, y=1.02, fontsize=14)
        
        if n_rows == 1:  # Handle single row case
            axs = [axs] if n_cols == 1 else axs

        for idx, pn in enumerate(pn_groups):
            ax = axs.flat[idx] if n_rows > 1 else axs[idx]
            
            if pn == "Overall":
                # Plot entire dataset
                blasto = data[data['BLASTO NY'] == 1]
                no_blasto = data[data['BLASTO NY'] == 0]
                group_data = data
            else:
                # Get data for this PN group
                group_data = data[data['merged_PN'] == pn]
                blasto = group_data[group_data['BLASTO NY'] == 1]
                no_blasto = group_data[group_data['BLASTO NY'] == 0]

            # Plot means with std
            if len(blasto) > 0:
                blasto_mean = blasto[temporal_columns].mean()
                blasto_std = blasto[temporal_columns].std()
                ax.plot(x, blasto_mean, label='Blasto', color='blue')
                ax.fill_between(x, blasto_mean - blasto_std, blasto_mean + blasto_std, color='blue', alpha=0.2)

            if len(no_blasto) > 0:
                no_blasto_mean = no_blasto[temporal_columns].mean()
                no_blasto_std = no_blasto[temporal_columns].std()
                ax.plot(x, no_blasto_mean, label='No Blasto', color='red')
                ax.fill_between(x, no_blasto_mean - no_blasto_std, no_blasto_mean + no_blasto_std, color='red', alpha=0.2)

            ax.set_title(f"{pn} (n={len(group_data)})", fontsize=10)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.grid(True)
            ax.legend()

        # Hide empty subplots
        for j in range(len(pn_groups), n_rows*n_cols):
            axs.flat[j].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight', dpi=300)
        plt.close()

    # Generate plots for each dataset
    for dataset_name, data in zip(["train", "val", "test"], [train_merged, val_merged, test_merged]):
        filename = f"stratified_mean_{dataset_name}_data_{temporal_data_type}_{days_to_consider}Days_seed{seed}.jpg"
        create_stratified_plot(
            data=data,
            title=f"Stratified Temporal Patterns - {dataset_name.capitalize()} Set ({days_to_consider} Days)",
            filename=filename
            )


if __name__ == "__main__":
    start_time = time.time()
    print("Didn't do anything")