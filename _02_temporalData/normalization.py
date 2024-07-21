'''
Normalization Strategies:
    Normalizzazione su tutto il dataset
        Vantaggi:
            Comparabilità globale: Facilita l'uso di modelli di machine learning, che spesso beneficiano di feature scalate uniformemente.
            Riduzione della varianza: Aiuta a stabilizzare i modelli predittivi, riducendo l'influenza degli outlier.

        Svantaggi:
            Diluzione delle differenze specifiche: Potrebbe nascondere variazioni importanti tra i wells di diversi pazienti.
            Rischio di overfitting: Se le caratteristiche uniche di un paziente sono rilevanti, questo approccio potrebbe non catturarle bene.

            

    Normalizzazione per singolo well
        Vantaggi:
            Preserva la variabilità intra-well: Mantiene le specificità di ogni well, che potrebbero essere cruciali per la predizione accurata.
            Riduzione della variabilità locale: Facilita l'analisi delle variazioni all'interno di ogni well, potenzialmente importanti per la formazione delle blastocisti.

        Svantaggi:
            Scarsa comparabilità tra wells: Potrebbe rendere difficile identificare pattern globali, necessitando di metodi di aggregazione più complessi.
            Rischio di sovradimensionamento: Troppa enfasi sulle caratteristiche individuali di un well potrebbe portare a modelli troppo specifici e meno generalizzabili.


    Normalizzazione per paziente
        Vantaggi:
            Equilibrio tra variabilità intra e inter-paziente: Mantiene le differenze significative tra pazienti, pur riducendo la varianza interna.
            Facilitazione del modello: Aiuta i modelli di machine learning a captare pattern rilevanti che sono consistenti tra i wells di un singolo paziente.

        Svantaggi:
            Variabilità interna meno evidente: Potrebbe mascherare variazioni all'interno dei wells di un paziente.
            Richiede dati consistenti: Questo approccio funziona meglio con un numero significativo di wells per paziente, altrimenti la normalizzazione potrebbe non essere robusta.



Decisione finale: NORMALIZZAZIONE PER PAZIENTE
    - Controlla la variabilità inter-individuale: Riduce l'impatto delle differenze tra pazienti, 
        facilitando l'identificazione di pattern rilevanti per la predizione.
    - Mantiene l'informazione intra-paziente: Conserva le specificità dei wells, che possono essere 
        cruciali per la formazione delle blastocisti.
    - Facilita l'apprendimento del modello: Modelli di machine learning spesso beneficiano di dati 
        normalizzati a livello di gruppo, migliorando la stabilità e la generalizzabilità.
'''
import sys
import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Definisci i percorsi dei file
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_02_temporalData as conf

def load_pickled_files(directory):
    files = os.listdir(directory)
    data = {}
    for file in files:
        if file.endswith(".pkl"):
            with open(os.path.join(directory, file), 'rb') as f:
                data[file[:-4]] = pickle.load(f)
    return data

def plot_signals(patient_id, original_data, normalized_data, dish_wells):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot original data
    for data, dish_well in zip(original_data, dish_wells):
        axes[0].plot(data, marker='o', label=dish_well)
    axes[0].set_title(f'Original Signals for Patient {patient_id}')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Signal Value')
    axes[0].legend()

    # Plot normalized data
    for data, dish_well in zip(normalized_data, dish_wells):
        axes[1].plot(data, marker='o', label=dish_well)
    axes[1].set_title(f'Normalized Signals for Patient {patient_id}')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Normalized Signal Value')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    csv_file_path = conf.csv_file_path
    output_csv_file_path = conf.output_csv_file_path

    # Carica i dati dal file CSV
    df_csv = pd.read_csv(csv_file_path)

    # Mantieni solo le colonne di interesse
    df_csv = df_csv[['patient_id', 'dish_well', 'BLASTO NY']]

    # Ottieni il percorso della cartella dello script corrente
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Carica i file dalla cartella specificata
    loaded_data = load_pickled_files(current_directory)
    if conf.temporalDataType not in loaded_data:
        raise KeyError(f"Key {conf.temporalDataType} not found in loaded data")

    print(loaded_data.keys())  # Mostra i nomi dei file caricati

    # Estrazione dei dati relativi a sum_mean_mag
    sum_mean_mag_dict = loaded_data[conf.temporalDataType]

    # Trasforma il dizionario in un DataFrame
    df_temporal = pd.DataFrame.from_dict(sum_mean_mag_dict, orient='index').reset_index()
    df_temporal.columns = ['dish_well'] + [f'value_{i+1}' for i in range(df_temporal.shape[1] - 1)]

    # Unisci i dati del file CSV con i dati temporali
    df_merged = pd.merge(df_csv, df_temporal, on='dish_well', how='inner')

    # Normalizza i dati per paziente
    normalized_values = []
    original_values = []
    dish_wells_list = []

    for patient_id, group in df_merged.groupby('patient_id'):
        temporal_data = group.iloc[:, 3:].values  # Ottieni solo le colonne dei valori temporali
        original_values.append(temporal_data)
        dish_wells_list.append(group['dish_well'].values)
        min_val = np.min(temporal_data)
        max_val = np.max(temporal_data)
        normalized_data = (temporal_data - min_val) / (max_val - min_val)
        normalized_values.append(normalized_data)

    # Sovrascrivi i valori normalizzati nel DataFrame
    normalized_values = np.vstack(normalized_values)
    df_merged.iloc[:, 3:] = normalized_values

    # Salva il risultato in un nuovo file csv
    df_merged.to_csv(output_csv_file_path, index=False)

    print(f"I dati normalizzati sono stati salvati in {output_csv_file_path}")

    # Plotting
    # Seleziona un esempio di paziente per il confronto
    example_patient_id = 55
    example_index = df_merged['patient_id'] == example_patient_id
    example_group = df_merged[example_index]
    example_dish_wells = dish_wells_list[np.where(df_merged['patient_id'].unique() == example_patient_id)[0][0]]
    example_original_data = original_values[np.where(df_merged['patient_id'].unique() == example_patient_id)[0][0]]
    example_normalized_data = example_group.iloc[:, 3:].values

    plot_signals(example_patient_id, example_original_data, example_normalized_data, example_dish_wells)
