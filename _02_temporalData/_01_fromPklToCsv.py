## Il file che ottengo con il flusso ottico è un file pickle. In questo script voglio creare il csv con
# le serie temporali e gli identificativi dei video, facendo un taglio al settimo giorno e facendo il padding
# delle serie temporali minori di 7 giorni ##

import pickle
import pandas as pd
import os
import numpy as np

def fromPickleToCsv(path_pkl, output_temporal_csv_path, num_frames_MaxDays):
    with open(path_pkl, 'rb') as f:
        data = pickle.load(f)

    # Processa i dati
    processed_data = []  # Lista per raccogliere le righe

    for key, value in data.items():
        if len(value) > num_frames_MaxDays:
            # Tronca le serie troppo lunghe
            value = value[:num_frames_MaxDays]
        elif len(value) < num_frames_MaxDays:
            # Applica padding con zeri
            padding_length = num_frames_MaxDays - len(value)
            value = np.pad(value, (0, padding_length), 'constant', constant_values=0)

        # Aggiungi una riga con chiave + valori
        processed_data.append([key] + list(value))

    columns = ['dish_well'] + [f"time_{i}" for i in range(num_frames_MaxDays)]  # Intestazioni
    df = pd.DataFrame(processed_data, columns=columns)

    # Salva il DataFrame in formato .csv
    df.to_csv(output_temporal_csv_path, index=False)




def create_final_csv(input_temporal_csv_path, original_csv_path, output_final_csv_path):
    # Leggi i file CSV
    temporal_data = pd.read_csv(input_temporal_csv_path)
    labels_data = pd.read_csv(original_csv_path)

    # Controlla ed elimina duplicati
    duplicates_temporal = temporal_data[temporal_data.duplicated(subset=["dish_well"], keep=False)]
    if not duplicates_temporal.empty:
        print(f"Duplicati trovati in temporal_data:\n{duplicates_temporal}")

    duplicates_labels = labels_data[labels_data.duplicated(subset=["dish_well"], keep=False)]
    if not duplicates_labels.empty:
        print(f"Duplicati trovati in labels_data:\n{duplicates_labels}")

    # Elimina duplicati
    temporal_data = temporal_data.drop_duplicates(subset=["dish_well"], keep="first")
    labels_data = labels_data.drop_duplicates(subset=["dish_well"], keep="first")
    print(f"\n========== Duplicati Eliminati! ==========\n")

    '''
    Duplicati trovati in labels_data:
        4565    640     D2019.03.13_S02233_I0141_D     5  D2019.03.13_S02233_I0141_D_5      41         normo  ...         -  23.2415225016666  22.9910616672222    -         -  2
        4568    640     D2019.03.13_S02233_I0141_D     5  D2019.03.13_S02233_I0141_D_5      41         normo  ...         -  23.2415225016666  22.9910616672222    -         -  2
    '''

    # Handle NaN/empty values and standardize annotations
    labels_data['eup_aneup'] = labels_data['eup_aneup'].fillna('non analizzato')
    
    # Standardize capitalization
    def standardize_eup_aneup(label):
        label = str(label).strip().lower()
        if label == 'euploide':
            return 'Euploide'
        elif label == 'aneuploide':
            return 'Aneuploide'
        elif label in ['non analizzato', 'nan', '']:
            return 'not_analyzed'
        else:
            return label.capitalize()
    
    labels_data['eup_aneup'] = labels_data['eup_aneup'].apply(standardize_eup_aneup)

    # Mantieni solo le colonne necessarie
    columns_to_keep = ["dish_well"] + [col for col in temporal_data.columns if col.startswith("time_")]
    temporal_data = temporal_data[columns_to_keep]

    # Unisci i due file usando la colonna "dish_well"
    merged_data = pd.merge(labels_data, temporal_data, on="dish_well", how="inner")

    # Seleziona solo le colonne richieste
    meta_colums = ["patient_id", "dish_well", "BLASTO NY", "eup_aneup", "PN", "maternal age"]
    columns_to_keep_final = meta_colums + [col for col in temporal_data.columns if col.startswith("time_")]
    merged_data = merged_data[columns_to_keep_final]

    # Rinomina le colonne temporali in value_1, value_2, ..., value_N
    num_temporal_columns = len(columns_to_keep_final) - len(meta_colums)
    new_column_names = meta_colums + [f"value_{i+1}" for i in range(num_temporal_columns)]
    merged_data.columns = new_column_names

    # Salva il nuovo file CSV
    try:
        merged_data.to_csv(output_final_csv_path, index=False)
        print(f"File CSV unito salvato in: {output_final_csv_path}")
    except Exception as e:
        print(f"Errore durante il salvataggio del file: {e}")
