'''
Creo il file csv normalizzato in output dal csv del test
'''

import os
import sys
import pandas as pd
import numpy as np

# Definisci i percorsi dei file
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_02_temporalData as conf


# Funzione per normalizzare per patient_id
def normalize_group_per_patient(group, check_number_last_column_changed):
    # Seleziona solo i valori > 0
    positive_values = group[group > 0]
    
    # Calcola il minimo e il massimo ignorando gli zeri
    min_val = np.min(positive_values)
    max_val = np.max(positive_values)

    # Verifica se il massimo è nelle ultime 10 colonne
    if max_val in group.iloc[:, -conf.n_last_colums_check_max:].values:
        check_number_last_column_changed[0] += 1
        # Imposta a zero le ultime 10 colonne della riga in cui ho trovato il massimo
        group.iloc[:, -conf.n_last_colums_check_max:] = 0

        # Ricalcola minimo e massimo ignorando gli zeri
        normalize_group_per_patient(group, check_number_last_column_changed)

    # Applica la normalizzazione Min-Max ignorando gli zeri
    normalized = (group - min_val) / (max_val - min_val)
    
    # Mantieni gli zeri invariati
    normalized[group == 0] = 0

    return normalized


def normalize_ALL(df_test):
    # Separo metadati e serie temporali
    time_series_test = df_test.iloc[:, 3:]

    # Calcolo il minimo e massimo del train da utilizzare anche per il validation
    min_test = time_series_test[time_series_test > 0].quantile(0.10).min()
    max_test = time_series_test[time_series_test > 0].quantile(0.90).max()

    # Normalizzo il train
    normalized_test = (time_series_test - min_test) / (max_test - min_test)
    normalized_test[time_series_test == 0] = 0

    return normalized_test


if __name__ == '__main__':
    csv_file_path = conf.output_csv_file_path_test
    output_csvNormalized_AllTest_file_path = conf.test_NormALL_data_path
    output_csvNormalized_Test_file_path = conf.test_data_path

    df_test = pd.read_csv(csv_file_path) #  output_csv_file_path_test

    if conf.normalization_type == "PerPatient":
        # Separare le prime 3 colonne (metadati) dalle serie temporali
        metadata = df_test.iloc[:, :3]

        # Applicare la normalizzazione per gruppo (patient_id)
        check_number_last_column_changed = [0]
        normalized_time_series = df_test.groupby('patient_id').apply(lambda x: normalize_group_per_patient(x.iloc[:, 3:], check_number_last_column_changed))
        print(f"Sono state cambiate le ultime colonne di {check_number_last_column_changed[0]} pazienti")

        # Rimuovi il MultiIndex risultante dal groupby
        normalized_time_series = normalized_time_series.reset_index(level=0, drop=True)

        # Concatenare i metadati con le serie temporali normalizzate
        normalized_df = pd.concat([metadata, normalized_time_series], axis=1)

        # Salva i dati normalizzati in un nuovo file CSV
        normalized_df.to_csv(output_csvNormalized_Test_file_path, index=False)
        print(f"I dati normalizzati per paziente sono stati salvati in {output_csvNormalized_Test_file_path}")

    elif conf.normalization_type == "TrainTest_ALL":
        # Normalizzazione train e validation
        normalized_test = normalize_ALL(df_test)

        # Salvo i dati normalizzati per train e validation
        train_metadata = df_test.iloc[:, :3]

        normalized_test_df = pd.concat([train_metadata, normalized_test], axis=1)

        normalized_test_df.to_csv(output_csvNormalized_AllTest_file_path, index=False)

        print(f"Dati normalizzati del test sono stati salvati in {output_csvNormalized_AllTest_file_path}")

    elif conf.normalization_type == "None":
        print("Nessuna normalizzazione fatta")

