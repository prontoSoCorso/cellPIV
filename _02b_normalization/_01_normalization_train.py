import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Definisci i percorsi dei file
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_02b_normalization as conf


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


def normalize_train_val(df_train, df_val):
    # Separo metadati e serie temporali
    time_series_train = df_train.iloc[:, 3:]
    time_series_val = df_val.iloc[:, 3:]

    # Calcolo il minimo e massimo del train da utilizzare anche per il validation
    min_val = time_series_train[time_series_train > 0].quantile(0.10).min()
    max_val = time_series_train[time_series_train > 0].quantile(0.90).max()

    # Normalizzo il train
    normalized_train = (time_series_train - min_val) / (max_val - min_val)
    normalized_train[time_series_train == 0] = 0

    # Uso il min e max del train per normalizzare il validation
    normalized_val = (time_series_val - min_val) / (max_val - min_val)
    normalized_val[time_series_val == 0] = 0

    # Uso min e max del train per chiamare funzione che normalizza test
    print("=====VALORI DI MIN E MAX PER NORMALIZZARE TRAIN=====")
    print(f"min val = {min_val}")
    print(f"max val = {max_val}")
    normalization_test.main(min_val=min_val, max_val=max_val)

    return normalized_train, normalized_val


if __name__ == '__main__':
    csv_file_path = conf.csv_file_path
    output_csv_file_path = conf.output_csv_file_path
    output_csv_normalized_file_path = conf.output_csvNormalized_file_path
    output_csvNormalized_AllTrain_file_path = conf.output_csvNormalized_AllTrain_file_path
    output_csvNormalized_AllVal_file_path = conf.output_csvNormalized_AllVal_file_path


    if conf.do_only_normalization:
        df_merged = pd.read_csv(output_csv_file_path) #  output_csv_file_path / output_csv_file_path_test
    
    else:
        # PARTE DI CREAZIONE DEL FILE CSV A PARTIRE DA UN FILE DICT SELEZIONATO

        # Carico i dati dal file CSV
        df_csv = pd.read_csv(csv_file_path)

        # Mantengo solo le colonne di interesse
        df_csv = df_csv[['patient_id', 'dish_well', 'BLASTO NY']]

        # Ottieni il percorso della cartella dello script corrente
        current_directory = os.path.dirname(os.path.abspath(__file__))

        # Carica i file dalla cartella specificata
        loaded_data = load_pickled_files(current_directory)
        if conf.temporalDataType not in loaded_data:
            raise KeyError(f"Key {conf.temporalDataType} not found in loaded data")
        print(loaded_data.keys())  # Mostra i nomi dei file caricati

        # Estrazione dei dati relativi a sum_mean_mag (o altra misura)
        choosen_dict = loaded_data[conf.temporalDataType]
        print(f"E' stata selezionata la misura: {conf.temporalDataType}")

        # Trasforma il dizionario in un DataFrame
        df_temporal = pd.DataFrame.from_dict(choosen_dict, orient='index').reset_index()
        df_temporal.columns = ['dish_well'] + [f'value_{i+1}' for i in range(df_temporal.shape[1] - 1)]

        # Unisci i dati del file CSV con i dati temporali
        df_merged = pd.merge(df_csv, df_temporal, on='dish_well', how='inner')

        # Rimuovi le righe che contengono valori mancanti
        df_merged.dropna(inplace=True)

        # Salva i dati originali in un nuovo file CSV
        df_merged.to_csv(output_csv_file_path, index=False)
        print(f"I dati originali sono stati salvati in {output_csv_file_path}")

    

    # INIZIO DELLA PARTE DI NORMALIZZAZIONE

    if conf.normalization_type == "PerPatient":
        # Separare le prime 3 colonne (metadati) dalle serie temporali
        metadata = df_merged.iloc[:, :3]

        # Applicare la normalizzazione per gruppo (patient_id)
        check_number_last_column_changed = [0]
        normalized_time_series = df_merged.groupby('patient_id').apply(lambda x: normalize_group_per_patient(x.iloc[:, 3:], check_number_last_column_changed))
        print(f"Sono state cambiate le ultime colonne di {check_number_last_column_changed[0]} pazienti")

        # Rimuovi il MultiIndex risultante dal groupby
        normalized_time_series = normalized_time_series.reset_index(level=0, drop=True)

        # Concatenare i metadati con le serie temporali normalizzate
        normalized_df = pd.concat([metadata, normalized_time_series], axis=1)

        # Salva i dati normalizzati in un nuovo file CSV
        normalized_df.to_csv(output_csv_normalized_file_path, index=False)
        print(f"I dati normalizzati per paziente sono stati salvati in {output_csv_normalized_file_path}")

    elif conf.normalization_type == "TrainTest_ALL":
        # Creo DataFrame in modo da avere ogni paziente con una sola riga
        patient_info = df_merged.groupby('patient_id').agg({
            'BLASTO NY': 'max'  # Prendo la classe "BLASTO NY" prevalente per ogni paziente
        }).reset_index()

        # Divido in train e validation mantenendo le proporzioni di BLASTO NY
        train_patients, val_patients = train_test_split(
            patient_info, test_size=0.3, random_state=conf.seed_everything(conf.seed), stratify=patient_info['BLASTO NY']
        )

        # Creo i DataFrame di train e validation usando il "patient" come criterio
        df_train = df_merged[df_merged['patient_id'].isin(train_patients['patient_id'])]
        df_val = df_merged[df_merged['patient_id'].isin(val_patients['patient_id'])]

        # Normalizzazione train e validation
        normalized_train, normalized_val = normalize_train_val(df_train, df_val)

        # Salvo i dati normalizzati per train e validation
        train_metadata = df_train.iloc[:, :3]
        val_metadata = df_val.iloc[:, :3]

        normalized_train_df = pd.concat([train_metadata, normalized_train], axis=1)
        normalized_val_df = pd.concat([val_metadata, normalized_val], axis=1)

        normalized_train_df.to_csv(output_csvNormalized_AllTrain_file_path, index=False)
        normalized_val_df.to_csv(output_csvNormalized_AllVal_file_path, index=False)

        print(f"Dati normalizzati train salvati in {output_csvNormalized_AllTrain_file_path}")
        print(f"Dati normalizzati validation salvati in {output_csvNormalized_AllVal_file_path}")

    elif conf.normalization_type == "None":
        print("Nessuna normalizzazione fatta")

