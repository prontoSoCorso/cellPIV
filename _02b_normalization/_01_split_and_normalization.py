import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

# Definisci i percorsi dei file
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_02b_normalization as conf
from config import utils

# Caricamento del file CSV
def load_data():
    data = pd.read_csv(conf.csv_file_path)

    # Cambiare i booleani in "Config_02b_normalization" per definire il numero di giorni da prendere in considerazione
    if conf.Only5Days:
        data = data.iloc[:,:utils.num_frames_5Days]

    if conf.Only3Days:
        data = data.iloc[:,:utils.num_frames_3Days]

    return data


# Split dei dati in base a patient_id
def stratified_split(data):
    # Crea un DataFrame aggregato a livello di `patient_id`
    patient_info = data.groupby('patient_id').agg({
        'BLASTO NY': 'max'  # Etichetta prevalente per ogni paziente
    }).reset_index()

    # Split stratificato a livello di paziente
    train_patients, temp_patients = train_test_split(
        patient_info,
        test_size=0.3,
        stratify=patient_info['BLASTO NY'],
        random_state=conf.seed
    )
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.5,
        stratify=temp_patients['BLASTO NY'],
        random_state=conf.seed
    )

    # Filtra i dati originali per ottenere i subset
    train_data = data[data['patient_id'].isin(train_patients['patient_id'])]
    val_data = data[data['patient_id'].isin(val_patients['patient_id'])]
    test_data = data[data['patient_id'].isin(test_patients['patient_id'])]

    return train_data, val_data, test_data


# Normalizzazione del train con quantile normalization
def normalize_data(train_data, val_data, test_data):
    # Seleziona solo le colonne temporali da normalizzare
    temporal_columns = [col for col in train_data.columns if col.startswith("value_")]

    # Separo i dati temporali
    train_temporal = train_data[temporal_columns]
    val_temporal = val_data[temporal_columns]
    test_temporal = test_data[temporal_columns]

    # Calcolo il 10° e il 90° percentile per il train
    min_val = train_temporal[train_temporal > 0].quantile(0.10).min()
    max_val = train_temporal[train_temporal > 0].quantile(0.90).max()

    print("=====VALORI DI MIN E MAX PER NORMALIZZAZIONE=====")
    print(f"min_val = {min_val}, max_val = {max_val}")

    # Normalizzo il train
    train_normalized = (train_temporal - min_val) / (max_val - min_val)
    train_normalized[train_temporal == 0] = 0

    # Normalizzo il validation usando min e max del train
    val_normalized = (val_temporal - min_val) / (max_val - min_val)
    val_normalized[val_temporal == 0] = 0

    # Normalizzo il test usando min e max del train
    test_normalized = (test_temporal - min_val) / (max_val - min_val)
    test_normalized[test_temporal == 0] = 0

    # Sostituisco i dati normalizzati nei DataFrame originali
    train_data[temporal_columns] = train_normalized
    val_data[temporal_columns] = val_normalized
    test_data[temporal_columns] = test_normalized

    return train_data, val_data, test_data


# Salvataggio dei file normalizzati
def save_data(train_data, val_data, test_data):
    if conf.Only3Days:
        train_data.to_csv(conf.normalized_train_path_3Days, index=False)
        val_data.to_csv(conf.normalized_val_path_3Days, index=False)
        test_data.to_csv(conf.normalized_test_path_3Days, index=False)
        print("Dati salvati con successo nei file CSV a 3 giorni normalizzati.")
        print("=======================================================")
        print(f"Train salvato in: {conf.normalized_train_path_3Days}")
        print(f"Validation salvato in: {conf.normalized_val_path_3Days}")
        print(f"Test salvato in: {conf.normalized_test_path_3Days}")

    elif conf.Only5Days:
        train_data.to_csv(conf.normalized_train_path_5Days, index=False)
        val_data.to_csv(conf.normalized_val_path_5Days, index=False)
        test_data.to_csv(conf.normalized_test_path_5Days, index=False)
        print("Dati salvati con successo nei file CSV a 5 giorni normalizzati.")
        print("=======================================================")
        print(f"Train salvato in: {conf.normalized_train_path_5Days}")
        print(f"Validation salvato in: {conf.normalized_val_path_5Days}")
        print(f"Test salvato in: {conf.normalized_test_path_5Days}")
    
    else:
        train_data.to_csv(conf.normalized_train_path_7Days, index=False)
        val_data.to_csv(conf.normalized_val_path_7Days, index=False)
        test_data.to_csv(conf.normalized_test_path_7Days, index=False)
        print("Dati salvati con successo nei file CSV a 7 giorni normalizzati.")
        print("=======================================================")
        print(f"Train salvato in: {conf.normalized_train_path_7Days}")
        print(f"Validation salvato in: {conf.normalized_val_path_7Days}")
        print(f"Test salvato in: {conf.normalized_test_path_7Days}")


def main():
    # Carica i dati
    data = load_data()

    # Split stratificato
    train_data, val_data, test_data = stratified_split(data)

    # Normalizza i dati
    train_data, val_data, test_data = normalize_data(train_data, val_data, test_data)

    # Salva i dataset normalizzati
    save_data(train_data, val_data, test_data)


if __name__ == "__main__":
    main()

